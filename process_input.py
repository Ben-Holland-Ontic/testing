import os
import json
import logging
import csv
import hashlib
import tempfile
import time
import zipfile
import tarfile
import subprocess
import shutil
import threading

import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont
import pdfplumber
import cv2

# configurable PDF render DPI (default 300)
PIPELINE_PDF_DPI = int(os.environ.get("PIPELINE_PDF_DPI", "300").strip())
# DPI used to normalize PDF renders before OCR
PIPELINE_PDF_TARGET_DPI = int(os.environ.get("PIPELINE_PDF_TARGET_DPI", "144").strip())

# Number of pages to render concurrently within a single PDF (thread pool).
# Independent from INGEST_WORKERS, which parallelizes across documents.
PIPELINE_PDF_RENDER_THREADS = int(os.environ.get("PIPELINE_PDF_RENDER_THREADS", "4").strip())

# How many documents the streaming ingest path keeps rendered ahead of the one
# currently being processed. This — not INGEST_WORKERS — is the knob that bounds
# ingest memory: resident page images ~= (prefetch + 1) documents, independent
# of corpus size. INGEST_WORKERS only applies to the legacy eager path
# (process_input_local) which materializes the whole corpus.
INGEST_PREFETCH_DOCS = int(os.environ.get("INGEST_PREFETCH_DOCS", "2").strip())

# Per-page render timeout. Deliberately separate from PIPELINE_PAGE_EXTRACT_TIMEOUT_SEC
# (which bounds OCR/extraction work in pipeline.py and defaults to 900s).
# Rendering is rasterization + deskew + classify and takes seconds on an idle
# machine — but on a CPU-only server the render threads compete with N OCR
# worker processes saturating every core, and renders legitimately take
# minutes. The old 45s default caused mass data loss under that contention:
# pages timed out wholesale, became UNSUPPORTED placeholders, and whole
# documents were skipped as "not_extracted" (observed: 18k of 18.1k pages
# skipped on an 8k-document CPU-server run). A timed-out page is dropped from
# the run entirely, so this timeout must be generous — it exists to catch
# genuinely hung renders, not slow ones.
PIPELINE_PDF_RENDER_TIMEOUT_SEC = int(os.environ.get("PIPELINE_PDF_RENDER_TIMEOUT_SEC", "300").strip())

# Consecutive per-document render timeouts before the thread pool is recycled.
# Higher than the GPU health check's N=2 latch on purpose: a render timeout is a
# noisier signal than a GPU health-check failure (a single complex/huge page can
# legitimately be slow without the pool itself being unhealthy). N=2 would recycle
# on normal page-content variance; N=5 means something is actually wrong with
# the pool, not just an unlucky page.
PIPELINE_PDF_RENDER_RECYCLE_THRESHOLD = int(os.environ.get("PIPELINE_PDF_RENDER_RECYCLE_THRESHOLD", "5").strip())

# pdfplumber.open() has no native timeout; wrap it in a thread with a hard ceiling.
# Default 300s (5 min) catches genuinely stuck/corrupted PDFs without penalising
# normal multi-page engineering documents.
PIPELINE_PDF_OPEN_TIMEOUT_SEC = int(os.environ.get("PIPELINE_PDF_OPEN_TIMEOUT_SEC", "300").strip())

# DPI-derived constants (A4 dimensions at the configured DPI)
_A4_WIDTH_PX = int(210 * PIPELINE_PDF_DPI / 25.4)
_A4_HEIGHT_PX = int(297 * PIPELINE_PDF_DPI / 25.4)

# SCAN CLASSIFIER CONFIGURATION
# Feature flag: when disabled, PDF path reverts to legacy _normalize_pdf_page_for_ocr only.
SCAN_CLASSIFIER_ENABLED = os.environ.get("PIPELINE_SCAN_CLASSIFIER", "1").strip() == "1"
# Reconstructed: fragmentation ratio above which a page is judged to be scanline-decomposed.
SCAN_FRAG_THRESHOLD = float(os.environ.get("SCAN_FRAG_THRESHOLD", "4.0").strip())
# Contaminated: minimum fraction of small blobs (relative to CC count) to count as speckle.
SCAN_SMALL_BLOB_THRESHOLD = float(os.environ.get("SCAN_SMALL_BLOB_THRESHOLD", "0.15").strip())
# Contaminated: minimum largest-CC area share (relative to total ink) proving dominant structure remains.
SCAN_MAX_CC_AREA_THRESHOLD = float(os.environ.get("SCAN_MAX_CC_AREA_THRESHOLD", "0.30").strip())
# Degraded: stroke-width coefficient-of-variation lower bound; quantisation/pixelation inflates
# stroke-width spread well above clean scans (which cluster tight around a single stroke width).
SCAN_STROKE_CV_HIGH = float(os.environ.get("SCAN_STROKE_CV_HIGH", "0.40").strip())
# Degraded: edge density lower bound (fraction of Canny edge pixels) indicating stairstep artefacts.
SCAN_EDGE_DENSITY_HIGH = float(os.environ.get("SCAN_EDGE_DENSITY_HIGH", "0.06").strip())
# Confidence below which a candidate branch is demoted to CG-Normal.
SCAN_CONFIDENCE_FALLBACK = float(os.environ.get("SCAN_CONFIDENCE_FALLBACK", "0.5").strip())
# Universal deskew: maximum |angle| in degrees we will try to correct; steeper is treated as intentional.
SCAN_DESKEW_MAX_ANGLE = float(os.environ.get("SCAN_DESKEW_MAX_ANGLE", "5.0").strip())
# Quality gate: when enabled, reject a branch preprocessor's output if it regresses severely
# vs the classifier's view of the input (ink coverage or CC density out of band) and fall
# back to CG-Normal. 0 disables entirely (pure classifier routing).
SCAN_QUALITY_GATE_ENABLED = os.environ.get("PIPELINE_SCAN_QUALITY_GATE", "1").strip() == "1"
# Ink coverage: output ink ratio must stay within [LO, HI] times the input ink ratio.
SCAN_QUALITY_INK_LO = float(os.environ.get("SCAN_QUALITY_INK_LO", "0.30").strip())
SCAN_QUALITY_INK_HI = float(os.environ.get("SCAN_QUALITY_INK_HI", "3.00").strip())
# CC density (components per pixel, resolution-invariant): output density must not explode
# above CC_HI * input density (fragmentation) and — for the reconstructed branch only — must
# not collapse below CC_LO_RECON * input density (over-merge).
SCAN_QUALITY_CC_HI = float(os.environ.get("SCAN_QUALITY_CC_HI", "2.00").strip())
SCAN_QUALITY_CC_LO_RECON = float(os.environ.get("SCAN_QUALITY_CC_LO_RECON", "0.10").strip())

# DATA CLASSES FOR IN-MEMORY PIPELINE

@dataclass
class PageData:
    """single page"""
    page_id: str
    document_id: str
    batch_id: str
    page_number: int
    image: np.ndarray 
    width: int
    height: int
    storage_path: str  
    patches: int = 60
    status: str = "PENDING"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentData:
    """document"""
    document_id: str
    batch_id: str
    source_path: str
    file_size: int
    status: str = "PENDING"
    pages: List[PageData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchData:
    """batch"""
    batch_id: str
    source_path: str
    received_by: str
    status: str = "PENDING"
    documents: List[DocumentData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def total_pages(self) -> int:
        return sum(len(doc.pages) for doc in self.documents)

# PREPROCESSING UTILITIES

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocess image: grayscale, denoise, adaptive threshold.
    Returns preprocessed image as numpy array."""
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        denoised = cv2.fastNlMeansDenoising(gray, None, h=6, templateWindowSize=7, searchWindowSize=21)
        
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 3
        )
        thresh = cv2.medianBlur(thresh, 3)
        
        return thresh
    except Exception as e:
        logging.error(f"Preprocessing failed: {str(e)}")
        raise


def _normalize_pdf_page_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    try:
        out = img_bgr

        # 1) DPI harmonization
        if PIPELINE_PDF_DPI > 0 and PIPELINE_PDF_TARGET_DPI > 0 and PIPELINE_PDF_TARGET_DPI < PIPELINE_PDF_DPI:
            scale = float(PIPELINE_PDF_TARGET_DPI) / float(PIPELINE_PDF_DPI)
            new_w = max(1, int(round(out.shape[1] * scale)))
            new_h = max(1, int(round(out.shape[0] * scale)))
            out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 2) Luminance normalization
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_float = l_chan.astype(np.float32)
        black_pt = float(np.percentile(l_float, 5.0))
        white_pt = float(np.percentile(l_float, 99.5))
        if white_pt - black_pt > 1.0:
            l_float = np.clip((l_float - black_pt) * (255.0 / (white_pt - black_pt)), 0.0, 255.0)

        gamma = 1.12
        l_float = 255.0 * np.power(np.clip(l_float / 255.0, 0.0, 1.0), gamma)
        l_out = l_float.astype(np.uint8)
        out = cv2.cvtColor(cv2.merge([l_out, a_chan, b_chan]), cv2.COLOR_LAB2BGR)

        out = cv2.GaussianBlur(out, (3, 3), 0.6)

        return out
    except Exception as e:
        logging.warning(f"PDF page normalization failed; using raw render: {e}")
        return img_bgr


def _normalize_image_page_for_ocr(img_bgr: np.ndarray) -> np.ndarray:

    try:
        out = img_bgr

        
        ycc = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycc)

       
        y = cv2.fastNlMeansDenoising(y, None, h=3, templateWindowSize=7, searchWindowSize=21)

        
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        y = clahe.apply(y)

        
        y_f = y.astype(np.float32)
        black_pt = float(np.percentile(y_f, 2.0))
        white_pt = float(np.percentile(y_f, 99.5))
        if white_pt - black_pt > 1.0:
            y_f = np.clip((y_f - black_pt) * (255.0 / (white_pt - black_pt)), 0.0, 255.0)
        y = y_f.astype(np.uint8)

        out = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)
        return out
    except Exception as e:
        logging.warning(f"Image page normalization failed; using raw upload: {e}")
        return img_bgr


# SCAN CLASSIFIER + BRANCHED PREPROCESSORS

# Shared reconstructed bridge kernel size: used by both the classifier fragmentation probe
# and the reconstructed branch close, so the two stay in lockstep at any PIPELINE_PDF_DPI.
_SCAN_RECON_BRIDGE_K = max(5, int(round(PIPELINE_PDF_DPI / 60.0)))


# --------------------------------------------------------------------------------------
# Page-quality capture (augmentation calibration)
# --------------------------------------------------------------------------------------
# Gated on the SAME switch as the rest of the sidecar (pipeline.py:2187), so the desktop
# app's Ingest-tab checkbox governs it. With the sidecar off, nothing below runs and
# extraction is byte-identical to before this hook existed.
_PQ_MEASURE = None          # cached callable, or False once a load has failed
_PQ_PHASH = None            # ditto, for the perceptual hash used to dedup template repeats


def _page_quality_enabled() -> bool:
    return str(os.environ.get("OCULI_SIDECAR_DISABLED", "")).strip().lower() \
        not in ("1", "true", "yes")


def _page_quality_measure():
    """-> measure_page callable, or None. Imported lazily and only once.

    The sidecar directory is placed on sys.path by the pipeline import, but this module is
    also imported directly by probes and tools, so the import is guarded rather than assumed
    and a failure is cached so it is not retried for every page of every document.
    """
    global _PQ_MEASURE
    if _PQ_MEASURE is False:
        return None
    if _PQ_MEASURE is not None:
        return _PQ_MEASURE
    try:
        from quality_metrics import measure_page       # sidecar module, cv2+numpy only
        _PQ_MEASURE = measure_page
    except Exception as exc:
        logging.info("page-quality capture unavailable (%s); extraction unaffected", exc)
        _PQ_MEASURE = False
        return None
    return _PQ_MEASURE


def _page_quality_phash():
    """-> the sidecar's _phash, or None. Reused rather than reimplemented so a hash in
    page_quality.parquet means the same thing as one in the graph exporter's tables."""
    global _PQ_PHASH
    if _PQ_PHASH is False:
        return None
    if _PQ_PHASH is not None:
        return _PQ_PHASH
    try:
        from oculi_graph_exporter import _phash
        _PQ_PHASH = _phash
    except Exception:
        _PQ_PHASH = False
        return None
    return _PQ_PHASH


def _deskew_bgr(img_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Universal upstream deskew. Derives its own grayscale/binary mask so it has no
    dependency on downstream classifier state. No-ops if Hough yields <10 qualifying lines
    or the median angle is sub-0.3°. Angles are folded into [-45, 45] robustly so steep
    diagonal structures (e.g. 135°) do not corrupt the median.
    """
    try:
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr, 0.0
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(binary_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360.0, threshold=100,
                                minLineLength=100, maxLineGap=10)
        if lines is None:
            return img_bgr, 0.0
        angles: List[float] = []
        for x1, y1, x2, y2 in lines[:, 0]:
            a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            a = ((a + 45.0) % 90.0) - 45.0
            if abs(a) <= SCAN_DESKEW_MAX_ANGLE:
                angles.append(a)
        if len(angles) < 10:
            return img_bgr, 0.0
        angle = float(np.median(angles))
        if abs(angle) < 0.3:
            return img_bgr, 0.0
        h, w = img_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rotated = cv2.warpAffine(
            img_bgr, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return rotated, angle
    except Exception as e:
        logging.warning(f"Deskew failed; using original image: {e}")
        return img_bgr, 0.0


def _harmonize_pdf_dpi_only(img_bgr: np.ndarray) -> np.ndarray:
    """Geometric DPI scaling equivalent to step 1 of _normalize_pdf_page_for_ocr, without
    luminance/gamma/blur. Uses INTER_NEAREST + Otsu re-threshold so reconstructed hard-binary
    strokes stay hard-binary after downscale.
    """
    if PIPELINE_PDF_DPI > 0 and PIPELINE_PDF_TARGET_DPI > 0 and PIPELINE_PDF_TARGET_DPI < PIPELINE_PDF_DPI:
        scale = float(PIPELINE_PDF_TARGET_DPI) / float(PIPELINE_PDF_DPI)
        new_w = max(1, int(round(img_bgr.shape[1] * scale)))
        new_h = max(1, int(round(img_bgr.shape[0] * scale)))
        out = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return img_bgr


def _compute_classifier_signals(
    gray: np.ndarray,
    binary_mask: np.ndarray,
    total_px: int,
) -> Dict[str, float]:
    """Extract cheap scalar signals that separate the 4 scan branches. All heavy work is done
    once here and reused by the decision function.
    """
    # Raw CC count (ink components), then stats for area distribution
    num_cc_raw, _, stats_raw, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    # Exclude background component (label 0) for area stats
    if num_cc_raw > 1:
        areas = stats_raw[1:, cv2.CC_STAT_AREA].astype(np.float64)
    else:
        areas = np.zeros((0,), dtype=np.float64)
    total_ink = float(areas.sum()) if areas.size > 0 else 0.0
    max_cc_area_ratio = float(areas.max() / total_ink) if total_ink > 0.0 else 0.0
    small_blob_px_threshold = ((PIPELINE_PDF_DPI / 100.0) ** 2) * 2.0
    if areas.size > 0:
        small_blob_ratio = float(np.count_nonzero(areas < small_blob_px_threshold)) / float(areas.size)
    else:
        small_blob_ratio = 0.0

    # Fragmentation: CC-count reduction after a vertical close with the shared bridge kernel.
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _SCAN_RECON_BRIDGE_K))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, bridge_kernel)
    # Count-only form: skip the stats/centroids arrays we don't use on the closed pass.
    num_cc_closed, _ = cv2.connectedComponents(closed_mask, connectivity=8)
    frag_ratio = float(num_cc_raw) / float(max(1, num_cc_closed))

    # Sharpness proxy (kept as an audit signal; no longer gates a branch decision).
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Edge density
    canny = cv2.Canny(gray, 80, 200)
    edge_density = float(cv2.countNonZero(canny)) / float(max(1, total_px))

    # Stroke-width variability via distance transform on the ink mask. Each ink pixel's
    # distance-to-nearest-background is approximately half the local stroke width, so the
    # coefficient of variation over all ink pixels captures how consistent stroke widths are
    # across the page. Clean rasters sit around 0.25-0.35; quantised/pixelated degraded
    # scans widen visibly (typically >0.4) because edge rounding swings stroke widths by
    # +/-1px along the same stroke.
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
    ink_distances = dist[binary_mask > 0]
    if ink_distances.size > 0:
        stroke_width_median_px = float(np.median(ink_distances))
        if stroke_width_median_px > 0.0:
            stroke_width_cv = float(ink_distances.std() / stroke_width_median_px)
        else:
            stroke_width_cv = 0.0
    else:
        stroke_width_median_px = 0.0
        stroke_width_cv = 0.0

    # Ink fraction and component density: two lines from values already computed here, and
    # the sidecar's image_quality row has always had columns for them.
    ink_ratio = (total_ink / float(max(1, total_px))) if total_px > 0 else 0.0
    cc_density = (float(num_cc_raw) / float(max(1, total_px))) * 1e6   # per megapixel

    return {
        "num_cc_raw": float(num_cc_raw),
        "frag_ratio": frag_ratio,
        "small_blob_ratio": small_blob_ratio,
        "max_cc_area_ratio": max_cc_area_ratio,
        "laplacian_var": laplacian_var,
        "edge_density": edge_density,
        "stroke_width_cv": stroke_width_cv,
        "stroke_width_median_px": stroke_width_median_px,
        "ink_ratio": ink_ratio,
        "cc_density": cc_density,
        # ---- aliases under the names the sidecar exporter actually reads.
        # oculi_graph_exporter._image_quality asks signals for "blur_metric",
        # "small_blob_frac" and "stroke_cv". Those keys were never produced here, so three
        # measurements that ARE taken on every page were written out as 0.0 - the columns
        # looked populated and carried nothing. Aliased rather than renamed because
        # _classify_scan_branch and _branch_output_acceptable read the original names.
        #
        # NOT aliased: line_thickness. The exporter reads it from the page-meta dict (never
        # set), and the obvious source, stroke_width_median_px, measures 0.955 on every page
        # tested - the DIST_L2 mask-3 transform saturates on thin strokes. Wiring it would
        # publish a constant as a measurement. Real stroke width comes from the page-quality
        # row's stroke_w_mean instead.
        "blur_metric": laplacian_var,
        "small_blob_frac": small_blob_ratio,
        "stroke_cv": stroke_width_cv,
    }


def _classify_scan_branch(signals: Dict[str, float]) -> Tuple[str, float]:
    """4-way decision ladder. Priority: Reconstructed > Contaminated > Degraded > CG-Normal.
    Confidence below SCAN_CONFIDENCE_FALLBACK is demoted to CG-Normal; the original pick is
    preserved as signals["raw_branch"] for audit.
    """
    frag_ratio = signals.get("frag_ratio", 1.0)
    small_blob_ratio = signals.get("small_blob_ratio", 0.0)
    max_cc_area_ratio = signals.get("max_cc_area_ratio", 0.0)
    stroke_width_cv = signals.get("stroke_width_cv", 0.0)
    edge_density = signals.get("edge_density", 0.0)

    if frag_ratio > SCAN_FRAG_THRESHOLD:
        branch = "reconstructed"
        conf = min(1.0, 0.6 + (frag_ratio - SCAN_FRAG_THRESHOLD) / 6.0)
    elif (small_blob_ratio > SCAN_SMALL_BLOB_THRESHOLD
          and max_cc_area_ratio > SCAN_MAX_CC_AREA_THRESHOLD
          and frag_ratio < 2.0):
        branch = "contaminated"
        conf = min(1.0, 0.5 + max(0.0, small_blob_ratio - SCAN_SMALL_BLOB_THRESHOLD) * 3.0)
    elif (stroke_width_cv > SCAN_STROKE_CV_HIGH
          and edge_density > SCAN_EDGE_DENSITY_HIGH
          and frag_ratio < 2.0):
        branch = "degraded"
        conf = min(1.0, 0.5 + max(0.0, stroke_width_cv - SCAN_STROKE_CV_HIGH) * 2.5)
    else:
        branch = "cg_normal"
        conf = 1.0

    if conf < SCAN_CONFIDENCE_FALLBACK and branch != "cg_normal":
        signals["raw_branch"] = branch
        branch = "cg_normal"

    return branch, float(conf)


def _preprocess_reconstructed(img_bgr: np.ndarray) -> np.ndarray:
    """Bridge scanline-decomposed strokes with a vertical close using the shared DPI-relative
    kernel, then reconstruct hard-binary ink onto a white BGR canvas. Final DPI is harmonised
    via INTER_NEAREST + Otsu re-threshold so dimensions match other branches without
    reintroducing grey.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _SCAN_RECON_BRIDGE_K))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    out = np.full_like(img_bgr, 255)
    out[closed > 0] = 0
    return _harmonize_pdf_dpi_only(out)


def _preprocess_degraded(img_bgr: np.ndarray) -> np.ndarray:
    """Recover stairstepped low-res raster: cubic upscale 1.5x, conservative unsharp
    (1.4/-0.4, sigma=1.0) to avoid ringing, area downscale, then existing luminance+DPI path.
    """
    h, w = img_bgr.shape[:2]
    up = cv2.resize(img_bgr, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(up, 1.4, blur, -0.4, 0)
    down = cv2.resize(sharp, (w, h), interpolation=cv2.INTER_AREA)
    return _normalize_pdf_page_for_ocr(down)


def _preprocess_contaminated(img_bgr: np.ndarray) -> np.ndarray:
    """Remove edge speckle on otherwise clean structure with a 3x3 median, then reuse the
    existing luminance+DPI pipeline for consistency with CG-Normal output.
    """
    despeckled = cv2.medianBlur(img_bgr, 3)
    return _normalize_pdf_page_for_ocr(despeckled)


def _branch_output_acceptable(
    input_mask: np.ndarray,
    input_num_cc: float,
    output_bgr: np.ndarray,
    branch: str,
) -> Tuple[bool, str]:
    """Cheap regression check: reject a branch preprocessor's output if ink coverage or
    CC density moved outside a sane band vs the classifier's input mask. Comparisons are
    resolution-invariant (ratios / densities), so branches that legitimately downscale
    (reconstructed / degraded / contaminated all target PIPELINE_PDF_TARGET_DPI) aren't
    penalised for image-size changes. Fail-open on any internal error so a broken gate
    never blocks an otherwise-good branch.
    """
    try:
        input_px = int(input_mask.size)
        if input_px <= 0:
            return True, "empty_input"
        input_ink_ratio = float(np.count_nonzero(input_mask)) / float(input_px)
        input_cc_density = float(input_num_cc) / float(input_px)

        out_gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY)
        _, out_mask = cv2.threshold(out_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        out_px = int(out_mask.size)
        if out_px <= 0:
            return False, "empty_output"
        out_ink_ratio = float(np.count_nonzero(out_mask)) / float(out_px)
        out_cc, _ = cv2.connectedComponents(out_mask, connectivity=8)
        out_cc_density = float(out_cc) / float(out_px)

        # Ink coverage band
        if input_ink_ratio > 1e-4:
            ink_ratio = out_ink_ratio / input_ink_ratio
            if ink_ratio < SCAN_QUALITY_INK_LO:
                return False, f"ink_lost:{ink_ratio:.2f}"
            if ink_ratio > SCAN_QUALITY_INK_HI:
                return False, f"ink_added:{ink_ratio:.2f}"

        # CC density band (fragmentation/over-merge guards)
        if input_cc_density > 1e-9:
            cc_ratio = out_cc_density / input_cc_density
            if cc_ratio > SCAN_QUALITY_CC_HI:
                return False, f"cc_exploded:{cc_ratio:.2f}"
            if branch == "reconstructed" and cc_ratio < SCAN_QUALITY_CC_LO_RECON:
                return False, f"cc_collapsed:{cc_ratio:.2f}"

        return True, "ok"
    except Exception as e:
        return True, f"gate_error:{str(e)[:40]}"


def _classify_and_branch(img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], str, float, int]:
    """Orchestrator: shared pre-step (gray + CLAHE lite + Otsu mask) -> signal computation
    -> branch decision -> dispatch to the matching per-branch preprocessor. Fully fail-safe:
    any exception anywhere inside this call returns _normalize_pdf_page_for_ocr(img_bgr) as
    the graceful CG-Normal fallback so extraction never degrades into placeholder pages.
    """
    t0 = time.perf_counter()
    try:
        # Shared pre-step
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
        gray_eq = clahe.apply(gray)
        _, binary_mask = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        total_px = int(gray.shape[0]) * int(gray.shape[1])

        signals: Dict[str, Any] = _compute_classifier_signals(gray_eq, binary_mask, total_px)
        branch, conf = _classify_scan_branch(signals)

        # Page-quality capture for augmentation calibration. Measured HERE because this is
        # the one place the raster is present by construction: page.image is released at
        # several points during extraction (pipeline.py:1027, :1883, process_input:1045),
        # long before the document-end cleanup, so anything downstream that assumes a live
        # raster is the bug that already cost us the classify-regions page images.
        #
        # Measured on `gray`, not `gray_eq`: CLAHE changes contrast and tone, and
        # contrast_p95_p05 / bg_level / laplacian_var must describe the page as scanned.
        # Its own try/except - a measurement failure must never cost an extraction.
        if _page_quality_enabled():
            try:
                _mp = _page_quality_measure()
                if _mp is not None:
                    _row = _mp(gray)
                    # phash must also be taken here: it needs the raster, and the writer
                    # downstream has only metadata. Hashed from img_bgr - the ORIGINAL page,
                    # before branch preprocessing - so two identical source pages that get
                    # routed to different scan branches still hash the same. Hashing the
                    # preprocessed output would make the hash branch-dependent and defeat
                    # the dedup it exists for.
                    _ph = _page_quality_phash()
                    if _ph is not None:
                        _row["phash"] = _ph(img_bgr)
                    signals["page_quality"] = _row
            except Exception as _pq_exc:
                logging.debug("page-quality capture skipped: %s", _pq_exc, exc_info=True)

        if branch == "reconstructed":
            candidate = _preprocess_reconstructed(img_bgr)
        elif branch == "contaminated":
            candidate = _preprocess_contaminated(img_bgr)
        elif branch == "degraded":
            candidate = _preprocess_degraded(img_bgr)
        else:
            candidate = _normalize_pdf_page_for_ocr(img_bgr)

        # Post-branch quality gate: only applies to non-default branches; CG-Normal is the
        # fallback itself so there's nothing to compare against. Rejection routes to CG-Normal
        # and records both the rejected branch and the reason in classifier_signals for audit.
        if SCAN_QUALITY_GATE_ENABLED and branch != "cg_normal":
            ok, reason = _branch_output_acceptable(
                binary_mask, signals.get("num_cc_raw", 0.0), candidate, branch,
            )
            if not ok:
                signals["rejected_branch"] = branch
                signals["rejection_reason"] = reason
                candidate = _normalize_pdf_page_for_ocr(img_bgr)
                branch = "cg_normal_rejected"

        out_bgr = candidate
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return out_bgr, signals, branch, conf, elapsed_ms
    except Exception as e:
        logging.warning(f"Scan classifier failed; falling back to CG-Normal path: {e}")
        fallback = _normalize_pdf_page_for_ocr(img_bgr)
        signals = {"error": str(e)[:200], "raw_branch": "cg_normal_fallback"}
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return fallback, signals, "cg_normal_fallback", 0.0, elapsed_ms


def calculate_patches(width: int, height: int) -> int:
    """Calculate patch count based on image size (A4 threshold at configured DPI)."""
    a4_area = _A4_WIDTH_PX * _A4_HEIGHT_PX
    return 20 if width * height >= a4_area else 10

# INGESTION: BATCH → DOCUMENT → PAGE

PDF_EXT = {'.pdf'}
# Image extensions
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

TEXT_CSV_EXT = {'.txt', '.csv'}
# Office: .docx via python-docx; .doc via LibreOffice (not currently available)
DOCX_EXT = {'.docx'}
DOC_EXT = {'.doc'}
DOCM_EXT = {'.docm'}
UNSUPPORTED_WORD_EXT = DOCX_EXT | DOC_EXT | DOCM_EXT
# Archives: expanded and contents processed
ARCHIVE_EXT = {'.zip', '.tar'}
# supported for batch/upload
SUPPORTED_EXT = PDF_EXT | IMAGE_EXT | TEXT_CSV_EXT | ARCHIVE_EXT


def _logical_path_posix(path: str) -> str:
    return path.replace("\\", "/")


def _document_id_for_relative_path(batch_id: str, rel_logical: str) -> str:
    """Stable id from batch + relative path (avoids collisions for same basename in different folders)."""
    digest = hashlib.sha256(f"{batch_id}|{rel_logical}".encode("utf-8")).hexdigest()[:16]
    return f"{batch_id}_DOC_{digest}"


def _relative_folder_from_logical(rel_logical: str) -> str:
    rel_logical = _logical_path_posix(rel_logical).strip("/")
    if not rel_logical or "/" not in rel_logical:
        return ""
    return rel_logical.rsplit("/", 1)[0]


def ingest_folder_fields_from_relative_path(rel_logical: str) -> Dict[str, Any]:
    """Derive subfolder labels from the path under the batch root (for internal validation / schema typing).

    ``ingest_document_class`` is the first directory segment under the batch (e.g. Drawing vs WI).
    ``ingest_immediate_parent_folder`` is the directory that directly contains the file.
    ``ingest_folder_segments`` lists every folder from the batch root down to that parent.
    """
    rel = _logical_path_posix(rel_logical).strip("/")
    parent = rel.rsplit("/", 1)[0] if rel and "/" in rel else ""
    parts = [p for p in parent.split("/") if p] if parent else []
    return {
        "ingest_folder_segments": parts,
        "ingest_document_class": parts[0] if parts else "",
        "ingest_immediate_parent_folder": parts[-1] if parts else "",
    }


_INTERNAL_VAL_METADATA_KEYS = (
    "relative_folder",
    "relative_path",
    "filename",
    "ingest_folder_segments",
    "ingest_document_class",
    "ingest_immediate_parent_folder",
)


def metadata_slice_for_internal_validator(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Subset of document metadata passed to InternalValidator (folder typing, paths)."""
    return {k: metadata[k] for k in _INTERNAL_VAL_METADATA_KEYS if k in metadata}


def _collect_input_entries(
    input_abs: str, temp_dirs_to_clean: List[str]
) -> List[Tuple[str, str]]:
    """Walk the input tree and return sorted (absolute_path, logical_relative_path)
    entries, expanding archives into temp dirs registered in temp_dirs_to_clean."""
    all_entries: List[Tuple[str, str]] = []
    # followlinks=False (default, made explicit): a directory junction/symlink on
    # a network share must not redirect the walk outside the selected root (e.g.
    # into \\attacker\share or C:\Windows). Input sources are user-selected and
    # gated to fixed/mapped drives by the frontend, but this keeps the traversal
    # contained regardless.
    for root, dirs, files in os.walk(input_abs, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXT:
                if ext and not fname.startswith("."):
                    rel = _logical_path_posix(os.path.relpath(os.path.join(root, fname), input_abs))
                    logging.warning("[INGEST] Skipping unsupported file type %r: %s", ext, rel)
                continue
            abs_path = os.path.join(root, fname)
            if not os.path.isfile(abs_path):
                continue
            rel_logical = _logical_path_posix(os.path.relpath(abs_path, input_abs))
            if ext in ARCHIVE_EXT:
                stem = rel_logical
                for suf in (".zip", ".tar", ".tar.gz", ".tgz"):
                    if stem.lower().endswith(suf):
                        stem = stem[: -len(suf)]
                        break
                entries = _expand_archive_entries(abs_path, stem, temp_dirs_to_clean)
                all_entries.extend(entries)
            else:
                all_entries.append((abs_path, rel_logical))
    all_entries.sort(key=lambda t: t[1].lower())
    return all_entries


def _count_pdf_pages(file_path: str) -> int:
    with pdfplumber.open(file_path) as pdf:
        return len(pdf.pages)


def _probe_page_count(file_path: str, ext: str) -> int:
    """Cheap page count without rendering any pixels (for progress/ETA totals).

    Bounded by PIPELINE_PDF_OPEN_TIMEOUT_SEC with a non-waiting shutdown so a
    pathological PDF cannot hang the scan; a failed probe just means the
    progress total treats the document as one page.
    """
    if ext not in PDF_EXT:
        return 1
    exc = ThreadPoolExecutor(max_workers=1)
    try:
        future = exc.submit(_count_pdf_pages, file_path)
        return max(1, int(future.result(timeout=PIPELINE_PDF_OPEN_TIMEOUT_SEC)))
    except Exception as e:
        logging.debug("Page-count probe failed for %s: %s", file_path, e)
        return 1
    finally:
        exc.shutdown(wait=False)


def _reject_unc_path(input_dir: str) -> None:
    """Refuse UNC network share paths (\\\\server\\share) independently of the UI.

    Security policy (enforced at the file-reader, not only the frontend): only
    local drives and mapped network drives — both of which resolve to a drive
    letter like ``X:`` — are permitted. A raw UNC path makes Windows
    auto-authenticate to an arbitrary named host, an NTLM-leak / hostile-share
    vector; a mapped drive was mounted out-of-band with the user's credentials
    and is scoped to what the user already may access. Invoking this module's
    CLI directly therefore cannot bypass the policy.
    """
    raw = str(input_dir or "").strip()
    if raw.startswith("\\\\") or raw.startswith("//"):
        raise ValueError(
            "Network share paths (\\\\server\\share) are not supported. Map the "
            "share to a drive letter (e.g. X:) and use that."
        )
    # Catch UNC that only becomes apparent after normalization.
    drive = os.path.splitdrive(os.path.abspath(raw))[0]
    if drive.startswith("\\\\") or drive.startswith("//"):
        raise ValueError(
            "Network share paths are not supported. Map the share to a drive "
            "letter (e.g. X:) and use that."
        )


def scan_input_local(
    input_dir: str,
    received_by: str = "local_test",
    probe_page_counts: bool = True,
    batch_id: Optional[str] = None,
) -> BatchData:
    """Phase 1 of ingestion: walk the corpus and build document STUBS only.

    No page rendering happens here — each DocumentData carries metadata plus an
    ``expected_page_count`` probe, and ``document.pages`` stays empty until
    ``ingest_document_pages`` is called for that document. This is what keeps
    peak memory independent of corpus size: an 8000-document upload costs
    kilobytes per document at scan time, not a 300-DPI render per page.

    Archives are expanded to temp dirs which must outlive rendering; callers
    release them with ``cleanup_batch_temp_dirs(batch)`` when the run finishes.
    """
    _reject_unc_path(input_dir)
    input_abs = os.path.abspath(input_dir)
    if not os.path.isdir(input_abs):
        raise ValueError(f"Input path is not a directory: {input_dir}")

    temp_dirs_to_clean: List[str] = []
    try:
        all_entries = _collect_input_entries(input_abs, temp_dirs_to_clean)
    except Exception:
        for d in temp_dirs_to_clean:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        raise

    if not all_entries:
        logging.warning(f"No supported files found under {input_dir} (recursive)")

    # document_id = hash(batch_id | relative_path), so a resumed run MUST reuse
    # its original batch_id — a fresh random id changes every document_id and
    # makes previously-persisted completion markers unmatchable (full rescan
    # from zero instead of a resume).
    if not batch_id:
        batch_id = f"BATCH_{np.random.randint(1e8):08d}"
    batch = BatchData(
        batch_id=batch_id,
        source_path=input_dir,
        received_by=received_by,
        status="PROCESSING",
        metadata={
            "ingest_type": "local_dir",
            "recursive_subfolders": True,
            "temp_dirs": temp_dirs_to_clean,
        },
    )

    def _build_stub(entry: Tuple[str, str]) -> Optional[DocumentData]:
        file_path, rel_logical = entry
        if not os.path.isfile(file_path):
            return None
        ext = os.path.splitext(rel_logical)[1].lower()
        document = DocumentData(
            document_id=_document_id_for_relative_path(batch_id, rel_logical),
            batch_id=batch_id,
            source_path=file_path,
            file_size=os.path.getsize(file_path),
            status="SCANNED",
            metadata={
                "ext": ext,
                "filename": os.path.basename(rel_logical),
                "relative_path": rel_logical,
                "relative_folder": _relative_folder_from_logical(rel_logical),
                **ingest_folder_fields_from_relative_path(rel_logical),
            },
        )
        if probe_page_counts:
            document.metadata["expected_page_count"] = _probe_page_count(file_path, ext)
        return document

    _scan_workers = int(os.environ.get("INGEST_WORKERS", "8").strip())
    if probe_page_counts and _scan_workers > 1 and len(all_entries) > 1:
        # The probe opens each file (no rendering); parallelize the I/O but
        # preserve the sorted entry order in batch.documents.
        with ThreadPoolExecutor(max_workers=_scan_workers) as executor:
            stubs = list(executor.map(_build_stub, all_entries))
    else:
        stubs = [_build_stub(entry) for entry in all_entries]
    batch.documents.extend(d for d in stubs if d is not None)

    batch.status = "SCANNED"
    logging.info(
        "Batch %s: scanned %d documents (~%d pages expected); pages render just-in-time",
        batch_id,
        batch.document_count,
        expected_total_pages(batch),
    )
    return batch


def expected_total_pages(batch: BatchData) -> int:
    """Page total for progress/ETA that works for stubs and rendered docs alike."""
    total = 0
    for d in batch.documents:
        if d.pages:
            total += len(d.pages)
        else:
            total += int((d.metadata or {}).get("expected_page_count") or 1)
    return total


def ingest_document_pages(document: DocumentData) -> DocumentData:
    """Phase 2 of ingestion: render one document's pages (idempotent)."""
    if document.pages:
        return document
    file_path = document.source_path
    ext = str((document.metadata or {}).get("ext") or os.path.splitext(file_path)[1].lower())
    try:
        document.pages = _extract_pages(document, ext, file_path)
        document.status = "PAGES_EXTRACTED"
    except Exception as e:
        logging.warning(f"Extract failed for {file_path}: {e}")
        document.pages = _create_placeholder_page(
            document, file_path, reason="extract_failed", detail=str(e)[:200]
        )
        document.status = "PAGES_EXTRACTED"
    return document


# ---------------------------------------------------------------------------------------
# Worker-side decoding
# ---------------------------------------------------------------------------------------
# The main process used to decode EVERY page of a document up front and hold every image
# until the whole document finished, then copy each one into the extraction worker. On a
# 1,000-page document that is 1,000 page images resident at once in the main process —
# the prod machine reached ~90% of 32 GB — plus a second copy of each page in the worker.
#
# Instead the main process now builds image-free STUBS (ids, page numbers, source path),
# and the extraction worker decodes each page itself, from the file, immediately before
# running OCR on it. Consequences:
#
#   * The main process holds NO page images. A page image exists once, inside the worker,
#     only while that page is being processed.
#   * No page image ever crosses a process boundary, and nothing is written to disk.
#   * Decoding runs in native code (pdfium, OpenCV, libtiff). A native crash there now
#     kills the WORKER, not the orchestrator — the prod failure (exit 0xC0000005) took the
#     whole run down because decoding ran in a thread of the main process.
#
# Pages are decoded by the SAME functions as before (_render_one_page, _extract_image_page),
# so the pixels OCR sees are unchanged. Only where and when they are produced moves.
#
# Text and CSV documents are rendered from text, not decoded from a native format, are
# small, and cannot be page-counted without rendering — they keep the old parent-side path.
#
# OCULI_WORKER_DECODE=0 restores the old behaviour exactly.

WORKER_DECODE = os.environ.get("OCULI_WORKER_DECODE", "1").strip().lower() in (
    "1", "true", "yes", "on")

PAGE_STUB_STATUS = "PENDING_RENDER"
_WORKER_RENDER_EXT = PDF_EXT | IMAGE_EXT


def is_page_stub(page: Any) -> bool:
    """True for an image-free page the extraction worker still has to decode."""
    return getattr(page, "image", None) is None and getattr(page, "status", "") == PAGE_STUB_STATUS


def ingest_document_page_stubs(document: DocumentData) -> DocumentData:
    """Phase 2 of ingestion WITHOUT decoding: one image-free stub per page (idempotent).

    Needs only the page count — for a PDF that is the cheap no-render probe. Failures that
    would previously have surfaced while rendering (an unopenable PDF) produce the same
    placeholder pages here, with the same reasons, so downstream sees what it saw before.
    """
    if document.pages:
        return document
    file_path = document.source_path
    ext = str((document.metadata or {}).get("ext") or os.path.splitext(file_path)[1].lower())
    if not WORKER_DECODE or ext not in _WORKER_RENDER_EXT:
        return ingest_document_pages(document)

    if ext in PDF_EXT:
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            n_pages = int(pool.submit(_count_pdf_pages, file_path).result(
                timeout=PIPELINE_PDF_OPEN_TIMEOUT_SEC))
        except TimeoutError:
            logging.warning("PDF open timeout document_id=%s file=%s after %ss",
                            document.document_id, file_path, PIPELINE_PDF_OPEN_TIMEOUT_SEC)
            document.pages = _create_placeholder_page(
                document, file_path, reason="pdf_open_timeout",
                detail=f"timeout_after_{PIPELINE_PDF_OPEN_TIMEOUT_SEC}s")
            document.status = "PAGES_EXTRACTED"
            return document
        except Exception as e:
            logging.warning(f"PDF extract failed for {file_path}: {e}")
            document.pages = _create_placeholder_page(
                document, file_path, reason="pdf_extract_failed", detail=str(e)[:200])
            document.status = "PAGES_EXTRACTED"
            return document
        finally:
            pool.shutdown(wait=False)
    else:
        n_pages = 1

    document.pages = [
        PageData(
            page_id=f"{document.document_id}_PG{i:03d}",
            document_id=document.document_id,
            batch_id=document.batch_id,
            page_number=i,
            image=None,
            width=0,
            height=0,
            storage_path=file_path,
            status=PAGE_STUB_STATUS,
            metadata={"render": "worker", "ext": ext},
        )
        for i in range(1, n_pages + 1)
    ]
    document.status = "PAGES_EXTRACTED"
    return document


def decode_failure_reason(ext: str) -> str:
    """The placeholder reason the old parent-side path used for the same failure."""
    return "pdf_page_render_failed" if ext in PDF_EXT else "extract_failed"


# ---- worker process state ------------------------------------------------------------
# One open PDF per worker, reused across that document's pages. Reopening per page would
# re-parse the whole file each time — the one way this design could cost more than before.
_WORKER_PDF: Dict[str, Any] = {"path": None, "pdf": None}
_WORKER_LOCK = threading.Lock()
_PREFETCH_POOL: Optional[ThreadPoolExecutor] = None
_PREFETCH: Dict[Tuple[str, int], Any] = {}


def _worker_pdf_page(file_path: str, page_number: int):
    with _WORKER_LOCK:
        if _WORKER_PDF["path"] != file_path:
            old = _WORKER_PDF["pdf"]
            _WORKER_PDF["path"], _WORKER_PDF["pdf"] = None, None
            if old is not None:
                try:
                    old.close()
                except Exception:
                    pass
            _WORKER_PDF["pdf"] = pdfplumber.open(file_path)
            _WORKER_PDF["path"] = file_path
        pdf = _WORKER_PDF["pdf"]
    return pdf.pages[page_number - 1]


def _test_native_crash_hook(file_path: str, page_number: int) -> None:
    """TEST-ONLY. Never set in production — same convention as PIPELINE_TEST_STUB_OCR.

    PIPELINE_TEST_CRASH_DECODE="<file basename>:<page number>" makes decoding that page
    kill its process exactly as a real decoder fault does: no exception, no cleanup, exit
    0xC0000005. That is the only way to exercise crash recovery through the real pipeline,
    including the case that matters most — a crash in the PREFETCH of the next page, which
    must not be blamed on the page being processed.
    """
    spec = os.environ.get("PIPELINE_TEST_CRASH_DECODE", "").strip()
    if not spec or ":" not in spec:
        return
    name, _, num = spec.rpartition(":")
    if os.path.basename(file_path) == name and str(page_number) == num.strip():
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k32.TerminateProcess(k32.GetCurrentProcess(), 0xC0000005)


def render_page_for_extraction(file_path: str, ext: str, page_number: int,
                               document_id: str, batch_id: str) -> PageData:
    """Decode exactly one page. Runs inside the extraction worker. Raises on failure."""
    _test_native_crash_hook(file_path, page_number)
    if ext in PDF_EXT:
        pdf_page = _worker_pdf_page(file_path, page_number)
        try:
            return _render_one_page(pdf_page, page_number, document_id, batch_id, file_path)
        finally:
            # Drop pdfplumber's per-page layout cache; with the file held open across
            # pages it would otherwise accumulate for every page rendered.
            for _free in ("close", "flush_cache"):
                fn = getattr(pdf_page, _free, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break
    if ext in IMAGE_EXT:
        doc = DocumentData(document_id=document_id, batch_id=batch_id,
                           source_path=file_path, file_size=0)
        return _extract_image_page(doc, file_path)[0]
    raise ValueError(f"worker decode does not support {ext!r}")


def _drain_prefetch(keep: Optional[Tuple[str, int]] = None) -> None:
    """Discard prefetched pages other than `keep`, waiting for any still decoding.

    Waiting matters: a prefetch still rendering from the open PDF must finish before that
    PDF can be closed for the next document.
    """
    for key in [k for k in _PREFETCH if k != keep]:
        fut = _PREFETCH.pop(key)
        try:
            fut.result()
        except Exception:
            pass


def take_page(file_path: str, ext: str, page_number: int,
              document_id: str, batch_id: str) -> PageData:
    """The decoded page: from the prefetch if it was started, otherwise decoded now."""
    key = (file_path, int(page_number))
    fut = _PREFETCH.pop(key, None)
    _drain_prefetch()
    if fut is not None:
        return fut.result()          # re-raises a decode error, exactly as decoding now would
    return render_page_for_extraction(file_path, ext, page_number, document_id, batch_id)


def prefetch_page(file_path: str, ext: str, page_number: int,
                  document_id: str, batch_id: str) -> None:
    """Start decoding the NEXT page on a background thread while this one is OCR'd.

    At most one page is ever held ahead, so the worker never has more than two pages in
    memory. Rasterization is still serialised by _PDFIUM_RENDER_LOCK, and OCR does not use
    pdfium, so decoding ahead cannot collide with the page being processed.
    """
    global _PREFETCH_POOL
    key = (file_path, int(page_number))
    if key in _PREFETCH:
        return
    _drain_prefetch()
    if _PREFETCH_POOL is None:
        _PREFETCH_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="page-prefetch")
    _PREFETCH[key] = _PREFETCH_POOL.submit(
        render_page_for_extraction, file_path, ext, page_number, document_id, batch_id)


def release_document_pages(document: DocumentData) -> None:
    """Drop a document's rendered page images (PageData shells remain)."""
    for page in document.pages or []:
        page.image = None


def cleanup_batch_temp_dirs(batch: BatchData) -> None:
    """Remove archive-expansion temp dirs once no document will render again."""
    for d in (batch.metadata or {}).pop("temp_dirs", []) or []:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


class DocumentPrefetcher:
    """Iterate a batch's documents with a bounded render-ahead window.

    At most ``prefetch`` documents are being rendered (or rendered-and-waiting)
    at any moment, so resident page-image memory is ~(prefetch + 1) documents
    while the pipeline works — regardless of how many documents the batch
    holds. The caller must release each document's images when done with it
    (``release_document_pages``); this class only bounds the load side.
    """

    def __init__(
        self,
        documents: List[DocumentData],
        prefetch: Optional[int] = None,
        skip_document_ids: Optional[set] = None,
    ):
        skip = skip_document_ids or set()
        self._docs = [d for d in documents if d.document_id not in skip]
        self._prefetch = max(1, int(prefetch if prefetch is not None else INGEST_PREFETCH_DOCS))
        self._pool = ThreadPoolExecutor(max_workers=self._prefetch)
        self._queue: deque = deque()
        self._next = 0
        self._fill()

    def _fill(self) -> None:
        while len(self._queue) < self._prefetch and self._next < len(self._docs):
            doc = self._docs[self._next]
            self._next += 1
            # Stubs, not decoded pages: decoding now happens in the extraction worker.
            # (Falls back to full parent-side decoding when OCULI_WORKER_DECODE=0.)
            self._queue.append((doc, self._pool.submit(ingest_document_page_stubs, doc)))

    def __len__(self) -> int:
        return len(self._docs)

    def __iter__(self) -> "DocumentPrefetcher":
        return self

    def __next__(self) -> DocumentData:
        if not self._queue:
            self.close()
            raise StopIteration
        doc, future = self._queue.popleft()
        try:
            future.result()
        except Exception as e:
            # ingest_document_pages handles its own failures with placeholder
            # pages; this only fires on truly unexpected errors.
            logging.warning("Prefetch render failed for %s: %s", doc.document_id, e)
        self._fill()
        return doc

    def close(self) -> None:
        self._pool.shutdown(wait=False)


def process_input_local(input_dir: str, received_by: str = "local_test") -> BatchData:
    """Legacy EAGER ingestion: scan + render every document up front.

    Peak memory is proportional to the whole corpus — fine for small test
    inputs, ruinous at thousands of documents. Large-scale callers
    (session_runtime, pipeline_main) use scan_input_local + just-in-time
    rendering instead.
    """
    batch = scan_input_local(input_dir, received_by, probe_page_counts=False)
    all_docs = list(batch.documents)

    # Parallel ingestion: documents are independent (file I/O + image rendering)
    # Default 8 workers; tune with INGEST_WORKERS env var. Use 1 worker if set explicitly.
    _ingest_workers = int(os.environ.get("INGEST_WORKERS", "8").strip())
    if _ingest_workers > 1 and len(all_docs) > 1:
        logging.info(f"Parallel ingestion enabled: {_ingest_workers} workers for {len(all_docs)} documents")
        with ThreadPoolExecutor(max_workers=_ingest_workers) as executor:
            list(executor.map(ingest_document_pages, all_docs))
    else:
        for document in all_docs:
            ingest_document_pages(document)

    batch.status = "INGESTED"
    logging.info(f"Batch {batch.batch_id}: {batch.document_count} documents, {batch.total_pages} total pages")
    cleanup_batch_temp_dirs(batch)
    return batch


def _extract_pages(document: DocumentData, ext: str, file_path: str) -> List[PageData]:
    ext = ext.lower()
    if ext in PDF_EXT:
        return _extract_pdf_pages(document, file_path)
    if ext in IMAGE_EXT:
        return _extract_image_page(document, file_path)
    if ext == '.txt':
        return _extract_txt_pages(document, file_path)
    if ext == '.csv':
        return _extract_csv_pages(document, file_path)
    logging.warning(f"[INGEST] File type {ext} not supported for OCR extraction, creating placeholder")
    return _create_placeholder_page(document, file_path)


# Process-wide mutex for pdfium rasterization (see comment in _render_one_page).
_PDFIUM_RENDER_LOCK = threading.Lock()


def _render_one_page(
    pdf_page,
    page_number: int,
    document_id: str,
    batch_id: str,
    file_path: str,
) -> PageData:
    """Render, deskew, and classify a single PDF page. Runs inside a worker thread.

    Raises on failure — the caller is responsible for catching and substituting
    a placeholder. Do not swallow exceptions here; a silently-failed render with
    no error surfaced is worse than a loud one.
    """
    page_id = f"{document_id}_PG{page_number:03d}"
    try:
        # pdfium (pypdfium2, which pdfplumber's to_image uses) is NOT
        # thread-safe: concurrent rasterization — even on fully separate
        # documents/handles — corrupts native state. Symptoms observed in
        # production and reproduced locally: intermittent "PDFium: Data format
        # error" (whole documents skipped as not_extracted) and native access
        # violations escalating to process exit 0x80000003. Rasterization must
        # be serialized process-wide; deskew/classification below stay parallel.
        with _PDFIUM_RENDER_LOCK:
            pil_img = pdf_page.to_image(resolution=PIPELINE_PDF_DPI).original.convert('RGB')
        img_array = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Universal upstream deskew: always on, independent of the classifier flag.
        img_bgr, deskew_deg = _deskew_bgr(img_bgr)

        if SCAN_CLASSIFIER_ENABLED:
            img_bgr, signals, branch, scan_conf, branch_ms = _classify_and_branch(img_bgr)
        else:
            img_bgr = _normalize_pdf_page_for_ocr(img_bgr)
            signals, branch, scan_conf, branch_ms = {}, "cg_normal_legacy", 1.0, 0

        h, w = img_bgr.shape[:2]
        patches = calculate_patches(w, h)

        return PageData(
            page_id=page_id,
            document_id=document_id,
            batch_id=batch_id,
            page_number=page_number,
            image=img_bgr,
            width=w,
            height=h,
            storage_path=file_path,
            patches=patches,
            status="READY_FOR_EXTRACTION",
            metadata={
                "source": "pdf",
                "pdf_page_index": page_number - 1,
                "render_dpi": PIPELINE_PDF_DPI,
                "target_dpi": PIPELINE_PDF_TARGET_DPI,
                "pdf_preprocessed": True,
                "scan_branch": branch,
                "scan_confidence": scan_conf,
                "classifier_signals": signals,
                "deskew_angle_deg": deskew_deg,
                "branch_preprocessing_ms": branch_ms,
            },
        )
    except Exception as e:
        # Re-raise with page context attached so the caller's log line is useful
        # without needing to inspect the original traceback type.
        raise RuntimeError(f"render_failed page_id={page_id}: {e}") from e


def _render_pdf_pages_parallel(
    page_objs: list,
    document_id: str,
    batch_id: str,
    file_path: str,
) -> List[PageData]:
    """Render PDF pages concurrently with bounded rolling submission, per-page
    timeout, and pool recycle after repeated consecutive timeouts.

    Returns pages in page-number order. Every page number from 1..len(page_objs)
    is guaranteed an entry in the result — either a real PageData or a
    placeholder — the returned list length always equals len(page_objs).
    """
    num_pages = len(page_objs)
    actual_workers = min(PIPELINE_PDF_RENDER_THREADS, num_pages)

    results_by_page_number: Dict[int, PageData] = {}
    consecutive_timeouts = 0

    executor = ThreadPoolExecutor(max_workers=actual_workers)

    try:
        page_iter = enumerate(page_objs, start=1)  # (page_number, pdf_page)
        futures: Dict[Any, int] = {}      # future -> page_number
        started: Dict[Any, float] = {}    # future -> submit time

        def submit_next() -> Optional[Any]:
            try:
                page_number, pdf_page = next(page_iter)
            except StopIteration:
                return None
            future = executor.submit(
                _render_one_page, pdf_page, page_number, document_id, batch_id, file_path
            )
            futures[future] = page_number
            started[future] = time.time()
            return future

        pending = set()
        for _ in range(actual_workers):
            f = submit_next()
            if f is None:
                break
            pending.add(f)

        while pending:
            done, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)

            for future in done:
                pending.discard(future)
                page_number = futures.pop(future, None)
                started.pop(future, None)
                if page_number is None:
                    continue

                # A page number can only be finalized once. A future that
                # completes after its slot was already filled by a timeout
                # placeholder (see below — threads can't be force-cancelled,
                # so a "timed out" thread keeps running and will eventually
                # complete on its own) must not overwrite the placeholder.
                if page_number in results_by_page_number:
                    logging.warning(
                        "late_render_result_discarded document_id=%s page_number=%s "
                        "reason=already_finalized",
                        document_id, page_number,
                    )
                    continue

                try:
                    page = future.result()
                    results_by_page_number[page_number] = page
                    consecutive_timeouts = 0
                except Exception as e:
                    logging.warning(
                        "PDF page render failed document_id=%s page_number=%s: %s",
                        document_id, page_number, e,
                    )
                    results_by_page_number[page_number] = _placeholder_page_data(
                        document_id, batch_id, page_number, file_path,
                        reason="pdf_page_render_failed", detail=str(e)[:200],
                    )

                nf = submit_next()
                if nf is not None:
                    pending.add(nf)

            # Timeout sweep: check elapsed time on everything still pending.
            # NOTE: future.cancel() does nothing for a thread that has already
            # started running — Python threads cannot be force-stopped. On
            # timeout we mark the page as a placeholder and free our own
            # bookkeeping slot so the rolling window keeps moving, but the
            # underlying thread keeps running in the background until it
            # finishes naturally. The `if page_number in results_by_page_number`
            # check above is what protects us from that late completion
            # clobbering the placeholder.
            now = time.time()
            for future in list(pending):
                page_number = futures.get(future)
                if page_number is None:
                    continue
                elapsed = now - started.get(future, now)
                if elapsed < PIPELINE_PDF_RENDER_TIMEOUT_SEC:
                    continue

                future.cancel()  # best-effort; no-op if already running
                pending.discard(future)
                futures.pop(future, None)
                started.pop(future, None)

                logging.warning(
                    "PDF page render timeout document_id=%s page_number=%s after %.1fs",
                    document_id, page_number, elapsed,
                )
                results_by_page_number[page_number] = _placeholder_page_data(
                    document_id, batch_id, page_number, file_path,
                    reason="pdf_page_render_timeout",
                    detail=f"timeout_after_{PIPELINE_PDF_RENDER_TIMEOUT_SEC}s",
                )
                consecutive_timeouts += 1

                if consecutive_timeouts >= PIPELINE_PDF_RENDER_RECYCLE_THRESHOLD:
                    logging.warning(
                        "PDF render pool recycle triggered document_id=%s "
                        "consecutive_timeouts=%d threshold=%d page_number=%s",
                        document_id, consecutive_timeouts,
                        PIPELINE_PDF_RENDER_RECYCLE_THRESHOLD, page_number,
                    )
                    # shutdown(wait=False): do NOT block waiting for any
                    # still-running (abandoned) threads to finish — that
                    # would defeat the point of recycling on a hang. The old
                    # executor's threads die on their own or at process exit;
                    # any of their results that arrive later are discarded by
                    # the "already_finalized" check above.
                    executor.shutdown(wait=False)
                    executor = ThreadPoolExecutor(max_workers=actual_workers)
                    consecutive_timeouts = 0
                    # Re-submit to fill the new executor's slots up to actual_workers,
                    # since shutting down the old one didn't free any *new* page
                    # submissions on its own.
                    while len(pending) < actual_workers:
                        nf = submit_next()
                        if nf is None:
                            break
                        pending.add(nf)

                # Try to keep the window full after a timeout-driven slot free.
                nf = submit_next()
                if nf is not None:
                    pending.add(nf)

    finally:
        executor.shutdown(wait=False)

    # Guarantee every page number 1..num_pages has an entry. If anything
    # was missed by a bookkeeping edge case, backfill with a placeholder
    # rather than silently shortening the page list.
    pages: List[PageData] = []
    for page_number in range(1, num_pages + 1):
        page = results_by_page_number.get(page_number)
        if page is None:
            logging.warning(
                "PDF page missing from results, backfilling placeholder "
                "document_id=%s page_number=%s", document_id, page_number,
            )
            page = _placeholder_page_data(
                document_id, batch_id, page_number, file_path,
                reason="pdf_page_unaccounted", detail="missing_from_results_by_page_number",
            )
        pages.append(page)

    return pages


def _open_pdf_pages(file_path: str) -> List[Any]:
    """Open a PDF and return its page objects as a plain list so the call
    can be wrapped in a ThreadPoolExecutor with a hard timeout."""
    with pdfplumber.open(file_path) as pdf:
        return list(pdf.pages)


def _extract_pdf_pages(document: DocumentData, file_path: str) -> List[PageData]:
    try:
        # pdfplumber.open() has no native timeout. Use a single-worker thread
        # pool so we can set a hard ceiling via future.result(timeout=...).
        # This prevents a single corrupted / pathological PDF from stalling
        # the whole ingestion pipeline indefinitely.
        with ThreadPoolExecutor(max_workers=1) as exc:
            future = exc.submit(_open_pdf_pages, file_path)
            page_objs = future.result(timeout=PIPELINE_PDF_OPEN_TIMEOUT_SEC)
        num_pages = len(page_objs)

        if num_pages == 0:
            return []

        if num_pages == 1:
            # No pool for single-page PDFs — sequential path, unchanged behavior.
            try:
                page = _render_one_page(
                    page_objs[0], 1, document.document_id, document.batch_id, file_path
                )
                return [page]
            except Exception as e:
                logging.warning(f"PDF page render failed for {file_path} page 1: {e}")
                return _create_placeholder_page(
                    document, file_path, reason="pdf_page_render_failed", detail=str(e)[:200]
                )

        return _render_pdf_pages_parallel(
            page_objs, document.document_id, document.batch_id, file_path
        )

    except TimeoutError:
        logging.warning(
            "PDF open timeout document_id=%s file=%s after %ss",
            document.document_id, file_path, PIPELINE_PDF_OPEN_TIMEOUT_SEC,
        )
        return _create_placeholder_page(
            document, file_path,
            reason="pdf_open_timeout",
            detail=f"timeout_after_{PIPELINE_PDF_OPEN_TIMEOUT_SEC}s",
        )

    except Exception as e:
        logging.warning(f"PDF extract failed for {file_path}: {e}")
        return _create_placeholder_page(document, file_path, reason="pdf_extract_failed", detail=str(e)[:200])


def _extract_image_page(document: DocumentData, file_path: str) -> List[PageData]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in PDF_EXT:
        return _extract_pdf_pages(document, file_path)
    img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {file_path}")
    
    # 3-channel BGR 
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    # Universal upstream deskew matches the PDF path; uploaded images go through the same
    # orientation correction before any downstream step. Branch classification itself stays
    # PDF-only per design (images skip the classifier).
    img_bgr, deskew_deg = _deskew_bgr(img_bgr)

    # Page-quality capture for image inputs. Branch classification stays PDF-only per the
    # design above, but the QUALITY measurement is independent of it - measure_page needs
    # only a greyscale raster. Without this the capture misses every uploaded image, and
    # work instructions carry a lot of them, so the calibration would be fitted to PDFs
    # alone and would not describe that part of the corpus at all.
    _pq_row = None
    if _page_quality_enabled():
        try:
            _mp = _page_quality_measure()
            if _mp is not None:
                _pq_row = _mp(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
                _ph = _page_quality_phash()
                if _ph is not None:
                    _pq_row["phash"] = _ph(img_bgr)
        except Exception as _pq_exc:
            logging.debug("page-quality capture skipped for image: %s", _pq_exc, exc_info=True)
            _pq_row = None

    h, w = img_bgr.shape[:2]
    patches = calculate_patches(w, h)
    page_id = f"{document.document_id}_PG001"
    
    page = PageData(
        page_id=page_id,
        document_id=document.document_id,
        batch_id=document.batch_id,
        page_number=1,
        image=img_bgr,
        width=w,
        height=h,
        storage_path=file_path,
        patches=patches,
        status="READY_FOR_EXTRACTION",
        metadata={
            "source": "image",
            "image_preprocessed": False,
            "deskew_angle_deg": deskew_deg,
            # Same shape the PDF path uses, so page_quality_export reads one structure for
            # both. scan_branch is deliberately absent: images skip the classifier, and
            # inventing a branch verdict here would fake a measurement we did not take.
            "classifier_signals": ({"page_quality": _pq_row} if _pq_row else {}),
        },
    )
    
    return [page]


def _create_placeholder_page(
    document: DocumentData,
    file_path: str,
    reason: str = "unsupported_format",
    detail: Optional[str] = None,
) -> List[PageData]:
    page_id = f"{document.document_id}_PG001"
    placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
    meta = {"source": "placeholder", "reason": reason}
    if detail:
        meta["detail"] = detail
    page = PageData(
        page_id=page_id,
        document_id=document.document_id,
        batch_id=document.batch_id,
        page_number=1,
        image=placeholder,
        width=100,
        height=100,
        storage_path=file_path,
        patches=0,
        status="UNSUPPORTED",
        metadata=meta,
    )
    return [page]


def _placeholder_page_data(
    document_id: str,
    batch_id: str,
    page_number: int,
    file_path: str,
    reason: str,
    detail: Optional[str] = None,
) -> PageData:
    """Single-page placeholder for use inside multi-page parallel rendering,
    where the page number isn't necessarily 1 and we don't have (or want to
    construct) a DocumentData. Mirrors _create_placeholder_page's shape."""
    page_id = f"{document_id}_PG{page_number:03d}"
    placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
    meta = {"source": "placeholder", "reason": reason}
    if detail:
        meta["detail"] = detail
    return PageData(
        page_id=page_id,
        document_id=document_id,
        batch_id=batch_id,
        page_number=page_number,
        image=placeholder,
        width=100,
        height=100,
        storage_path=file_path,
        patches=0,
        status="UNSUPPORTED",
        metadata=meta,
    )


def _render_text_to_image(
    text: str,
    width: int = 1200,
    line_height: int = 28,
    font_size: int = 18,
    margin: int = 40,
) -> Image.Image:
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
    lines = [s for s in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if s.strip() or text.count("\n") > 0]
    if not lines:
        lines = [""]
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    line_widths = [draw.textbbox((0, 0), ln, font=font)[2] for ln in lines]
    content_w = min(width - 2 * margin, max(line_widths) + 20 if line_widths else width - 2 * margin)
    content_h = len(lines) * line_height
    img_w = content_w + 2 * margin
    img_h = content_h + 2 * margin
    pil_img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(pil_img)
    for i, ln in enumerate(lines):
        draw.text((margin, margin + i * line_height), ln, fill=(0, 0, 0), font=font)
    return pil_img


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    img_array = np.array(pil_img)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def _extract_txt_pages(document: DocumentData, file_path: str) -> List[PageData]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        logging.warning(f"Failed to read {file_path}: {e}")
        return _create_placeholder_page(document, file_path)
    chunks = [t.strip() for t in text.split("\f") if t.strip()]
    if not chunks:
        chunks = [text or "(empty)"]
    pages = []
    for i, chunk in enumerate(chunks):
        pil_img = _render_text_to_image(chunk)
        img_bgr = _pil_to_bgr(pil_img)
        h, w = img_bgr.shape[:2]
        page_number = i + 1
        page_id = f"{document.document_id}_PG{page_number:03d}"
        page = PageData(
            page_id=page_id,
            document_id=document.document_id,
            batch_id=document.batch_id,
            page_number=page_number,
            image=img_bgr,
            width=w,
            height=h,
            storage_path=file_path,
            patches=calculate_patches(w, h),
            status="READY_FOR_EXTRACTION",
            metadata={"source": "txt", "chunk_index": i},
        )
        pages.append(page)
    return pages


def _extract_csv_pages(document: DocumentData, file_path: str) -> List[PageData]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.reader(f))
    except Exception as e:
        logging.warning(f"Failed to read CSV {file_path}: {e}")
        return _create_placeholder_page(document, file_path)
    if not rows:
        return _extract_txt_pages(document, file_path)
    cell_w = 120
    cell_h = 28
    margin = 40
    cols = max(len(r) for r in rows)
    img_w = margin * 2 + cols * cell_w
    img_h = margin * 2 + len(rows) * cell_h
    pil_img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
    draw = ImageDraw.Draw(pil_img)
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            x = margin + ci * cell_w + 4
            y = margin + ri * cell_h + 4
            draw.text((x, y), str(cell)[:30], fill=(0, 0, 0), font=font)
    img_bgr = _pil_to_bgr(pil_img)
    h, w = img_bgr.shape[:2]
    page_id = f"{document.document_id}_PG001"
    page = PageData(
        page_id=page_id,
        document_id=document.document_id,
        batch_id=document.batch_id,
        page_number=1,
        image=img_bgr,
        width=w,
        height=h,
        storage_path=file_path,
        patches=calculate_patches(w, h),
        status="READY_FOR_EXTRACTION",
        metadata={"source": "csv"},
    )
    return [page]


def _extract_docx_pages(document: DocumentData, file_path: str) -> List[PageData]:
    try:
        from docx import Document as DocxDocument
    except ImportError:
        logging.warning("python-docx not installed; .docx files will use placeholder. pip install python-docx")
        return _create_placeholder_page(document, file_path, reason="docx_import_failed")
    path_abs = os.path.abspath(file_path)
    if not os.path.isfile(path_abs):
        logging.warning(f"DOCX file not found: {path_abs}")
        return _create_placeholder_page(document, file_path, reason="docx_file_not_found", detail=path_abs[:200])
    try:
        doc = DocxDocument(path_abs)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            paragraphs = ["(empty docx)"]
        text = "\n".join(paragraphs)
        # Reuse text rendering: write to a temp file and use _extract_txt_pages, or render here
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
            tf.write(text)
            tmp_path = tf.name
        try:
            return _extract_txt_pages(document, tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        logging.warning(f"Failed to read docx {file_path}: {e}")
        return _create_placeholder_page(document, file_path, reason="docx_extract_failed", detail=str(e)[:200])


def _extract_doc_pages(document: DocumentData, file_path: str) -> List[PageData]:
    tmp_dir = tempfile.mkdtemp()
    try:
        cmd = [
            "soffice", "--headless", "--convert-to", "pdf", "--outdir", tmp_dir, file_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logging.warning(f"LibreOffice conversion failed for {file_path}: {e}. Install LibreOffice for .doc support.")
            return _create_placeholder_page(document, file_path)
        base = os.path.splitext(os.path.basename(file_path))[0] + ".pdf"
        out_pdf = os.path.join(tmp_dir, base)
        if os.path.isfile(out_pdf):
            return _extract_pdf_pages(document, out_pdf)
        return _create_placeholder_page(document, file_path)
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def _expand_archive_entries(
    archive_abs: str,
    logical_base: str,
    temp_dirs_out: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Expand zip/tar to a temp dir. Returns (absolute_path, path_relative_to_batch_root)."""
    ext = os.path.splitext(archive_abs)[1].lower()
    if ext not in ARCHIVE_EXT:
        return []
    logical_base = _logical_path_posix(logical_base).strip("/")
    entries: List[Tuple[str, str]] = []

    def _member_rel_path(inner_name: str) -> str:
        inner = _logical_path_posix(inner_name).lstrip("./")
        if logical_base:
            return f"{logical_base}/{inner}" if inner else logical_base
        return inner

    if ext == ".zip":
        try:
            with zipfile.ZipFile(archive_abs, "r") as zf:
                # Decide BEFORE extracting. This previously called zf.extractall(tmp_dir) and
                # filtered afterwards, so every member was written to disk in full -- including
                # the ones immediately discarded as unsupported -- and the temp directory is only
                # released when the whole RUN ends, not per document. On a corpus of archives
                # holding raster scans that is the full uncompressed size of every archive, sitting
                # in %TEMP% for the life of the run, for files the runtime never opens.
                wanted = []
                for name in zf.namelist():
                    if name.startswith("__MACOSX") or "/." in name or "\\." in name:
                        continue
                    base = os.path.basename(name.rstrip("/"))
                    if not base:
                        continue
                    member_ext = os.path.splitext(base)[1].lower()
                    if member_ext not in SUPPORTED_EXT:
                        if member_ext and not base.startswith("."):
                            logging.warning("[INGEST] Skipping unsupported file type %r inside zip: %s", member_ext, base)
                        continue
                    wanted.append(name)
                # Nothing usable in this archive: do not create a temp directory for it either.
                if not wanted:
                    return []
                tmp_dir = tempfile.mkdtemp()
                if temp_dirs_out is not None:
                    temp_dirs_out.append(tmp_dir)
                zf.extractall(tmp_dir, members=wanted)
                for name in wanted:
                    full = os.path.normpath(os.path.join(tmp_dir, name))
                    if os.path.isfile(full):
                        entries.append((full, _member_rel_path(name)))
        except Exception as e:
            logging.warning(f"Failed to expand zip {archive_abs}: {e}")
    elif ext == ".tar":
        try:
            with tarfile.open(archive_abs, "r:*") as tf:
                # Same correction as the zip path above: select members first, extract only
                # those. tarfile.extractall takes TarInfo objects rather than names.
                wanted = []
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name
                    if name.startswith("__MACOSX") or "/." in name:
                        continue
                    base = os.path.basename(name)
                    if not base:
                        continue
                    member_ext = os.path.splitext(base)[1].lower()
                    if member_ext not in SUPPORTED_EXT:
                        if member_ext and not base.startswith("."):
                            logging.warning("[INGEST] Skipping unsupported file type %r inside tar: %s", member_ext, base)
                        continue
                    wanted.append(member)
                if not wanted:
                    return []
                tmp_dir = tempfile.mkdtemp()
                if temp_dirs_out is not None:
                    temp_dirs_out.append(tmp_dir)
                tf.extractall(tmp_dir, members=wanted)
                for member in wanted:
                    full = os.path.normpath(os.path.join(tmp_dir, member.name))
                    if os.path.isfile(full):
                        entries.append((full, _member_rel_path(member.name)))
        except Exception as e:
            logging.warning(f"Failed to expand tar {archive_abs}: {e}")
    return entries


# UTILITY FUNCTIONS

def get_batch_summary(batch: BatchData) -> Dict[str, Any]:
    return {
        "batch_id": batch.batch_id,
        "status": batch.status,
        "document_count": batch.document_count,
        "total_pages": batch.total_pages,
        "documents": [
            {
                "document_id": doc.document_id,
                "page_count": len(doc.pages),
                "status": doc.status
            }
            for doc in batch.documents
        ]
    }


def iterate_pages(batch: BatchData):
    for document in batch.documents:
        for page in document.pages:
            yield document, page

# MAIN ENTRY POINT (for testing)

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Test ingestion
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "debug"
    
    if os.path.exists(test_dir):
        batch = process_input_local(test_dir)
        summary = get_batch_summary(batch)
        print(json.dumps(summary, indent=2))
    else:
        print(f"Directory not found: {test_dir}")