"""Page-quality measurement — runs on BOTH sides of the air gap.

Why this module exists
----------------------
Domain page images cannot leave the production device (IP). So the degradation statistics
we need in order to calibrate augmentation have to be measured *there* and travel back as
numbers only.

Calibration means comparing the production distribution against the distribution our
augmented public pages produce. That comparison is only meaningful if both sides measure
**identically** — so this one module is run in both places rather than each side
implementing from a description.

    production (inside the sidecar):  measure_page(gray, table_boxes) -> row per page
    here (on augmented DocLayNet):    same call, same columns
    then: match the histograms, and the augmentation is fitted to the real domain.

Dependencies: cv2 + numpy only. No torch, no project imports. Drop it into the sidecar
as-is.

Output: a flat dict of floats/ints, one row per page. ~40 columns, so 10,000 pages is
about 3 MB of parquet — nothing to transfer.

Every metric here maps to a specific augmentation knob; see AUGMENTATION_TARGET below.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SCHEMA_VERSION = 2

# What each column is FOR. Three different consumers, and conflating them is the mistake
# that makes a column look like dead weight:
#
#   AUGMENTATION_TARGET - sets a degradation knob (clean page -> dirty page).
#   SCALE_TARGET        - describes apparent feature scale: how large text is RELATIVE to
#                         the page. A 1025px DocLayNet page and a 1024 tile of an A1 drawing
#                         differ here by 2-3x, which is the largest appearance gap between
#                         the two domains and has nothing to do with degradation.
#
#                         CORRECTED 2026-11-06, having asserted otherwise earlier: this is
#                         NOT fixed by resampling the page. RTDetrImageProcessor forces every
#                         input to 640x640 (train.py:338), so a uniform resample is exactly
#                         cancelled - measured 11.00 px text at x0.40, x0.60, x1.00 and x1.60.
#                         What does move it is the FIELD OF VIEW: the same page padded onto a
#                         1.5x / 2.0x canvas measures 7.00 / 5.00. Apparent scale is
#                         text_px / page_extent, so only changing the extent changes it.
#                         (A.Downscale does nothing here either - it resamples down and back
#                         up to the same dimensions, costing sharpness only.)
#   DOMAIN_TARGET       - describes page/class APPEARANCE. No degradation knob can act on
#                         these; they inform corpus composition (merge.py takes per-source
#                         `repeats`/`samples`) and synthetic page generation. Class set is
#                         never changed - these adjust how the existing classes look.
#
# A column in none of the three is an intermediate or a diagnostic.

SCALE_TARGET = {
    "text_height_p50": "apparent text scale (px on a 1024-page). Matched by changing the "
                       "public page's FIELD OF VIEW, not by resampling it - see above",
}

DOMAIN_TARGET = {
    "rule_len_total":   "ruled length per page - the strongest line-art vs prose signal",
    "ink_coverage":     "ink fraction - line-art sits low; prose does not",
    "text_cc_count":    "sparse annotation vs dense paragraphs",
    "junctions_expected": "ruled-cell density",
    "page_w":           "sheet size and orientation (landscape A0/A1 vs portrait A4)",
    "page_h":           "sheet size and orientation",
    "diag_ink_frac":    "diagonal ink share - hatching, leaders, section lines. Zero on prose",
    "orient_entropy":   "how many orientations the ink occupies, prose being near-degenerate",
    "stroke_modes":     "distinct stroke-width modes - ISO line weights are multi-modal",
    "streak_count":     "scanner streaks per page (informs n_ops / tier_probs, not a knob)",
    "streak_strength":  "how pronounced those streaks are",
}

# metric -> which augmentation parameter it calibrates
AUGMENTATION_TARGET = {
    "gap_per_100px":        "LineBreakage.gaps_per_100px",
    "gap_len_p50":          "LineBreakage.gap_len (lower)",
    "gap_len_p90":          "LineBreakage.gap_len (upper)",
    "stroke_w_mean":        "StrokeJitter kernel size",
    "stroke_w_cv":          "StrokeJitter anisotropy / spread",
    "border_missing_frac":  "BorderDropout.border_p",
    "junction_missing_frac": "CellStructureNoise.junction_frac",
    "speckle_per_megapx":   "PhotocopyNoise.sp_frac",
    "laplacian_var":        "GaussianBlur limit",
    "contrast_p95_p05":     "RandomBrightnessContrast / RandomGamma range",
    "bg_level":             "paper value used by every erase op",
    "illum_range":          "ScannerIllumination.strength",
    "bleed_corr":           "BleedThrough.alpha",
    "is_inverted":          "Inversion probability",
    "skew_deg":             "Affine.rotate range",
    "text_height_p50":      "effective resolution / Downscale range",
}


# --------------------------------------------------------------------------- helpers
def _ink_mask(gray: np.ndarray) -> np.ndarray:
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m


def _rule_masks(ink: np.ndarray, min_len: int) -> Tuple[np.ndarray, np.ndarray]:
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, min_len), 1))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, min_len)))
    return (cv2.morphologyEx(ink, cv2.MORPH_OPEN, h_k),
            cv2.morphologyEx(ink, cv2.MORPH_OPEN, v_k))


def _runs_of_zero(row: np.ndarray) -> List[int]:
    """Lengths of contiguous zero runs strictly inside the first..last nonzero span."""
    nz = np.flatnonzero(row)
    if nz.size < 2:
        return []
    seg = row[nz[0]:nz[-1] + 1]
    out, run = [], 0
    for v in seg:
        if v:
            if run:
                out.append(run)
                run = 0
        else:
            run += 1
    return out


# --------------------------------------------------------------------------- metrics
def _gap_stats(h_mask: np.ndarray, v_mask: np.ndarray, scale: float,
               max_gap_frac: float = 0.15,
               min_ink: int = 20, min_span: int = 40) -> Dict[str, float]:
    """Gap-length distribution along detected rules.

    The most important measurement here: it sets LineBreakage directly, and line breakage
    is the degradation that matters most for engineering tables.

    Scanned per ROW/COLUMN, not per connected component. A broken rule splits into several
    components, so component-local analysis sees no gaps at all - the first version of this
    function returned 0.00 on pages with heavy, visible breakage. Scanning the full row
    spans the splits.

    Zero-runs longer than `max_gap_frac` of the row's ink span are treated as genuine
    separation (two tables side by side, a column of whitespace) rather than breakage.
    Lengths are normalised to a 1024px page so differing DPI is comparable across sites.

    `min_ink` and `min_span` MUST arrive already scaled by the page-scale factor. They are
    pixel counts, and every length reported here is divided by `scale`; leaving them as raw
    constants made the rate page-size dependent, which is fatal for calibration. Measured on
    one page resized, same configured breakage throughout:

        long side   1023   1841   3507   4962   7020
        measured    0.076  0.116  0.379  0.544  1.339     (configured: 0.100)

    i.e. correct only near 1800px, and 13x over at A1/300dpi. Production rasters are
    3300-14000px, so uncorrected this would set the degrader far more aggressive than
    reality - the exact failure degrade.py:78-100 records having already corrected by hand.
    """
    gaps: List[int] = []
    total_len = 0

    for mask, axis in ((h_mask, 1), (v_mask, 0)):
        if not mask.any():
            continue
        arr = mask if axis == 1 else mask.T          # scan rows either way
        ink_per_line = (arr > 0).sum(axis=1)
        # lines carrying a real rule, not stray ink
        for idx in np.flatnonzero(ink_per_line > min_ink):
            row = (arr[idx] > 0).astype(np.uint8)
            nz = np.flatnonzero(row)
            if nz.size < 2:
                continue
            span = int(nz[-1] - nz[0] + 1)
            if span < min_span:
                continue
            limit = max(2, int(span * max_gap_frac))
            runs = _runs_of_zero(row)
            kept = [g for g in runs if g <= limit]
            gaps.extend(kept)
            # the ink actually present, not the raw span
            total_len += span - sum(g for g in runs if g > limit)

    if not gaps or total_len == 0:
        return {"gap_per_100px": 0.0, "gap_len_p50": 0.0, "gap_len_p90": 0.0,
                "gap_count": 0, "rule_len_total": float(total_len / scale)}

    g = np.asarray(gaps, dtype=np.float32) / scale
    norm_len = total_len / scale
    return {
        "gap_per_100px": float(len(gaps) / norm_len * 100.0),
        "gap_len_p50": float(np.percentile(g, 50)),
        "gap_len_p90": float(np.percentile(g, 90)),
        "gap_count": int(len(gaps)),
        "rule_len_total": float(norm_len),
    }


def _stroke_stats(ink: np.ndarray, scale: float) -> Dict[str, float]:
    """Stroke width via distance transform — sets StrokeJitter kernel size."""
    if not ink.any():
        return {"stroke_w_mean": 0.0, "stroke_w_cv": 0.0}
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    vals = dt[dt > 0]
    if vals.size == 0:
        return {"stroke_w_mean": 0.0, "stroke_w_cv": 0.0}
    w = vals * 2.0 / scale
    mean = float(w.mean())
    return {"stroke_w_mean": mean,
            "stroke_w_cv": float(w.std() / mean) if mean > 1e-6 else 0.0}


def _border_completeness(gray: np.ndarray, ink: np.ndarray,
                         table_boxes: Optional[Sequence[Sequence[float]]]) -> Dict[str, float]:
    """Fraction of table border edges that are largely absent -> BorderDropout.border_p.

    Needs the table boxes, which OCULI already has. Without them this returns NaN rather
    than guessing.
    """
    if not table_boxes:
        return {"border_missing_frac": float("nan"), "border_edges_checked": 0}
    missing = checked = 0
    H, W = ink.shape[:2]
    for box in table_boxes:
        x, y, w, h = (float(v) for v in box[:4])
        if w < 20 or h < 20:
            continue
        band = max(2, int(round(min(w, h) * 0.01)))
        edges = {
            "top": ink[max(0, int(y) - band):int(y) + band, int(x):int(x + w)],
            "bottom": ink[int(y + h) - band:min(H, int(y + h) + band), int(x):int(x + w)],
            "left": ink[int(y):int(y + h), max(0, int(x) - band):int(x) + band],
            "right": ink[int(y):int(y + h), int(x + w) - band:min(W, int(x + w) + band)],
        }
        for e in edges.values():
            if e.size == 0:
                continue
            checked += 1
            axis = 0 if e.shape[0] <= e.shape[1] else 1
            present = (e > 0).any(axis=axis).mean()
            if present < 0.5:          # less than half the edge has ink
                missing += 1
    return {"border_missing_frac": float(missing / checked) if checked else float("nan"),
            "border_edges_checked": int(checked)}


def _junction_stats(h_mask: np.ndarray, v_mask: np.ndarray) -> Dict[str, float]:
    """How many rule crossings survive -> CellStructureNoise.junction_frac.

    Expected crossings are where an h-rule row and a v-rule column coincide; actual are
    where both masks have ink. The shortfall is broken/merged cell structure.
    """
    h_rows = np.flatnonzero((h_mask > 0).sum(axis=1) > h_mask.shape[1] * 0.1)
    v_cols = np.flatnonzero((v_mask > 0).sum(axis=0) > v_mask.shape[0] * 0.1)
    if h_rows.size == 0 or v_cols.size == 0:
        return {"junction_missing_frac": float("nan"), "junctions_expected": 0}

    def _cluster(idx, tol=3):
        out, cur = [], [idx[0]]
        for v in idx[1:]:
            if v - cur[-1] <= tol:
                cur.append(v)
            else:
                out.append(int(np.mean(cur))); cur = [v]
        out.append(int(np.mean(cur)))
        return out

    rows, cols = _cluster(h_rows), _cluster(v_cols)
    both = cv2.bitwise_and(h_mask, v_mask)
    expected = len(rows) * len(cols)
    if expected == 0:
        return {"junction_missing_frac": float("nan"), "junctions_expected": 0}
    found = 0
    for r in rows:
        for c in cols:
            r0, r1 = max(0, r - 4), min(both.shape[0], r + 5)
            c0, c1 = max(0, c - 4), min(both.shape[1], c + 5)
            if both[r0:r1, c0:c1].any():
                found += 1
    return {"junction_missing_frac": float(1.0 - found / expected),
            "junctions_expected": int(expected)}


def _tone_noise_illum(gray: np.ndarray, ink: np.ndarray) -> Dict[str, float]:
    g = gray.astype(np.float32)
    p05, p50, p95 = np.percentile(g, [5, 50, 95])
    bg = float(np.percentile(g, 95))

    # speckle: tiny connected components per megapixel -> PhotocopyNoise.sp_frac
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    small = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] <= 4)
    megapx = gray.size / 1e6

    # illumination: background level across a coarse grid
    k = 8
    hh, ww = gray.shape[0] // k, gray.shape[1] // k
    tiles = [float(np.percentile(g[i * hh:(i + 1) * hh, j * ww:(j + 1) * ww], 90))
             for i in range(k) for j in range(k)
             if g[i * hh:(i + 1) * hh, j * ww:(j + 1) * ww].size]
    illum_range = float(max(tiles) - min(tiles)) if tiles else 0.0

    return {
        "bg_level": bg,
        "ink_level": float(p05),
        "contrast_p95_p05": float(p95 - p05),
        "median_level": float(p50),
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "speckle_per_megapx": float(small / megapx) if megapx else 0.0,
        "ink_coverage": float((ink > 0).mean()),
        "illum_range": illum_range,
        "is_inverted": float(p50 < 128),
    }


def _bleed_and_skew(gray: np.ndarray, ink: np.ndarray) -> Dict[str, float]:
    """Mirror correlation as a bleed-through proxy; Hough-based skew estimate."""
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    mir = cv2.flip(small, 1)
    a, b = small - small.mean(), mir - mir.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    bleed = float((a * b).sum() / denom) if denom > 1e-6 else 0.0

    skew = float("nan")
    lines = cv2.HoughLines(cv2.Canny(gray, 50, 150, apertureSize=3), 1, np.pi / 360, 200)
    if lines is not None and len(lines):
        angs = []
        for l in lines[:200]:
            th = float(l[0][1]) * 180.0 / np.pi
            for ref in (0.0, 90.0, 180.0):
                if abs(th - ref) < 15:
                    angs.append(th - ref)
        if angs:
            skew = float(np.median(angs))
    return {"bleed_corr": bleed, "skew_deg": skew}


def _text_height(ink: np.ndarray, h_mask: np.ndarray, v_mask: np.ndarray,
                 scale: float) -> Dict[str, float]:
    """Median height of non-rule ink components — an effective-resolution proxy."""
    rules = cv2.bitwise_or(h_mask, v_mask)
    text = cv2.bitwise_and(ink, cv2.bitwise_not(rules))
    n, _, stats, _ = cv2.connectedComponentsWithStats(text, 8)
    hs = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, n)
          if 3 <= stats[i, cv2.CC_STAT_HEIGHT] <= 80 and stats[i, cv2.CC_STAT_AREA] >= 6]
    return {"text_height_p50": float(np.median(hs) / scale) if hs else float("nan"),
            "text_cc_count": int(len(hs))}


def _orientation_stats(gray: np.ndarray, ink: np.ndarray) -> Dict[str, float]:
    """Where the ink points. The strongest line-art vs prose signal there is.

    _rule_masks uses axis-aligned MORPH_RECT kernels, so every 45-degree hatch, dimension
    leader and section line is invisible to it. On an engineering drawing that is a large
    share of the ink; on a paper it is near zero.

    Measured by morphological opening with oriented kernels, the same idiom _rule_masks
    uses, rather than a gradient-angle histogram: a first attempt with Sobel angles scored
    1.000 diagonal on a page of purely horizontal and vertical rules, because the strong-
    gradient pixels cluster at line termini and crossings rather than along the strokes.
    An opening either fits a run of ink at that angle or it does not, which is predictable
    and testable.
    """
    out = {"diag_ink_frac": float("nan"), "orient_entropy": float("nan")}
    total_ink = float(np.count_nonzero(ink))
    if total_ink <= 0:
        return out

    k = 15                                      # run length a stroke must sustain
    eye = np.eye(k, dtype=np.uint8)
    kernels = {
        "h": cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)),
        "v": cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)),
        "d1": eye,                              # 45 degrees
        "d2": np.fliplr(eye).copy(),            # 135 degrees
    }
    share = {}
    for name, kern in kernels.items():
        kept = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kern)
        share[name] = float(np.count_nonzero(kept)) / total_ink

    out["diag_ink_frac"] = float(share["d1"] + share["d2"])
    tot = sum(share.values())
    if tot > 0:
        p = np.array([share[n] / tot for n in ("h", "v", "d1", "d2")], dtype=np.float64)
        nz = p[p > 0]
        out["orient_entropy"] = float(-(nz * np.log2(nz)).sum() / np.log2(4))
    return out


def _stroke_modality(ink: np.ndarray, scale: float) -> Dict[str, float]:
    """How many distinct stroke widths the page uses.

    ISO line weights (0.25/0.35/0.5/0.7mm) make a drawing's stroke-width histogram
    MULTI-modal. stroke_w_cv reports spread and cannot tell bimodal from noisy-unimodal, so
    it cannot see the difference between a drawing and a smudged paper.
    """
    out = {"stroke_modes": float("nan")}
    if not ink.any():
        return out
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    vals = dt[dt > 0] * 2.0 / max(scale, 1e-6)
    if vals.size < 64:
        return out
    hist, _ = np.histogram(vals, bins=24, range=(0.0, 12.0))
    if hist.sum() <= 0:
        return out
    h = hist.astype(np.float64) / hist.sum()
    h = np.convolve(h, np.array([0.25, 0.5, 0.25]), mode="same")   # damp single-bin noise
    floor = 0.05 * h.max()
    modes = 0
    for i in range(1, len(h) - 1):
        if h[i] >= h[i - 1] and h[i] > h[i + 1] and h[i] > floor:
            modes += 1
    out["stroke_modes"] = float(modes)
    return out


def _streak_stats(gray: np.ndarray, ink: np.ndarray) -> Dict[str, float]:
    """Scanner streaks: narrow sustained deviations in the PAPER, not the ink.

    Measured on the background only, so page content cannot be mistaken for a streak. A
    streak shows up as a column (or row) whose background level departs from its neighbours
    and stays departed down the whole page.

    streak_p in degrade.py is one of the eight knobs neutralised to 1.0, so this calibrates
    no knob directly - it informs n_ops and the tier mix, and serves the closed-loop check
    that augmented pages match the production distribution.
    """
    out = {"streak_count": 0.0, "streak_strength": 0.0}
    paper = cv2.bitwise_not(ink)
    g = gray.astype(np.float32)
    best_n, best_s = 0.0, 0.0
    for axis in (0, 1):
        cnt = (paper > 0).sum(axis=axis).astype(np.float32)
        tot = (g * (paper > 0)).sum(axis=axis)
        valid = cnt > (0.25 * gray.shape[axis])      # enough paper along that line to judge
        if valid.sum() < 32:
            continue
        prof = np.full(cnt.shape, np.nan, dtype=np.float32)
        np.divide(tot, np.maximum(cnt, 1.0), out=prof, where=valid)
        good = prof[valid]
        med = float(np.median(good))
        # robust spread; a streak is a departure well beyond the paper's own variation
        mad = float(np.median(np.abs(good - med))) or 1e-3
        dev = np.abs(prof - med) / (1.4826 * mad)
        hits = np.nan_to_num(dev, nan=0.0) > 6.0
        # count runs, not pixels: one streak is a contiguous group of columns
        runs, prev = 0, False
        for v in hits:
            if v and not prev:
                runs += 1
            prev = bool(v)
        if runs > best_n:
            best_n = float(runs)
            best_s = float(np.nan_to_num(dev, nan=0.0).max())
    out["streak_count"] = best_n
    out["streak_strength"] = best_s if best_n > 0 else 0.0
    return out


# --------------------------------------------------------------------------- entry
def measure_page(gray: np.ndarray,
                 table_boxes: Optional[Sequence[Sequence[float]]] = None,
                 min_rule_len: int = 40,
                 canonical_long_side: int = 1024) -> Dict[str, float]:
    """One page in, one flat row of metrics out.

    `gray`        single-channel uint8 page raster (any size).
    `table_boxes` xywh in page pixels. Optional, but border/junction metrics are NaN
                  without it, and those calibrate two augmentation ops.

    Lengths are normalised to a 1024px page so that pages scanned at different DPI are
    directly comparable between sites.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)

    src_h, src_w = gray.shape[:2]

    # Measure at a CANONICAL size, always. Scaling each pixel threshold by the page factor
    # was tried first and only narrowed the page-size drift (17.6x spread -> 4.5x): Otsu,
    # the opening kernel and the connected-component statistics do not all respond to
    # resolution the same way, so no set of scaled constants makes them agree. Resampling
    # once, up front, removes the dependence by construction - and makes an A0 at 300dpi
    # (14040px) cost the same as an A4.
    #
    # After this, scale == 1.0 and every length below is already in 1024-page units, which
    # is also exactly what text_height_p50 needs to be for SCALE_TARGET to mean anything.
    long_side = max(src_h, src_w)
    if long_side != canonical_long_side and long_side > 0:
        f = canonical_long_side / float(long_side)
        gray = cv2.resize(gray, (max(1, int(round(src_w * f))), max(1, int(round(src_h * f)))),
                          interpolation=cv2.INTER_AREA if f < 1.0 else cv2.INTER_CUBIC)
        # table_boxes arrive in ORIGINAL page pixels; the raster just moved under them.
        if table_boxes:
            table_boxes = [[float(b[0]) * f, float(b[1]) * f,
                            float(b[2]) * f, float(b[3]) * f] for b in table_boxes]

    h, w = gray.shape[:2]
    scale = 1.0
    ink = _ink_mask(gray)
    h_mask, v_mask = _rule_masks(ink, min_rule_len)

    row: Dict[str, float] = {
        "schema_version": SCHEMA_VERSION,
        # the page as it really is, not as measured - needed for aspect/sheet class
        "page_w": int(src_w), "page_h": int(src_h),
        "measured_at": int(canonical_long_side),
    }
    row.update(_gap_stats(h_mask, v_mask, scale))
    row.update(_stroke_stats(ink, scale))
    row.update(_border_completeness(gray, ink, table_boxes))
    row.update(_junction_stats(h_mask, v_mask))
    row.update(_tone_noise_illum(gray, ink))
    row.update(_bleed_and_skew(gray, ink))
    row.update(_text_height(ink, h_mask, v_mask, scale))
    # --- appearance / domain descriptors (schema 2). No degradation knob reads these; see
    # DOMAIN_TARGET. They inform corpus composition and synthesis, and the class set is
    # never changed by them.
    row.update(_orientation_stats(gray, ink))
    row.update(_stroke_modality(ink, scale))
    row.update(_streak_stats(gray, ink))
    return row


def column_names() -> List[str]:
    """Stable column order, for declaring the parquet schema up front."""
    probe = measure_page(np.full((512, 512), 255, np.uint8), [[10, 10, 100, 100]])
    return list(probe.keys())
