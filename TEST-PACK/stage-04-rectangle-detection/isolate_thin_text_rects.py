"""
Isolate table/rect candidates (especially around Material row) and compute thin-text metrics.

Works in two modes:
1) Metadata-only: uses outputs/*/digitised_document.json + preclassifier_stats.json
2) Full crop mode: additionally loads source page image, writes crops, and scores preprocess variants

Usage examples:
  py tools/isolate_thin_text_rects.py --output-root outputs --batch-id BATCH_08962480
  py tools/isolate_thin_text_rects.py --output-root outputs --batch-id BATCH_08962480 --image-root "C:\\path\\to\\images"
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


TARGET_DOC_SUFFIXES = [
    "Screenshot_18-3-2026_101153__jpeg",
    "Screenshot_23-3-2026_82649__jpeg",
    "1759857039776_jpg",
]


@dataclass
class RectCandidate:
    doc_id: str
    table_id: str
    cell_id: str
    poly_id: str
    text: str
    bbox: List[float]
    row_idx: int
    col_idx: int
    row_group: str


def _safe_read_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_doc_suffix(doc_id: str) -> str:
    # "BATCH_08962480_DOC_1759857039776_jpg" -> "1759857039776.jpg"
    suffix = doc_id.split("_DOC_", 1)[-1]
    if suffix.endswith("__jpeg"):
        return suffix[:-6] + ".jpeg"
    if suffix.endswith("_jpg"):
        return suffix[:-4] + ".jpg"
    if suffix.endswith("_png"):
        return suffix[:-4] + ".png"
    return suffix


def _canonical_name(s: str) -> str:
    base = os.path.splitext(s)[0].lower()
    ext = os.path.splitext(s)[1].lower()
    # Normalize separators and collapse repeated underscores, then strip edge underscores.
    for ch in (" ", "-", "."):
        base = base.replace(ch, "_")
    while "__" in base:
        base = base.replace("__", "_")
    base = base.strip("_")
    return f"{base}{ext}"


def _find_image_for_doc(doc_id: str, image_root: Optional[str]) -> Optional[str]:
    if not image_root or not os.path.isdir(image_root):
        return None
    wanted = _normalize_doc_suffix(doc_id).lower()
    wanted_c = _canonical_name(wanted)
    for root, _, files in os.walk(image_root):
        for fn in files:
            fn_l = fn.lower()
            if fn_l == wanted:
                return os.path.join(root, fn)
            if _canonical_name(fn_l) == wanted_c:
                return os.path.join(root, fn)
    return None


def _group_rows(cells: List[dict], tol: float = 6.0) -> List[float]:
    ys = sorted([float((c["bbox"][1] + c["bbox"][3]) / 2.0) for c in cells])
    rows: List[float] = []
    for y in ys:
        if not rows or abs(y - rows[-1]) > tol:
            rows.append(y)
        else:
            rows[-1] = (rows[-1] + y) / 2.0
    return rows


def _nearest_idx(v: float, arr: List[float]) -> int:
    if not arr:
        return 0
    return int(np.argmin([abs(v - a) for a in arr]))


def _compute_thin_metrics(gray: np.ndarray) -> Dict[str, float]:
    out = {
        "sharpness": 0.0,
        "contrast": 0.0,
        "fg_ratio": 0.0,
        "stroke_p10": 0.0,
        "stroke_p50": 0.0,
        "edge_density": 0.0,
        "line_strength_hv": 0.0,
        "thin_score": 0.0,
    }
    if gray is None or gray.size == 0:
        return out
    out["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    out["contrast"] = float(np.std(gray))
    h, w = gray.shape[:2]
    if h < 6 or w < 6:
        return out
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw
    ink = inv > 0
    out["fg_ratio"] = float(np.mean(ink))
    if np.count_nonzero(ink):
        dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
        vals = dist[dist > 0]
        if vals.size > 0:
            out["stroke_p10"] = float(np.percentile(vals, 10))
            out["stroke_p50"] = float(np.percentile(vals, 50))
    edges = cv2.Canny(blur, 50, 140)
    out["edge_density"] = float(np.count_nonzero(edges)) / float(h * w)
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 6), 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, h // 6)))
    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k_h)
    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k_v)
    line_pix = float(np.count_nonzero(h_lines) + np.count_nonzero(v_lines))
    total_ink = float(np.count_nonzero(inv))
    out["line_strength_hv"] = line_pix / max(total_ink, 1.0)
    thin_score = 0.0
    thin_score += 0.35 * (1.0 if out["fg_ratio"] < 0.08 else max(0.0, min(1.0, (0.16 - out["fg_ratio"]) / 0.08)))
    thin_score += 0.30 * (1.0 if out["stroke_p50"] < 1.1 else max(0.0, min(1.0, (1.9 - out["stroke_p50"]) / 0.8)))
    thin_score += 0.20 * (1.0 if out["edge_density"] < 0.045 else max(0.0, min(1.0, (0.09 - out["edge_density"]) / 0.045)))
    thin_score += 0.15 * max(0.0, min(1.0, (out["line_strength_hv"] - 0.55) / 0.45))
    out["thin_score"] = float(max(0.0, min(1.0, thin_score)))
    return out


def _preprocess_variant(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "baseline":
        return gray
    if mode == "clahe":
        return cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)
    if mode == "thin_blackhat":
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        return cv2.addWeighted(gray, 1.0, bh, 0.9, 0)
    if mode == "linear_up2x":
        src = gray.astype(np.float32) / 255.0
        lin = np.power(src, 2.2)
        up = cv2.resize(lin, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        srgb = np.power(np.clip(up, 0.0, 1.0), 1.0 / 2.2)
        return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)
    return gray


def _extract_candidates(doc_payload: dict) -> List[RectCandidate]:
    out: List[RectCandidate] = []
    pages = (doc_payload or {}).get("pages", {})
    for _, page in pages.items():
        for r in page.get("results", []):
            if r.get("data_format") != "TABLE":
                continue
            t = r.get("value")
            if not isinstance(t, dict):
                continue
            table_id = str(t.get("table_id", "T?"))
            cells = t.get("cells") or []
            if not cells:
                continue
            row_centers = _group_rows(cells, tol=6.0)
            # pick anchor rows around material-like text
            material_rows: set = set()
            for c in cells:
                tx = str(c.get("text", "")).upper()
                y = float((c["bbox"][1] + c["bbox"][3]) / 2.0)
                r_idx = _nearest_idx(y, row_centers)
                if "MATERIAL" in tx or "SUPPLIER" in tx or tx.strip() in {"DET", "REOC"}:
                    material_rows.add(r_idx)
            for c in cells:
                y = float((c["bbox"][1] + c["bbox"][3]) / 2.0)
                x = float((c["bbox"][0] + c["bbox"][2]) / 2.0)
                row_idx = _nearest_idx(y, row_centers)
                col_idx = int(round(x))
                if material_rows and (row_idx in material_rows or row_idx in {m - 1 for m in material_rows} or row_idx in {m + 1 for m in material_rows}):
                    group = "material_adjacent"
                else:
                    group = "other"
                out.append(
                    RectCandidate(
                        doc_id=str(doc_payload.get("document_id", "")),
                        table_id=table_id,
                        cell_id=str(c.get("cell_id", "")),
                        poly_id=str(c.get("poly_id", "")),
                        text=str(c.get("text", "")),
                        bbox=[float(v) for v in c.get("bbox", [0, 0, 0, 0])],
                        row_idx=row_idx,
                        col_idx=col_idx,
                        row_group=group,
                    )
                )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--image-root", default=None, help="Optional root containing original page images")
    parser.add_argument("--report-dir", default="debug_outputs/rect_crop_metrics")
    args = parser.parse_args()

    report_dir = os.path.abspath(args.report_dir)
    os.makedirs(report_dir, exist_ok=True)
    batch_dir = os.path.join(os.path.abspath(args.output_root), args.batch_id)
    if not os.path.isdir(batch_dir):
        raise FileNotFoundError(f"Batch outputs not found: {batch_dir}")

    summary = {"batch_id": args.batch_id, "docs": []}

    for name in sorted(os.listdir(batch_dir)):
        if "_DOC_" not in name:
            continue
        if not any(name.endswith(s) for s in TARGET_DOC_SUFFIXES):
            continue
        doc_dir = os.path.join(batch_dir, name)
        digitized = _safe_read_json(os.path.join(doc_dir, "digitised_document.json"))
        if not digitized:
            continue
        image_path = _find_image_for_doc(digitized.get("document_id", name), args.image_root)
        image = cv2.imread(image_path) if image_path else None
        doc_out = {
            "doc_id": digitized.get("document_id", name),
            "doc_dir": doc_dir,
            "image_path": image_path,
            "candidates": [],
        }
        candidates = _extract_candidates(digitized)
        for c in candidates:
            rec = {
                "table_id": c.table_id,
                "cell_id": c.cell_id,
                "poly_id": c.poly_id,
                "text": c.text,
                "bbox": c.bbox,
                "row_idx": c.row_idx,
                "row_group": c.row_group,
            }
            if image is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in c.bbox]
                x1 = max(0, min(image.shape[1] - 1, x1))
                x2 = max(0, min(image.shape[1], x2))
                y1 = max(0, min(image.shape[0] - 1, y1))
                y2 = max(0, min(image.shape[0], y2))
                if x2 > x1 and y2 > y1:
                    crop = image[y1:y2, x1:x2]
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    rec["metrics"] = _compute_thin_metrics(gray)
                    variants = {}
                    for mode in ("baseline", "clahe", "thin_blackhat", "linear_up2x"):
                        vg = _preprocess_variant(gray, mode)
                        variants[mode] = _compute_thin_metrics(vg)
                        crop_dir = os.path.join(report_dir, c.doc_id, mode)
                        os.makedirs(crop_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(crop_dir, f"{c.poly_id}_{c.cell_id}.png"), vg)
                    rec["variant_metrics"] = variants
            doc_out["candidates"].append(rec)
        summary["docs"].append(doc_out)

    out_json = os.path.join(report_dir, f"{args.batch_id}_thin_rect_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {out_json}")
    print("Note: if image_path is null, provide --image-root to enable crop metrics/images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

