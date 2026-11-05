import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from workers.cluster_ocr import _run_paddleocr_on_crops


TARGET_DOC_SUFFIXES = [
    "Screenshot_18-3-2026_101153__jpeg",
    "Screenshot_23-3-2026_82649__jpeg",
    "1759857039776_jpg",
]


def _canonical_name(s: str) -> str:
    base, ext = os.path.splitext(s.lower())
    for ch in (" ", "-", "."):
        base = base.replace(ch, "_")
    while "__" in base:
        base = base.replace("__", "_")
    return base.strip("_") + ext


def _doc_to_filename(doc_id: str) -> str:
    suffix = doc_id.split("_DOC_", 1)[-1]
    if suffix.endswith("__jpeg"):
        return suffix[:-6] + ".jpeg"
    if suffix.endswith("_jpg"):
        return suffix[:-4] + ".jpg"
    if suffix.endswith("_png"):
        return suffix[:-4] + ".png"
    return suffix


def _find_image(doc_id: str, image_root: str) -> str:
    wanted = _doc_to_filename(doc_id).lower()
    wanted_c = _canonical_name(wanted)
    for root, _, files in os.walk(image_root):
        for fn in files:
            fn_l = fn.lower()
            if fn_l == wanted or _canonical_name(fn_l) == wanted_c:
                return os.path.join(root, fn)
    return ""


def _row_centers(cells: List[dict], tol: float = 6.0) -> List[float]:
    ys = sorted([(c["bbox"][1] + c["bbox"][3]) / 2.0 for c in cells])
    out: List[float] = []
    for y in ys:
        if not out or abs(y - out[-1]) > tol:
            out.append(y)
        else:
            out[-1] = (out[-1] + y) / 2.0
    return out


def _nearest_idx(v: float, arr: List[float]) -> int:
    if not arr:
        return 0
    return int(np.argmin([abs(v - x) for x in arr]))


def _prep_linear_up(gray: np.ndarray, scale: float = 2.0) -> np.ndarray:
    h, w = gray.shape[:2]
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    src = gray.astype(np.float32) / 255.0
    lin = np.power(src, 2.2)
    up = cv2.resize(lin, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    srgb = np.power(np.clip(up, 0.0, 1.0), 1.0 / 2.2)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


def _prep_thin_blackhat(gray: np.ndarray) -> np.ndarray:
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return cv2.addWeighted(gray, 1.0, bh, 1.0, 0)


def _prep_clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)


def _prep_ungrid(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    work = gray.copy()
    _, bw = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 5), 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 5)))
    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k_h)
    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k_v)
    lines = cv2.bitwise_or(h_lines, v_lines)
    cleaned = cv2.subtract(inv, lines)
    return 255 - cleaned


def _variants(gray: np.ndarray) -> Dict[str, np.ndarray]:
    up = _prep_linear_up(gray, 2.2)
    return {
        "raw": gray,
        "clahe": _prep_clahe(gray),
        "blackhat": _prep_thin_blackhat(gray),
        "linear_up2x": up,
        "linear_up2x_blackhat": _prep_thin_blackhat(up),
        "ungrid_blackhat": _prep_thin_blackhat(_prep_ungrid(up)),
    }


def _ocr_variant_images(images: Dict[str, np.ndarray], poly_id: str) -> Dict[str, dict]:
    names = list(images.keys())
    crops = [cv2.cvtColor(images[n], cv2.COLOR_GRAY2BGR) for n in names]
    rect_infos = [(0, 0, i, poly_id) for i in range(len(crops))]
    out = _run_paddleocr_on_crops(crops, rect_infos, labels=["small"] * len(crops))
    scored: Dict[str, dict] = {}
    for i, name in enumerate(names):
        lines = out[i] if i < len(out) else []
        texts = [str(x.get("text", "")).strip() for x in lines if str(x.get("text", "")).strip()]
        confs = [float(x.get("confidence", 0.0) or 0.0) for x in lines]
        joined = " | ".join(texts)
        scored[name] = {
            "n_lines": len(lines),
            "text": joined,
            "char_count": len("".join(texts)),
            "avg_conf": float(np.mean(confs)) if confs else 0.0,
            "max_conf": max(confs) if confs else 0.0,
        }
    return scored


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="outputs")
    ap.add_argument("--batch-id", default="BATCH_08962480")
    ap.add_argument("--image-root", default="test_data")
    ap.add_argument("--save-dir", default="debug_outputs/right_col_variant_crops")
    args = ap.parse_args()

    batch_dir = os.path.join(args.output_root, args.batch_id)
    os.makedirs(args.save_dir, exist_ok=True)
    rows_out = []

    for doc_folder in sorted(os.listdir(batch_dir)):
        if "_DOC_" not in doc_folder or not any(doc_folder.endswith(s) for s in TARGET_DOC_SUFFIXES):
            continue
        doc_dir = os.path.join(batch_dir, doc_folder)
        dig_path = os.path.join(doc_dir, "digitised_document.json")
        if not os.path.isfile(dig_path):
            continue
        payload = json.load(open(dig_path, "r", encoding="utf-8"))
        doc_id = payload.get("document_id", doc_folder)
        img_path = _find_image(doc_id, args.image_root)
        if not img_path:
            continue
        image = cv2.imread(img_path)
        if image is None:
            continue

        for _, page in payload.get("pages", {}).items():
            for result in page.get("results", []):
                if result.get("data_format") != "TABLE":
                    continue
                t = result.get("value")
                if not isinstance(t, dict):
                    continue
                cells = t.get("cells") or []
                if not cells:
                    continue
                # right-most column by x-mid
                x_mids = sorted(list({(c["bbox"][0] + c["bbox"][2]) / 2.0 for c in cells}))
                right_x = x_mids[-1]
                cols = [x for x in x_mids if abs(x - right_x) <= 2.0]
                col_x = cols[0] if cols else right_x

                rcent = _row_centers(cells)
                material_rows = set()
                for c in cells:
                    txt = str(c.get("text", "")).upper()
                    y = (c["bbox"][1] + c["bbox"][3]) / 2.0
                    if "MATERIAL" in txt:
                        material_rows.add(_nearest_idx(y, rcent))

                for c in cells:
                    x = (c["bbox"][0] + c["bbox"][2]) / 2.0
                    if abs(x - col_x) > 2.0:
                        continue
                    txt = str(c.get("text", "")).strip()
                    y = (c["bbox"][1] + c["bbox"][3]) / 2.0
                    row_idx = _nearest_idx(y, rcent)
                    if txt:
                        continue
                    if material_rows and row_idx not in material_rows and row_idx not in {r - 1 for r in material_rows} and row_idx not in {r + 1 for r in material_rows}:
                        continue
                    x1, y1, x2, y2 = [int(round(v)) for v in c["bbox"]]
                    x1 = max(0, min(image.shape[1] - 1, x1))
                    x2 = max(0, min(image.shape[1], x2))
                    y1 = max(0, min(image.shape[0] - 1, y1))
                    y2 = max(0, min(image.shape[0], y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = image[y1:y2, x1:x2]
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    var_imgs = _variants(gray)
                    ocr = _ocr_variant_images(var_imgs, str(c.get("poly_id", "")))
                    # choose best by char_count then avg_conf
                    best_name = max(ocr.keys(), key=lambda k: (ocr[k]["char_count"], ocr[k]["avg_conf"]))
                    out_row = {
                        "doc_id": doc_id,
                        "table_id": t.get("table_id", ""),
                        "cell_id": c.get("cell_id", ""),
                        "poly_id": c.get("poly_id", ""),
                        "row_idx": row_idx,
                        "bbox": c.get("bbox", []),
                        "best_variant": best_name,
                        "best_text": ocr[best_name]["text"],
                        "best_char_count": ocr[best_name]["char_count"],
                        "variants": ocr,
                    }
                    rows_out.append(out_row)
                    save_base = os.path.join(args.save_dir, doc_id, str(c.get("poly_id", "")))
                    os.makedirs(save_base, exist_ok=True)
                    for name, g in var_imgs.items():
                        cv2.imwrite(os.path.join(save_base, f"{name}.png"), g)

    rows_out.sort(key=lambda r: (r["best_char_count"], r["variants"][r["best_variant"]]["avg_conf"]), reverse=True)
    out_path = os.path.join(args.save_dir, f"{args.batch_id}_right_col_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows_out}, f, indent=2)
    print(f"Wrote: {out_path}")
    print(f"Rows tested: {len(rows_out)}")
    for r in rows_out:
        print(
            f"{r['doc_id']}\t{r['poly_id']}\t{r['cell_id']}\tbest={r['best_variant']}\t"
            f"chars={r['best_char_count']}\ttext={r['best_text']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

