"""Dump every line's candidate set ONCE, so all later tuning costs nothing.

DIAGNOSTIC. Threshold fitting and policy experiments do not need a GPU -- they need the
CANDIDATES. Re-running four ONNX models (plus their TensorRT engines, ~1.3GB on top of
the pipeline's own ~8.5GB) for every experiment is what made this machine crawl.

So: one pass, nothing else running, results to JSON. After this, every policy question is
answered offline on CPU with no models loaded, in under a second.

Run it ALONE -- no other probe, no extraction, nothing else on the GPU.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY dump_candidates.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

BK = r"C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main"
for p in (BK, os.path.join(BK, "workers"), os.path.join(BK, "input"),
          os.path.join(BK, "cleaners")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("OCULI_SUPPORTING_MODEL_DIR", r"C:\OCULI.pd1\TESTING\onnx-models")
os.environ.setdefault("PADDLEOCR_USE_GPU", "1")

OUT = r"C:\OCULI.pd1\TESTING\RESULTS\hard_crops"
CROP_DIR = os.path.join(OUT, "crops")
MANIFEST = os.path.join(OUT, "manifest.json")
DUMP = os.path.join(OUT, "candidates.json")
CHUNK = 16          # small: bounds peak memory, costs a little throughput


def geometry_features(crop):
    """Cheap, model-free description of a crop: is this even text-shaped?

    Computed here so the offline tuner can experiment with geometry rejection without
    touching an image again.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape[:2]
    _t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = binary > 0
    if not ink.any():
        return {"h": h, "w": w, "ink": 0.0, "components": 0, "long_run_frac": 0.0,
                "row_fill_max": 0.0, "col_fill_max": 0.0}

    # Connected components: text is many small blobs; a border is one long thin one.
    n_lab, _lab, stats, _cent = cv2.connectedComponentsWithStats(
        binary.astype("uint8"), connectivity=8)

    # Straight runs: a ruling line fills a whole row or column.
    row_fill = ink.mean(axis=1)
    col_fill = ink.mean(axis=0)

    # Fraction of ink sitting in components that span most of a dimension.
    long_ink = 0
    for i in range(1, n_lab):
        x, y, cw, ch, area = stats[i]
        if cw >= 0.8 * w or ch >= 0.8 * h:
            long_ink += area
    total_ink = int(ink.sum())

    return {
        "h": int(h), "w": int(w),
        "ink": round(float(ink.mean()), 4),
        "components": int(n_lab - 1),
        "long_run_frac": round(long_ink / max(1, total_ink), 4),
        "row_fill_max": round(float(row_fill.max()), 4),
        "col_fill_max": round(float(col_fill.max()), 4),
    }


def main() -> int:
    import cv2
    import cluster_ocr
    import ocr_backend
    import ocr_supporting as S

    ocr_backend.register_cuda_dll_dirs()
    engine = cluster_ocr._get_paddleocr()
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    crops = [m for m in manifest if m["primary_text"]]
    print(f"{len(crops)} crops the primary read; extracting line regions...")

    rows = []
    images = []
    for m in crops:
        img = cv2.imread(os.path.join(CROP_DIR, m["file"]))
        if img is None:
            continue
        res = cluster_ocr._normalize_paddle_list_result(engine.ocr(img))
        for ln in cluster_ocr._parse_paddle_result_to_lines(res, 0, 0, 0, "DUMP"):
            text = (ln.get("text") or "").strip()
            bbox = ln.get("bbox") or []
            if not text or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
            x1 = max(0, x1 - 2); y1 = max(0, y1 - 2)
            x2 = min(img.shape[1], x2 + 2); y2 = min(img.shape[0], y2 + 2)
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            region = img[y1:y2, x1:x2]
            rows.append({
                "file": m["file"], "page": m["page"], "label": m.get("label"),
                "primary": text,
                "primary_conf": round(float(ln.get("confidence", 0.0) or 0.0), 4),
                "crop_conf": m["primary_conf"],
                "geometry": geometry_features(region),
            })
            images.append(region)
        del img

    print(f"{len(rows)} line regions; running the supporting models in chunks of {CHUNK}")
    for name in S.MODEL_NAMES:
        rec = S.get_recogniser(name)
        if rec is None:
            for r in rows:
                r.setdefault("models", {})[name] = ["", 0.0]
            continue
        done = 0
        for off in range(0, len(images), CHUNK):
            batch = images[off:off + CHUNK]
            for r, (text, conf) in zip(rows[off:off + CHUNK], rec.recognise(batch)):
                r.setdefault("models", {})[name] = [text, round(float(conf), 4)]
            done += len(batch)
        print(f"  {name}: {done} regions")
        # Release this model before loading the next: one at a time keeps peak memory to
        # roughly a single model plus its engine, instead of all four at once.
        S._TLS.cache.pop(name, None)
        del rec
        gc.collect()

    with open(DUMP, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(rows)} candidate sets -> {DUMP}")
    print("Every policy experiment from here is offline, CPU-only, no models.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
