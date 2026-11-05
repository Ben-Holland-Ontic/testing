"""Cut real text crops out of a real drawing, for the supporting-model unit tests.

DIAGNOSTIC / fixture builder. Runs the PRIMARY recogniser over a page, keeps the lines
it read with high confidence, and writes each one out as a PNG plus a manifest of what
it read. Real crops matter here: a synthetic render of a font would not exercise the
scan noise, thin strokes and tight cropping the adapters actually meet.

The manifest text is the PRIMARY's reading, which is a strong hint but not ground truth.
`expected` is filled in by hand after looking at the crops; anything still marked null
is carried as "no ground truth" and only used for cross-model agreement, never as a
correctness assertion.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY build_ocr_fixtures.py
"""
from __future__ import annotations

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

os.environ.setdefault("PADDLEOCR_USE_GPU", "1")

OUT_DIR = r"C:\OCULI.pd1\TESTING\RESOURCES\fixtures"
CROP_DIR = os.path.join(OUT_DIR, "crops")
PAGES = [
    os.path.join(BK, "test_data", "01_original.png"),
    r"C:\Users\benja\OneDrive\Desktop\data\Ball Valve dwg 1.PNG",
]
MIN_CONF = 0.95
MIN_CHARS = 3
MAX_PER_PAGE = 10
PAD = 2


def main() -> int:
    import cv2
    import numpy as np
    import cluster_ocr

    os.makedirs(CROP_DIR, exist_ok=True)
    engine = cluster_ocr._get_paddleocr()

    manifest = []
    for page_path in PAGES:
        if not os.path.isfile(page_path):
            print(f"  skip (missing): {page_path}")
            continue
        img = cv2.imread(page_path)
        if img is None:
            print(f"  skip (unreadable): {page_path}")
            continue
        out = engine.ocr(img)
        res = cluster_ocr._normalize_paddle_list_result(out)
        lines = cluster_ocr._parse_paddle_result_to_lines(res, 0, 0, 0, "FIXTURE")
        print(f"{os.path.basename(page_path)}: {len(lines)} line(s)")

        kept = 0
        for ln in sorted(lines, key=lambda l: -float(l.get("confidence", 0.0) or 0.0)):
            if kept >= MAX_PER_PAGE:
                break
            text = (ln.get("text") or "").strip()
            conf = float(ln.get("confidence", 0.0) or 0.0)
            if conf < MIN_CONF or len(text) < MIN_CHARS:
                continue
            bbox = ln.get("bbox") or []
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
            x1 = max(0, x1 - PAD); y1 = max(0, y1 - PAD)
            x2 = min(img.shape[1], x2 + PAD); y2 = min(img.shape[0], y2 + PAD)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crop = img[y1:y2, x1:x2]
            name = f"{os.path.splitext(os.path.basename(page_path))[0]}_{kept:02d}.png"
            name = name.replace(" ", "_")
            cv2.imwrite(os.path.join(CROP_DIR, name), crop)
            manifest.append({
                "file": name,
                "page": os.path.basename(page_path),
                "primary_text": text,
                "primary_conf": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
                "size_hw": [int(crop.shape[0]), int(crop.shape[1])],
                "expected": None,   # filled in by hand after visual check
            })
            kept += 1
        print(f"  kept {kept}")

    manifest.sort(key=lambda m: m["file"])
    path = os.path.join(OUT_DIR, "crops.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(manifest)} crop(s) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
