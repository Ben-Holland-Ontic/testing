"""
Export cluster OCR crops: detector crop, padded crop (Paddle input before preprocess),
and image after _preprocess_for_ocr. Does not run inference.

Usage (from bk-main):
  py tools/dump_cluster_ocr_crops.py "C:\\path\\to\\page.jpg" outputs\\crop_debug\\run1
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("input", "workers", "cleaners", "bootstrap"):
    sys.path.insert(0, os.path.join(_ROOT, p))
sys.path.insert(0, _ROOT)

import cv2
import numpy as np

import cluster_ocr as co
from rectangle_detection import detect_rectangles


def main():
    ap = argparse.ArgumentParser(description="Dump cluster OCR crops for inspection.")
    ap.add_argument("image_path", help="Path to BGR image (jpg/png)")
    ap.add_argument("out_dir", help="Output directory (created if missing)")
    args = ap.parse_args()

    img_path = os.path.abspath(args.image_path)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print("FAILED to load:", img_path, file=sys.stderr)
        sys.exit(1)
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    rect_data, _ = detect_rectangles(
        image=img_bgr,
        frame_rect=None,
        page_id="dump_crops",
        adaptive_thresh=True,
        debug=False,
        basename=None,
        gray=gray,
    )
    rectangles = rect_data.get("rectangles", []) if rect_data else []
    meta_path = os.path.join(out_dir, "_manifest.txt")
    with open(meta_path, "w", encoding="utf-8") as mf:
        mf.write(f"image={img_path}\n{w}x{h}\nrectangles={len(rectangles)}\n")
        mf.write(f"CLUSTER_OCR_EDGE_SUPPRESS={os.environ.get('CLUSTER_OCR_EDGE_SUPPRESS', '')!r}\n")

    for i, r in enumerate(rectangles):
        c = r.get("crop")
        if c is None or c.size == 0:
            continue
        cv2.imwrite(os.path.join(out_dir, f"{i:03d}_01_detector_crop.png"), c)

    counter = [0]

    def _wrap(crop_bgr, label, thin_kernel=None, thin_stroke=False, stroke_median=None, stroke_mean=None):
        idx = counter[0]
        base = os.path.join(out_dir, f"{idx:03d}_02")
        if crop_bgr is not None and crop_bgr.size:
            cv2.imwrite(f"{base}_padded_before_preprocess.png", crop_bgr)
        out = _orig(
            crop_bgr,
            label,
            thin_kernel=thin_kernel,
            thin_stroke=thin_stroke,
            stroke_median=stroke_median,
            stroke_mean=stroke_mean,
        )
        if out is not None and out.size:
            cv2.imwrite(f"{base}_after_preprocess.png", out)
        counter[0] += 1
        return out

    _orig = co._preprocess_for_ocr
    co._preprocess_for_ocr = _wrap
    try:
        co.process_rectangles(rectangles, page_id="dump_crops", page_h=h, page_w=w)
    finally:
        co._preprocess_for_ocr = _orig

    with open(meta_path, "a", encoding="utf-8") as mf:
        mf.write(f"preprocess_calls={counter[0]}\n")

    print("Wrote crops under", out_dir)
    print("Manifest:", meta_path)


if __name__ == "__main__":
    main()
