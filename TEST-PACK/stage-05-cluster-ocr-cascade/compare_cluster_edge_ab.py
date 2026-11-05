"""
Compare cluster OCR output with CLUSTER_OCR_EDGE_SUPPRESS off vs on (same rectangles, same run).

Usage:
  py tools/compare_cluster_edge_ab.py "C:\\path\\image.jpg"
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


def _joined_cluster_text(result: dict) -> str:
    lines = result.get("Cluster_Line") or []
    parts = []
    for ln in lines:
        t = (ln.get("text") or ln.get("text_raw") or "").strip()
        if t:
            parts.append(t)
    return "\n".join(sorted(parts))


def run_once(img_bgr, edge_suppress):
    if edge_suppress is None:
        os.environ.pop("CLUSTER_OCR_EDGE_SUPPRESS", None)
    else:
        os.environ["CLUSTER_OCR_EDGE_SUPPRESS"] = edge_suppress

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rect_data, _ = detect_rectangles(
        image=img_bgr,
        frame_rect=None,
        page_id="edge_ab",
        adaptive_thresh=True,
        debug=False,
        basename=None,
        gray=gray,
    )
    rectangles = rect_data.get("rectangles", []) if rect_data else []
    out = co.process_rectangles(rectangles, page_id="edge_ab", page_h=h, page_w=w)
    n_lines = len(out.get("Cluster_Line") or [])
    n_clusters = len(out.get("clusters") or [])
    jt = _joined_cluster_text(out)
    return {
        "n_rects": len(rectangles),
        "cluster_line_count": n_lines,
        "cluster_groups": n_clusters,
        "text_sorted_joined_len": len(jt),
        "text_sorted_joined": jt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path")
    args = ap.parse_args()
    img_path = os.path.abspath(args.image_path)
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print("FAILED:", img_path, file=sys.stderr)
        sys.exit(1)
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    print("Image:", img_path, img_bgr.shape[1], "x", img_bgr.shape[0])
    a = run_once(img_bgr, None)
    b = run_once(img_bgr, "1")
    print("\n--- EDGE SUPPRESS OFF (default) ---")
    print("rects", a["n_rects"], "cluster_lines", a["cluster_line_count"], "groups", a["cluster_groups"], "joined_len", a["text_sorted_joined_len"])
    print("\n--- EDGE SUPPRESS ON (CLUSTER_OCR_EDGE_SUPPRESS=1) ---")
    print("rects", b["n_rects"], "cluster_lines", b["cluster_line_count"], "groups", b["cluster_groups"], "joined_len", b["text_sorted_joined_len"])

    same_text = a["text_sorted_joined"] == b["text_sorted_joined"]
    print("\nSorted-joined cluster text identical:", same_text)
    if not same_text:
        import difflib
        d = list(
            difflib.unified_diff(
                a["text_sorted_joined"].splitlines(),
                b["text_sorted_joined"].splitlines(),
                lineterm="",
                n=0,
            )
        )
        print("Diff (first 40 lines):")
        for line in d[:40]:
            print(line)


if __name__ == "__main__":
    main()
