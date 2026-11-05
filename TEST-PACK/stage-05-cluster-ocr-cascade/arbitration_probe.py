"""Verify handwriting arbitration on real pages, through the shipped code path.

DIAGNOSTIC, read-only. Measures three things that matter:

  1. SCOPE — arbitration must touch ONLY crops the router labels "handwriting". If it
     fires on anything else it is doing work (and taking risk) on 94% of crops for nothing.
  2. EFFECT — what the text actually becomes, arbitrated vs not.
  3. COST — the extra time spent on the second recognition pass.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY arbitration_probe.py --pages 25
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

BK = r"C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main"
for p in (BK, os.path.join(BK, "workers"), os.path.join(BK, "input"),
          os.path.join(BK, "cleaners")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PADDLEOCR_USE_GPU", "1")
os.environ.setdefault("PADDLE_ANGLE_CLS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handwriting_label_census import extract_pdfs, render_pages  # noqa: E402


def collect(pages, arbitrate: bool):
    """Run the real dispatch over every page, returning {crop_key: (text, conf, label)}."""
    import cv2
    import importlib
    os.environ["OCULI_HW_ARBITRATE"] = "1" if arbitrate else "0"
    import cluster_ocr
    importlib.reload(cluster_ocr)
    from rectangle_detection import detect_rectangles

    out = {}
    labels_seen = []
    real = cluster_ocr._run_paddleocr_on_crops_impl
    call = {"n": 0}

    def wrapper(crops_bgr, rect_infos, labels=None):
        res = real(crops_bgr, rect_infos, labels)
        if labels:
            base = call["n"]
            for k, (lab, lines) in enumerate(zip(labels, res)):
                labels_seen.append(lab)
                txt = " | ".join((l.get("text", "") or "") for l in (lines or []))
                conf = (sum(float(l.get("confidence", 0) or 0) for l in (lines or []))
                        / max(1, len(lines or [])))
                out[base + k] = (txt, conf, lab)
            call["n"] = base + len(labels)
        return res

    cluster_ocr._run_paddleocr_on_crops_impl = wrapper
    t0 = time.perf_counter()
    try:
        for src, idx, img in pages:
            try:
                rd, _ = detect_rectangles(
                    image=img, frame_rect=None, page_id=f"{src}#{idx}",
                    adaptive_thresh=True, debug=False, basename=None,
                    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                rects = (rd or {}).get("rectangles", []) or []
                if rects:
                    cluster_ocr.process_rectangles(
                        rects, page_id=f"{src}#{idx}",
                        page_h=img.shape[0], page_w=img.shape[1])
            except Exception:
                continue
    finally:
        cluster_ocr._run_paddleocr_on_crops_impl = real
    return out, time.perf_counter() - t0, labels_seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=25)
    ap.add_argument("--show", type=int, default=18)
    args = ap.parse_args()

    pages = render_pages(extract_pdfs(), args.pages)
    print(f"rendered {len(pages)} pages\n")

    print("run 1: arbitration OFF (handwriting model unchecked)")
    off, t_off, labels = collect(pages, arbitrate=False)
    print(f"  {len(off)} crops in {t_off:.1f}s")

    print("run 2: arbitration ON")
    on, t_on, _ = collect(pages, arbitrate=True)
    print(f"  {len(on)} crops in {t_on:.1f}s\n")

    keys = sorted(set(off) & set(on))
    hw_keys = [k for k in keys if off[k][2] == "handwriting"]
    other_keys = [k for k in keys if off[k][2] != "handwriting"]

    changed_hw = [k for k in hw_keys if off[k][0] != on[k][0]]
    changed_other = [k for k in other_keys if off[k][0] != on[k][0]]

    print("=" * 78)
    print("1. SCOPE — arbitration must only touch handwriting-labelled crops")
    print("=" * 78)
    print(f"  crops compared            : {len(keys)}")
    print(f"  handwriting-labelled      : {len(hw_keys)} ({len(hw_keys)/max(1,len(keys))*100:.2f}%)")
    print(f"  CHANGED, handwriting      : {len(changed_hw)}")
    print(f"  CHANGED, everything else  : {len(changed_other)}  <- must be 0")
    print(f"  -> {'CORRECTLY SCOPED' if not changed_other else 'LEAKING OUTSIDE SCOPE'}")

    print()
    print("=" * 78)
    print("2. EFFECT on handwriting-labelled crops")
    print("=" * 78)
    c_off = sum(off[k][1] for k in hw_keys) / max(1, len(hw_keys))
    c_on = sum(on[k][1] for k in hw_keys) / max(1, len(hw_keys))
    print(f"  mean confidence  : {c_off:.3f} -> {c_on:.3f}")
    print(f"  swapped to general: {len(changed_hw)}/{len(hw_keys)}")
    print()
    for k in changed_hw[:args.show]:
        print(f"    hw={off[k][0][:34]!r:<38} -> {on[k][0][:34]!r}")

    print()
    print("=" * 78)
    print("3. COST")
    print("=" * 78)
    print(f"  total time: {t_off:.1f}s -> {t_on:.1f}s  "
          f"({(t_on/max(t_off,1e-9)-1)*100:+.1f}%)")
    return 0 if not changed_other else 1


if __name__ == "__main__":
    raise SystemExit(main())
