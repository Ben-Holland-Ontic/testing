"""A/B the crops the router sends to handwriting: 91-class model vs 18,710-class model.

DIAGNOSTIC, read-only. The census showed 5.66% of crops get labelled "handwriting" by a
geometric heuristic (stroke variance / baseline irregularity) that fires on degraded PRINT
as readily as on handwriting. Fixing PADDLE_HANDWRITING_REC_DIR activates that path, so the
question is simply: for those crops, which model reads them better?

There is no fallback in the pipeline if the handwriting model returns confident nonsense,
so this is the check that decides whether the fix ships.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY handwriting_ab_probe.py --pages 25
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
import zipfile

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=25)
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args()

    import cv2
    import cluster_ocr
    from rectangle_detection import detect_rectangles

    pdfs = extract_pdfs()
    pages = render_pages(pdfs, args.pages)
    print(f"rendered {len(pages)} pages")

    # Capture the crops the router labels "handwriting", straight off the dispatch.
    hw_crops = []
    real = cluster_ocr._run_paddleocr_on_crops

    def spy(crops_bgr, rect_infos, labels=None):
        if labels:
            for crop, lab in zip(crops_bgr, labels):
                if lab == "handwriting" and crop is not None and crop.size:
                    hw_crops.append(crop)
        return real(crops_bgr, rect_infos, labels)

    cluster_ocr._run_paddleocr_on_crops = spy
    try:
        for src, idx, img in pages:
            try:
                rect_data, _ = detect_rectangles(
                    image=img, frame_rect=None, page_id=f"{src}#{idx}",
                    adaptive_thresh=True, debug=False, basename=None,
                    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                rects = (rect_data or {}).get("rectangles", []) or []
                if rects:
                    cluster_ocr.process_rectangles(
                        rects, page_id=f"{src}#{idx}",
                        page_h=img.shape[0], page_w=img.shape[1])
            except Exception:
                continue
    finally:
        cluster_ocr._run_paddleocr_on_crops = real

    print(f"captured {len(hw_crops)} crops labelled 'handwriting'\n")
    if not hw_crops:
        print("none captured"); return 1

    general = cluster_ocr._get_paddleocr()
    hw = cluster_ocr._get_paddleocr_handwriting()
    if hw is None:
        print("handwriting engine not available"); return 1

    def read(engine, crop):
        try:
            out = engine.ocr(crop)
            items = out[0] if out else []
            if not items:
                return "", 0.0
            # Longest line wins; these crops are single cells.
            best = max(items, key=lambda it: len(it[1][0] or ""))
            return (best[1][0] or "").strip(), float(best[1][1])
        except Exception as exc:
            return f"<{type(exc).__name__}>", 0.0

    rows = []
    for c in hw_crops:
        gt, gs = read(general, c)
        ht, hs = read(hw, c)
        rows.append((gt, gs, ht, hs))

    same = sum(1 for g, _, h, _ in rows if g == h)
    g_empty = sum(1 for g, _, _, _ in rows if not g)
    h_empty = sum(1 for _, _, h, _ in rows if not h)
    g_chars = sum(len(g) for g, _, _, _ in rows)
    h_chars = sum(len(h) for _, _, h, _ in rows)
    g_conf = sum(s for _, s, _, _ in rows) / len(rows)
    h_conf = sum(s for _, _, _, s in rows) / len(rows)

    print("=" * 78)
    print(f"A/B over {len(rows)} handwriting-labelled crops")
    print("=" * 78)
    print(f"  identical output      : {same}/{len(rows)} ({same/len(rows)*100:.1f}%)")
    print(f"  empty result          : general {g_empty}   handwriting {h_empty}")
    print(f"  total chars returned  : general {g_chars}   handwriting {h_chars}")
    print(f"  mean confidence       : general {g_conf:.3f}   handwriting {h_conf:.3f}")
    # Per-crop, not just the mean: a confidence-based router is only viable if the
    # better engine is also the more confident one CROP BY CROP.
    g_wins = sum(1 for _, gs, _, hs in rows if gs > hs)
    h_wins = sum(1 for _, gs, _, hs in rows if hs > gs)
    ties = len(rows) - g_wins - h_wins
    print()
    print("  PER-CROP CONFIDENCE (which engine is more confident?)")
    print(f"    general more confident    : {g_wins}/{len(rows)} ({g_wins/len(rows)*100:.1f}%)")
    print(f"    handwriting more confident: {h_wins}/{len(rows)} ({h_wins/len(rows)*100:.1f}%)")
    print(f"    tie                       : {ties}")
    # If we routed by max-confidence, how often would we pick the general model?
    margin = [gs - hs for _, gs, _, hs in rows]
    margin.sort()
    print(f"    (general - handwriting) confidence margin: "
          f"min {margin[0]:+.3f}  median {margin[len(margin)//2]:+.3f}  max {margin[-1]:+.3f}")
    hw_conf_when_wrong = [hs for g, _, h, hs in rows if g and h and g != h]
    if hw_conf_when_wrong:
        print(f"    handwriting confidence when it disagrees: "
              f"mean {sum(hw_conf_when_wrong)/len(hw_conf_when_wrong):.3f}, "
              f"max {max(hw_conf_when_wrong):.3f}")
    print()
    print(f"  {'GENERAL (18,710-class)':<40} {'HANDWRITING (91-class)':<36}")
    print("  " + "-" * 76)
    for gt, gs, ht, hs in rows[:args.show]:
        mark = "  " if gt == ht else "<>"
        print(f"{mark} {gt[:36]!r:<40} {ht[:32]!r:<36}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
