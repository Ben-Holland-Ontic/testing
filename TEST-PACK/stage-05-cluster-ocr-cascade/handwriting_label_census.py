"""How often does the pre-classifier route a crop to the handwriting recogniser?

DIAGNOSTIC, read-only. Answers the question that decides whether fixing
PADDLE_HANDWRITING_REC_DIR is safe to ship: the fix activates a previously dead path, and
the router is a geometric HEURISTIC (stroke-width variance / baseline irregularity), not a
model — so degraded PRINT can be labelled "handwriting" as readily as real handwriting.

Method: render pages from the corpus PDFs, run cluster_ocr.process_rectangles on each (the
real pipeline entry point, which is what computes the labels), and tally them. The tally is
taken by monkeypatching the label sink, so no shipped code is modified.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY handwriting_label_census.py --pages 40
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time
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

CORPUS_ZIP = os.path.join(BK, "test_data", "14-07-26 (805 pgs).zip")
WORK = r"C:\OCULI.pd1\TESTING\RESULTS\corpus"


def extract_pdfs() -> list:
    os.makedirs(WORK, exist_ok=True)
    out = []
    with zipfile.ZipFile(CORPUS_ZIP) as z:
        for name in z.namelist():
            if not name.lower().endswith(".pdf"):
                continue
            dest = os.path.join(WORK, os.path.basename(name))
            if not os.path.isfile(dest):
                with z.open(name) as src, open(dest, "wb") as fh:
                    fh.write(src.read())
            out.append(dest)
    return sorted(out)


def render_pages(pdfs, limit):
    """Render at the pipeline's own DPI so crop geometry — and therefore the stroke
    statistics the label heuristic keys on — matches a real run."""
    import pypdfium2 as pdfium
    pages = []
    for path in pdfs:
        if len(pages) >= limit:
            break
        try:
            doc = pdfium.PdfDocument(path)
        except Exception as exc:
            print(f"  skip {os.path.basename(path)}: {exc}")
            continue
        for i in range(len(doc)):
            if len(pages) >= limit:
                break
            try:
                bmp = doc[i].render(scale=200 / 72.0)      # 200 dpi
                pages.append((os.path.basename(path), i, bmp.to_numpy()[:, :, :3][:, :, ::-1]))
            except Exception:
                continue
        doc.close()
    return pages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=40)
    args = ap.parse_args()

    import numpy as np
    import cv2
    import cluster_ocr
    from rectangle_detection import detect_rectangles

    print(f"handwriting model resolved: {cluster_ocr.PADDLE_HANDWRITING_REC_DIR}")
    print(f"  dir ok: {cluster_ocr._handwriting_rec_dir_ok()}")

    pdfs = extract_pdfs()
    print(f"corpus: {len(pdfs)} PDFs -> rendering up to {args.pages} pages at 200dpi")
    pages = render_pages(pdfs, args.pages)
    print(f"rendered {len(pages)} pages\n")
    if not pages:
        print("no pages rendered"); return 2

    # Tally labels by intercepting the OCR dispatch, which receives the per-crop labels.
    counts = collections.Counter()
    per_page = []
    real = cluster_ocr._run_paddleocr_on_crops

    def spy(crops_bgr, rect_infos, labels=None):
        if labels:
            counts.update(labels)
        return real(crops_bgr, rect_infos, labels)

    cluster_ocr._run_paddleocr_on_crops = spy
    try:
        for n, (src, idx, img) in enumerate(pages, 1):
            before = sum(counts.values())
            t0 = time.perf_counter()
            try:
                # Rectangles come from the detection worker, exactly as pipeline.py does
                # (pipeline.py:1176). process_rectangles([]) classifies nothing, which is
                # why the first attempt at this census reported 0 crops.
                rect_data, _vis = detect_rectangles(
                    image=img, frame_rect=None, page_id=f"{src}#{idx}",
                    adaptive_thresh=True, debug=False, basename=None,
                    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                rects = (rect_data or {}).get("rectangles", []) or []
                if not rects:
                    continue
                cluster_ocr.process_rectangles(
                    rects, page_id=f"{src}#{idx}", page_h=img.shape[0], page_w=img.shape[1])
            except Exception as exc:
                print(f"  page {n}: {type(exc).__name__}: {str(exc)[:90]}")
                continue
            per_page.append(sum(counts.values()) - before)
            if n % 10 == 0:
                print(f"  {n}/{len(pages)} pages, {sum(counts.values())} crops, "
                      f"{time.perf_counter()-t0:.1f}s/page")
    finally:
        cluster_ocr._run_paddleocr_on_crops = real

    total = sum(counts.values())
    print("\n" + "=" * 68)
    print(f"LABEL CENSUS over {len(pages)} pages, {total} crops")
    print("=" * 68)
    if total == 0:
        print("  no crops were classified — process_rectangles needs real rectangle")
        print("  input; run this inside a full extraction instead.")
        return 1
    for label, c in counts.most_common():
        bar = "#" * int(60 * c / total)
        print(f"  {label:<12} {c:6d}  {c/total*100:6.2f}%  {bar}")
    hw = counts.get("handwriting", 0)
    print("\n" + "=" * 68)
    print(f"HANDWRITING SHARE: {hw}/{total} = {hw/total*100:.2f}%")
    if hw == 0:
        print("  -> the fix activates nothing on this corpus; zero risk.")
    elif hw / total < 0.01:
        print("  -> under 1%: low blast radius, safe to ship with monitoring.")
    else:
        print("  -> material share. These crops now go to a 91-class model instead of")
        print("     the 18,710-class one, with NO fallback if the result is garbage.")
        print("     A/B the text quality before shipping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
