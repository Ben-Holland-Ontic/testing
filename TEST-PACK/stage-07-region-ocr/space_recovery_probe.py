"""Four ways to get spaces out of three recognisers that cannot emit them.

DIAGNOSTIC. PARSeq (94), SAR (126) and MASTER (126) have no space in their charsets.
That is not an oversight: they are WORD-level scene-text recognisers, trained on crops
containing a single word. Handing them a whole line is using them wrong, and the missing
space is the symptom. So the options are about changing the INPUT unit, not the charset.

Strategies measured here, on the fixture crops that contain a space:

  whole      current behaviour -- one line crop in, one space-free string out
  gap        split the crop at wide background columns, read each piece, join with ' '
  doctr_det  use doctr's word detector (already loaded in this runtime for region_ocr)
             to get word boxes, read each, join with ' '
  donor      read the whole crop, then re-insert spaces from a model that HAS them
             (the primary, or CATMuS), aligned by edit distance

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY space_recovery_probe.py
"""
from __future__ import annotations

import difflib
import json
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

os.environ.setdefault("OCULI_SUPPORTING_MODEL_DIR", r"C:\OCULI.pd1\TESTING\onnx-models")
os.environ.setdefault("PADDLEOCR_USE_GPU", "1")

FIX = r"C:\OCULI.pd1\TESTING\RESOURCES\fixtures"
CROPS = os.path.join(FIX, "crops")
SPACELESS = ("parseq", "sar", "master")


# ---------------------------------------------------------------------------------
# Strategy: gap splitting
# ---------------------------------------------------------------------------------
def split_on_gaps(crop_bgr, min_gap_frac=0.28, pad=2):
    """Cut a line crop into word images at wide background columns.

    Measures the image rather than guessing at content: binarise, count ink per column,
    and treat a run of empty columns wider than `min_gap_frac` x the crop height as a
    word break. The threshold is relative to text height, so it does not care about
    resolution.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    _th, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (binary > 0).sum(axis=0)
    h, w = gray.shape
    min_gap = max(2, int(round(min_gap_frac * h)))

    # Runs of zero-ink columns that are wide enough to be a space.
    breaks = []
    run_start = None
    for x in range(w):
        if ink[x] == 0:
            if run_start is None:
                run_start = x
        else:
            if run_start is not None and x - run_start >= min_gap:
                breaks.append((run_start, x))
            run_start = None
    # A trailing run is margin, not a space.

    pieces, prev = [], 0
    for a, b in breaks:
        if a > prev:
            pieces.append((max(0, prev - pad), min(w, a + pad)))
        prev = b
    if prev < w:
        pieces.append((max(0, prev - pad), w))
    pieces = [(a, b) for a, b in pieces if b - a >= 3]
    if not pieces:
        return [crop_bgr]
    return [crop_bgr[:, a:b] for a, b in pieces]


# ---------------------------------------------------------------------------------
# Strategy: doctr word detection
# ---------------------------------------------------------------------------------
_DET = None


def split_on_doctr(crop_bgr, pad=2):
    """Use doctr's word detector to cut the crop into word boxes, left to right."""
    global _DET
    import cv2
    import numpy as np

    if _DET is None:
        import region_ocr
        _DET = region_ocr._get_doctr_predictor()

    h, w = crop_bgr.shape[:2]
    # The detector wants a page-like image; a 25px-tall strip detects poorly, so give it
    # room. Upscaling is cheap relative to a recognition pass.
    scale = max(1.0, 64.0 / max(1, h))
    big = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
    out = _DET([rgb])[0]
    boxes = out["words"] if isinstance(out, dict) else out
    if boxes is None or len(boxes) == 0:
        return [crop_bgr]
    xs = []
    for b in np.asarray(boxes):
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        xs.append((int(x1 * w), int(x2 * w)))
    xs.sort()
    pieces = []
    for a, b in xs:
        a = max(0, a - pad)
        b = min(w, b + pad)
        if b - a >= 3:
            pieces.append(crop_bgr[:, a:b])
    return pieces or [crop_bgr]


# ---------------------------------------------------------------------------------
# Strategy: space donor
# ---------------------------------------------------------------------------------
def reinsert_spaces(candidate: str, donor: str) -> str:
    """Put donor's spaces into a space-free candidate, aligned by edit distance.

    The donor supplies LAYOUT (where the words break), the candidate supplies
    CHARACTERS. Neither has to be right about the other's job.
    """
    if not candidate or " " not in donor:
        return candidate
    donor_ns = donor.replace(" ", "")
    if not donor_ns:
        return candidate
    # Index in donor_ns after which a space falls.
    space_after = set()
    k = 0
    for ch in donor:
        if ch == " ":
            if k > 0:
                space_after.add(k - 1)
        else:
            k += 1

    sm = difflib.SequenceMatcher(a=donor_ns, b=candidate, autojunk=False)
    # donor_ns index -> candidate index
    mapping = {}
    for a0, b0, size in sm.get_matching_blocks():
        for d in range(size):
            mapping[a0 + d] = b0 + d
    cut_after = set()
    for di in space_after:
        # Nearest mapped donor index at or before the break.
        j = di
        while j >= 0 and j not in mapping:
            j -= 1
        if j >= 0:
            cut_after.add(mapping[j])

    out = []
    for i, ch in enumerate(candidate):
        out.append(ch)
        if i in cut_after and i != len(candidate) - 1:
            out.append(" ")
    return "".join(out)


# ---------------------------------------------------------------------------------
def cer(a: str, b: str) -> float:
    """Character error rate of `a` against reference `b`."""
    if not b:
        return 0.0 if not a else 1.0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n] / n


def main() -> int:
    import cv2
    import ocr_supporting as S

    manifest = json.load(open(os.path.join(FIX, "crops.json"), encoding="utf-8"))
    items = [(m, cv2.imread(os.path.join(CROPS, m["file"]))) for m in manifest]
    items = [(m, im) for m, im in items if im is not None and " " in m["primary_text"]]
    print(f"{len(items)} multi-word crops\n")

    catmus = S.get_recogniser("catmus")
    catmus_reads = {m["file"]: t for (m, _), (t, _c) in
                    zip(items, catmus.recognise([im for _, im in items]))} if catmus else {}

    results = {}
    timings = {}
    for name in SPACELESS:
        rec = S.get_recogniser(name)
        if rec is None:
            print(f"  {name}: unavailable")
            continue
        results[name] = {}
        timings[name] = {}

        for strategy in ("whole", "gap", "doctr_det", "donor_primary", "donor_catmus"):
            t0 = time.perf_counter()
            reads = {}
            for m, im in items:
                if strategy == "whole":
                    reads[m["file"]] = rec.recognise([im])[0][0]
                elif strategy in ("gap", "doctr_det"):
                    pieces = (split_on_gaps(im) if strategy == "gap" else split_on_doctr(im))
                    words = [t for t, _ in rec.recognise(pieces)]
                    reads[m["file"]] = " ".join(w for w in words if w)
                elif strategy == "donor_primary":
                    reads[m["file"]] = reinsert_spaces(
                        rec.recognise([im])[0][0], m["primary_text"])
                else:
                    reads[m["file"]] = reinsert_spaces(
                        rec.recognise([im])[0][0], catmus_reads.get(m["file"], ""))
            timings[name][strategy] = (time.perf_counter() - t0) / max(1, len(items)) * 1000
            results[name][strategy] = reads

    # ---- report -------------------------------------------------------------------
    strategies = ["whole", "gap", "doctr_det", "donor_primary", "donor_catmus"]
    print(f"{'model':<8} {'strategy':<14} {'exact':<8} {'meanCER':<9} {'ms/crop':<8}")
    print("-" * 52)
    for name in results:
        for strategy in strategies:
            reads = results[name].get(strategy)
            if reads is None:
                continue
            exact = sum(1 for m, _ in items if reads[m["file"]] == m["primary_text"])
            mean_cer = sum(cer(reads[m["file"]], m["primary_text"]) for m, _ in items) / len(items)
            print(f"{name:<8} {strategy:<14} {exact}/{len(items):<6} {mean_cer:<9.3f} "
                  f"{timings[name][strategy]:<8.1f}")
        print()

    print("\nPer-crop detail (reference = primary):")
    for m, _ in items:
        print(f"\n  {m['file']}  ref={m['primary_text']!r}")
        print(f"    {'catmus':<22} {catmus_reads.get(m['file'], '')!r}")
        for name in results:
            for strategy in strategies:
                if strategy in results[name]:
                    print(f"    {name + '/' + strategy:<22} {results[name][strategy][m['file']]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
