"""Are the supporting models bad, or were they being fed the wrong pixels?

DIAGNOSTIC, and it settles an argument I got wrong once already.

An earlier measurement handed each supporting model the WHOLE crop and reported character
error rates of 0.50-0.76 against the primary, and I concluded the models were too weak to
ensemble. That comparison was not fair. The primary runs DETECTION inside the crop and
recognises the line box it finds; the supporting models were handed the entire rectangle
-- cell borders, surrounding geometry, whitespace and all. Two different images, one
scored against the other.

This runs the comparison the plan actually specified: take the primary's own detected line
regions, crop to them, and ask the supporting models about THOSE pixels. Same input unit,
same content, 1:1 comparable.

If CER collapses, the models were fine and the architecture was wrong.
If it does not move, the models really are out of domain and the ensemble has no basis.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY line_region_probe.py
"""
from __future__ import annotations

import json
import os
import statistics
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
SAMPLE = 300          # crops to test; keep it bounded so this stays a quick experiment
CHUNK = 32


def cer(a: str, b: str) -> float:
    if not b:
        return 0.0 if not a else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(b)


def key(text: str) -> str:
    return "".join(text.split()).casefold()


def main() -> int:
    import cv2
    import cluster_ocr
    import ocr_backend
    import ocr_supporting as S

    ocr_backend.register_cuda_dll_dirs()
    engine = cluster_ocr._get_paddleocr()

    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    pool = [m for m in manifest if m["primary_text"] and m["primary_conf"] >= 0.95]
    pool = pool[:SAMPLE]
    print(f"{len(pool)} crops the primary read at conf >= 0.95\n")

    # ---- pass 1: ask the primary WHERE the text is, and crop to it ----------------
    whole_imgs, line_imgs, refs = [], [], []
    multi = 0
    for m in pool:
        img = cv2.imread(os.path.join(CROP_DIR, m["file"]))
        if img is None:
            continue
        out = engine.ocr(img)
        res = cluster_ocr._normalize_paddle_list_result(out)
        lines = cluster_ocr._parse_paddle_result_to_lines(res, 0, 0, 0, "PROBE")
        if not lines:
            continue
        if len(lines) > 1:
            multi += 1
        for ln in lines:
            text = (ln.get("text") or "").strip()
            bbox = ln.get("bbox") or []
            if not text or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
            x1 = max(0, x1 - 2); y1 = max(0, y1 - 2)
            x2 = min(img.shape[1], x2 + 2); y2 = min(img.shape[0], y2 + 2)
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            line_imgs.append(img[y1:y2, x1:x2])
            whole_imgs.append(img)
            refs.append(text)

    print(f"{len(refs)} detected line regions "
          f"({multi} of the crops held more than one line)\n")
    if not refs:
        print("nothing to compare")
        return 1

    # ---- pass 2: same models, two different inputs --------------------------------
    results = {}
    for label, images in (("WHOLE CROP (what I measured before)", whole_imgs),
                          ("DETECTED LINE REGION (what the plan specified)", line_imgs)):
        per = {n: [] for n in S.MODEL_NAMES}
        exact = {n: 0 for n in S.MODEL_NAMES}
        for off in range(0, len(images), CHUNK):
            batch = images[off:off + CHUNK]
            batch_refs = refs[off:off + CHUNK]
            for n in S.MODEL_NAMES:
                rec = S.get_recogniser(n)
                if rec is None:
                    continue
                for ref, (got, _c) in zip(batch_refs, rec.recognise(batch)):
                    per[n].append(cer(key(got), key(ref)))
                    exact[n] += (key(got) == key(ref))
        results[label] = (per, exact)

    print("=" * 74)
    print(f"{'model':<10} {'WHOLE CROP':<26} {'DETECTED LINE REGION':<26} {'change'}")
    print(f"{'':<10} {'CER      exact':<26} {'CER      exact':<26}")
    print("=" * 74)
    wl = list(results)[0]
    ll = list(results)[1]
    for n in S.MODEL_NAMES:
        pw, ew = results[wl]
        pl, el = results[ll]
        if not pw[n]:
            continue
        cw, cl = statistics.mean(pw[n]), statistics.mean(pl[n])
        xw, xl = ew[n] / len(pw[n]), el[n] / len(pl[n])
        print(f"{n:<10} {cw:<8.3f} {xw:<17.1%} {cl:<8.3f} {xl:<17.1%} "
              f"CER {cw - cl:+.3f}, exact {xl - xw:+.1%}")

    print("\nIf the right-hand column is much better, the models were never the problem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
