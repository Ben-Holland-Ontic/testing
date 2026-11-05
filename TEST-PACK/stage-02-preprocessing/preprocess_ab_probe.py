"""Does the primary's preprocessing help or hurt the supporting models?

DIAGNOSTIC. `cluster_ocr` preprocesses every crop before OCR (cluster_ocr.py:1673):
upscale ladders, Otsu and contrast rescue for `numeric`, denoise for `blurry`, thin-text
dilation. All of that is tuned for PP-OCRv6, and the supporting models never had a say.

PARSeq, SAR and MASTER are SCENE-TEXT models -- trained on photographs of signs, labels
and packaging, where text is greyscale, anti-aliased and noisy. Handing them a binarised,
dilated, upscaled crop is not obviously the favour it is to a CTC model trained on
document scans. It may be actively harmful.

So: capture both the RAW crop and the PREPROCESSED crop for the same region, take the
primary's detected line box on the preprocessed image, map it onto the raw image, and ask
every supporting model about both. Same line, same geometry, two versions of the pixels.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY preprocess_ab_probe.py
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

OUT = r"C:\OCULI.pd1\TESTING\RESULTS\preprocess_ab"
DATA = r"C:\Users\benja\OneDrive\Desktop\data"
CHUNK = 32
MAX_PAIRS = 320


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


def key(t: str) -> str:
    return "".join(t.split()).casefold()


def capture_pairs():
    """(raw_crop, processed_crop) for every crop the pipeline preprocesses."""
    import cv2
    import cluster_ocr
    import pipeline
    from process_input import PageData

    pairs = []
    original = cluster_ocr._preprocess_for_ocr

    def hook(crop, label, **kw):
        processed = original(crop, label, **kw)
        if crop is not None and processed is not None and crop.size and processed.size:
            pairs.append((crop.copy(), processed.copy(), label))
        return processed

    cluster_ocr._preprocess_for_ocr = hook
    try:
        for n, name in enumerate(sorted(f for f in os.listdir(DATA)
                                        if f.lower().endswith((".png", ".jpg", ".jpeg"))), 1):
            img = cv2.imread(os.path.join(DATA, name))
            if img is None:
                continue
            page = PageData(page_id=f"AB_PG{n:03d}", document_id="AB_DOC",
                            batch_id="AB_BATCH", page_number=n, image=img,
                            width=img.shape[1], height=img.shape[0],
                            storage_path=os.path.join(DATA, name))
            pipeline.process_page_extraction(page)
            print(f"  {name}: {len(pairs)} pairs so far")
    finally:
        cluster_ocr._preprocess_for_ocr = original
    return pairs


def main() -> int:
    import cv2
    import cluster_ocr
    import ocr_backend
    import ocr_supporting as S

    ocr_backend.register_cuda_dll_dirs()
    os.makedirs(OUT, exist_ok=True)

    print("capturing raw/preprocessed pairs from a real extraction run...")
    pairs = capture_pairs()
    print(f"\n{len(pairs)} crops preprocessed")

    engine = cluster_ocr._get_paddleocr()
    rows = []           # (ref_text, raw_line, proc_line)
    changed_size = 0
    for raw, proc, _label in pairs:
        if len(rows) >= MAX_PAIRS:
            break
        res = cluster_ocr._normalize_paddle_list_result(engine.ocr(proc))
        lines = cluster_ocr._parse_paddle_result_to_lines(res, 0, 0, 0, "AB")
        if not lines:
            continue
        sy = raw.shape[0] / max(1, proc.shape[0])
        sx = raw.shape[1] / max(1, proc.shape[1])
        if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
            changed_size += 1
        for ln in lines:
            text = (ln.get("text") or "").strip()
            conf = float(ln.get("confidence", 0.0) or 0.0)
            bbox = ln.get("bbox") or []
            if not text or conf < 0.95 or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
            px1, py1 = max(0, int(x1) - 2), max(0, int(y1) - 2)
            px2 = min(proc.shape[1], int(x2) + 2)
            py2 = min(proc.shape[0], int(y2) + 2)
            rx1, ry1 = max(0, int(x1 * sx) - 2), max(0, int(y1 * sy) - 2)
            rx2 = min(raw.shape[1], int(x2 * sx) + 2)
            ry2 = min(raw.shape[0], int(y2 * sy) + 2)
            if px2 - px1 < 6 or py2 - py1 < 6 or rx2 - rx1 < 6 or ry2 - ry1 < 6:
                continue
            rows.append((text, raw[ry1:ry2, rx1:rx2], proc[py1:py2, px1:px2]))

    print(f"{len(rows)} line regions with a confident reference "
          f"({changed_size} crops were resized by preprocessing)\n")
    if not rows:
        print("nothing to compare")
        return 1

    results = {}
    for label, index in (("PREPROCESSED (what ships today)", 2), ("RAW (no preprocessing)", 1)):
        per = {n: [] for n in S.MODEL_NAMES}
        exact = {n: 0 for n in S.MODEL_NAMES}
        for off in range(0, len(rows), CHUNK):
            part = rows[off:off + CHUNK]
            imgs = [r[index] for r in part]
            refs = [r[0] for r in part]
            for n in S.MODEL_NAMES:
                rec = S.get_recogniser(n)
                if rec is None:
                    continue
                for ref, (got, _c) in zip(refs, rec.recognise(imgs)):
                    per[n].append(cer(key(got), key(ref)))
                    exact[n] += (key(got) == key(ref))
            del imgs
        results[label] = (per, exact)

    labels = list(results)
    print("=" * 78)
    print(f"{'model':<10} {labels[0]:<30} {labels[1]:<26} {'better?'}")
    print("=" * 78)
    for n in S.MODEL_NAMES:
        pa, ea = results[labels[0]]
        pb, eb = results[labels[1]]
        if not pa[n]:
            continue
        ca, cb = statistics.mean(pa[n]), statistics.mean(pb[n])
        xa, xb = ea[n] / len(pa[n]), eb[n] / len(pb[n])
        verdict = "RAW" if cb < ca else "preprocessed"
        print(f"{n:<10} CER {ca:<6.3f} exact {xa:<12.1%} CER {cb:<6.3f} exact {xb:<9.1%} "
              f"{verdict} (exact {xb - xa:+.1%})")

    print("\nThe reference is the primary's own reading of the PREPROCESSED crop, so this")
    print("measures agreement with the primary, not absolute truth -- but the comparison")
    print("between the two columns is like for like.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
