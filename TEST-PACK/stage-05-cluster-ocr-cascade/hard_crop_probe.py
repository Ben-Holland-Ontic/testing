"""Capture the crops the PRIMARY fails on, and see whether the cascade recovers them.

DIAGNOSTIC, and the one that decides whether this whole plan is worth shipping.

Everything measured so far used a fixture set built from lines the primary read at
confidence >= 0.95 -- the EASY crops. On those, voting is safe (20/20 kept, 0
displacements) but by construction it cannot show a gain. The case the plan exists for is
the opposite one: crops that contain ink and return NOTHING, measured at 37% of crops.

So this hooks the real dispatch (`cluster_ocr._run_paddleocr_on_crops_impl`), runs real
page extraction over the reference pages, and keeps every crop together with what the
primary made of it. Then it runs the four supporting models and the vote over the failures
and reports what was recovered.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY hard_crop_probe.py                 # capture + measure
    & $PY hard_crop_probe.py --capture-only
"""
from __future__ import annotations

import argparse
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
DATA = r"C:\Users\benja\OneDrive\Desktop\data"
BASELINE = r"C:\OCULI.pd1\TESTING\RESULTS\parity\phase0_baseline.json"

# A crop counts as "has ink" on the same terms the plan's trigger uses: a dark-pixel
# fraction above a floor, on a crop big enough to hold a character at all.
INK_RATIO_MIN = 0.02
MIN_SIDE = 8


def ink_ratio(crop):
    import cv2
    import numpy as np
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if min(gray.shape[:2]) < MIN_SIDE:
        return 0.0
    _t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float((binary > 0).mean())


def capture() -> int:
    """Run real extraction over the reference pages, keeping every crop."""
    import cv2
    import cluster_ocr
    import pipeline
    from process_input import PageData

    os.makedirs(CROP_DIR, exist_ok=True)
    captured = []
    original = cluster_ocr._run_paddleocr_on_crops_impl

    state = {"page": ""}

    def hook(crops_bgr, rect_infos, labels=None):
        results = original(crops_bgr, rect_infos, labels)
        for i, crop in enumerate(crops_bgr):
            if crop is None or crop.size == 0:
                continue
            lines = results[i] if i < len(results) else []
            text = " ".join((l.get("text") or "") for l in lines).strip()
            confs = [float(l.get("confidence", 0.0) or 0.0) for l in lines]
            captured.append({
                "page": state["page"],
                "index": len(captured),
                "label": (labels[i] if labels and i < len(labels) else None),
                "n_lines": len(lines),
                "primary_text": text,
                "primary_conf": (sum(confs) / len(confs)) if confs else 0.0,
                "size_hw": [int(crop.shape[0]), int(crop.shape[1])],
                "ink": round(ink_ratio(crop), 4),
                "_crop": crop,
            })
        return results

    cluster_ocr._run_paddleocr_on_crops_impl = hook
    try:
        pages = sorted(f for f in os.listdir(DATA)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))
        for n, name in enumerate(pages, 1):
            img = cv2.imread(os.path.join(DATA, name))
            if img is None:
                continue
            state["page"] = name
            page = PageData(
                page_id=f"PROBE_PG{n:03d}", document_id="PROBE_DOC",
                batch_id="PROBE_BATCH", page_number=n, image=img,
                width=img.shape[1], height=img.shape[0],
                storage_path=os.path.join(DATA, name),
            )
            before = len(captured)
            pipeline.process_page_extraction(page)
            print(f"  {name}: {len(captured) - before} crops")
    finally:
        cluster_ocr._run_paddleocr_on_crops_impl = original

    manifest = []
    for rec in captured:
        crop = rec.pop("_crop")
        fname = f"{rec['index']:05d}.png"
        cv2.imwrite(os.path.join(CROP_DIR, fname), crop)
        rec["file"] = fname
        manifest.append(rec)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\ncaptured {len(manifest)} crops -> {MANIFEST}")
    return 0


def measure() -> int:
    import cv2
    import ocr_backend
    import ocr_credibility as Cr
    import ocr_supporting as S
    import ocr_voting as V

    ocr_backend.register_cuda_dll_dirs()
    manifest = json.load(open(MANIFEST, encoding="utf-8"))

    empty_inked = [m for m in manifest
                   if not m["primary_text"] and m["ink"] >= INK_RATIO_MIN
                   and min(m["size_hw"]) >= MIN_SIDE]
    empty_blank = [m for m in manifest
                   if not m["primary_text"] and (m["ink"] < INK_RATIO_MIN
                                                 or min(m["size_hw"]) < MIN_SIDE)]
    low_conf = [m for m in manifest if m["primary_text"] and m["primary_conf"] < 0.85]
    good = [m for m in manifest if m["primary_text"] and m["primary_conf"] >= 0.85]

    total = len(manifest)
    print("=" * 72)
    print("WHAT THE PRIMARY DID, ACROSS EVERY CROP THE PIPELINE ACTUALLY SAW")
    print("=" * 72)
    for name, group in (("read it, conf >= 0.85", good),
                        ("read it, conf < 0.85", low_conf),
                        ("EMPTY but contains ink", empty_inked),
                        ("empty, no ink (correctly)", empty_blank)):
        print(f"  {name:<28} {len(group):>5}  {100 * len(group) / max(1, total):>5.1f}%")
    print(f"  {'TOTAL':<28} {total:>5}")

    targets = empty_inked + low_conf
    if not targets:
        print("\nnothing to recover")
        return 0

    # Batch statistics from what the primary DID read -- the credibility layer's real
    # input at this point in a run.
    corpus = [m["primary_text"] for m in good]
    stats = Cr.build_statistics((t, 1.0) for t in corpus)
    cred = lambda t: Cr.credibility(t, stats).total  # noqa: E731

    # STREAMED, in chunks. An earlier version of this probe loaded every crop image and
    # handed ~500 to a model at once: it took 28GB of RAM and made the machine unusable,
    # and it blew the TensorRT profile (max batch 32) so every model silently fell back
    # to the CUDA EP. Both are fixed -- ocr_supporting caps batches now -- but a
    # diagnostic still has no business holding a whole corpus of images in memory.
    CHUNK = 32
    print(f"\nrunning 4 supporting models over {len(targets)} failed/weak crops "
          f"in chunks of {CHUNK}...")

    recovered, still_empty, improved_low, n_empty, n_seen = [], 0, [], 0, 0
    for offset in range(0, len(targets), CHUNK):
        imgs, metas = [], []
        for m in targets[offset:offset + CHUNK]:
            img = cv2.imread(os.path.join(CROP_DIR, m["file"]))
            if img is not None:
                imgs.append(img)
                metas.append(m)
        if not imgs:
            continue

        reads = {}
        for name in S.MODEL_NAMES:
            rec = S.get_recogniser(name)
            reads[name] = rec.recognise(imgs) if rec else [("", 0.0)] * len(imgs)

        for idx, m in enumerate(metas):
            n_seen += 1
            cands = [V.Candidate("primary", m["primary_text"], m["primary_conf"])]
            cands += [V.Candidate(n, reads[n][idx][0], reads[n][idx][1])
                      for n in S.MODEL_NAMES]
            donor = S.choose_donor(reads["catmus"][idx][0], m["primary_text"])
            d = V.decide(cands, donor=donor, credibility_of=cred)
            row = {**{k: m[k] for k in ("file", "page", "size_hw", "ink", "primary_text")},
                   "vote": d.text, "reason": d.reason,
                   "models": {n: reads[n][idx][0] for n in S.MODEL_NAMES}}
            if not m["primary_text"]:
                n_empty += 1
                if d.text:
                    recovered.append(row)
                else:
                    still_empty += 1
            elif d.changed:
                improved_low.append(row)

        del imgs, reads
        if (offset // CHUNK) % 5 == 0:
            print(f"    {n_seen}/{len(targets)} crops, {len(recovered)} recovered so far")

    print("\n" + "=" * 72)
    print("RECOVERY")
    print("=" * 72)
    print(f"  empty-but-inked crops tried     {n_empty}")
    print(f"  recovered (corroborated text)   {len(recovered)}"
          f"   {100 * len(recovered) / max(1, n_empty):.1f}%")
    print(f"  still empty                     {still_empty}")
    print(f"  low-confidence crops changed    {len(improved_low)} of {len(low_conf)}")

    with open(os.path.join(OUT, "recovered.json"), "w", encoding="utf-8") as f:
        json.dump({"recovered": recovered, "changed_low_conf": improved_low},
                  f, indent=2, ensure_ascii=False)

    print("\nsample of recovered text (verify these by eye before believing them):")
    for row in recovered[:25]:
        models = "  ".join(f"{k}={v!r}" for k, v in row["models"].items() if v)
        print(f"  {row['file']} {row['size_hw']} ink={row['ink']}")
        print(f"      -> {row['vote']!r}")
        print(f"         {models}")
    print(f"\nfull list -> {os.path.join(OUT, 'recovered.json')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-only", action="store_true")
    ap.add_argument("--measure-only", action="store_true")
    args = ap.parse_args()
    if not args.measure_only:
        if capture():
            return 1
    if not args.capture_only:
        return measure()
    return 0


if __name__ == "__main__":
    sys.exit(main())
