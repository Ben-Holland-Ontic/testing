"""What the four supporting recognisers actually cost, under TensorRT, on real crops.

DIAGNOSTIC. The cascade budget in the plan (~1.2 ms/crop) was built from numbers measured
on the CUDA EP with random tensors and a free choice of batch size. Three things have
changed since:

  * the primary needed TensorRT on this card because the CUDA EP re-initialises per
    distinct input shape, so CUDA-EP numbers are not the ones the cascade will pay;
  * CATMuS cannot be batched across differing widths (it changes the reading), so its
    batch-8 number was never achievable in the first place;
  * SAR and MASTER now split wide crops, which means more than one inference per crop.

This measures the WHOLE path a cascade tier pays -- preprocess, infer, decode -- on the
real fixture crops, and reports preprocessing separately because it runs on the CPU and
can dominate a cheap model.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY supporting_latency_probe.py            # TensorRT (default)
    $env:OCULI_OCR_TRT='0'; & $PY supporting_latency_probe.py    # CUDA EP, for contrast
"""
from __future__ import annotations

import json
import os
import statistics
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
REPEATS = 20


def _load_crops():
    import cv2
    manifest = json.load(open(os.path.join(FIX, "crops.json"), encoding="utf-8"))
    out = []
    for m in manifest:
        img = cv2.imread(os.path.join(CROPS, m["file"]))
        if img is not None:
            out.append(img)
    return out


def _time(fn, repeats=REPEATS):
    """Median of `repeats` runs, in ms. Median, not mean: one TRT re-init or a GC pause
    should not set the number the cascade budget is built on."""
    fn()  # warm
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples), min(samples)


def main() -> int:
    import numpy as np
    import ocr_backend
    import ocr_supporting as S

    ocr_backend.register_cuda_dll_dirs()
    trt_on = ocr_backend.trt_available()
    print(f"TensorRT available: {trt_on}   (OCULI_OCR_TRT="
          f"{os.environ.get('OCULI_OCR_TRT', '1')})")
    print(f"engine cache: {ocr_backend.trt_cache_dir()}\n")

    crops = _load_crops()
    print(f"{len(crops)} real fixture crops\n")

    # ---- load, and report what each session ACTUALLY bound -------------------------
    print("=" * 74)
    print("BINDING AND ENGINE BUILD")
    print("=" * 74)
    print(f"{'model':<8} {'load s':<9} {'first call s':<14} {'bound providers'}")
    recs = {}
    for name in S.MODEL_NAMES:
        t0 = time.perf_counter()
        rec = S.get_recogniser(name)
        load_s = time.perf_counter() - t0
        if rec is None:
            print(f"{name:<8} {'-':<9} {'-':<14} UNAVAILABLE")
            continue
        recs[name] = rec
        t0 = time.perf_counter()
        rec.recognise(crops[:1])
        first_s = time.perf_counter() - t0
        bound = [p.replace("ExecutionProvider", "") for p in rec.sess.get_providers()]
        flag = "" if (not trt_on or "Tensorrt" in bound) else "   <-- NOT on TensorRT"
        print(f"{name:<8} {load_s:<9.2f} {first_s:<14.2f} {','.join(bound)}{flag}")

    if not recs:
        print("\nno models loaded; nothing to measure")
        return 1

    # ---- per-crop cost at the batch sizes the cascade could actually use -----------
    print("\n" + "=" * 74)
    print("PER-CROP COST, WHOLE PATH (preprocess + infer + decode), REAL CROPS")
    print("=" * 74)
    print(f"{'model':<8} {'batch':<7} {'ms/batch':<11} {'ms/crop':<10} {'best ms/crop'}")
    per_crop = {}
    for name, rec in recs.items():
        per_crop[name] = {}
        for bs in (1, 4, 8, 16):
            batch = (crops * ((bs // len(crops)) + 1))[:bs]
            med, best = _time(lambda b=batch: rec.recognise(b))
            per_crop[name][bs] = med / bs
            print(f"{name:<8} {bs:<7} {med:<11.2f} {med / bs:<10.3f} {best / bs:.3f}")
        print()

    # ---- how much of that is CPU preprocessing? -----------------------------------
    print("=" * 74)
    print("PREPROCESSING SHARE (CPU, runs whether or not the GPU is fast)")
    print("=" * 74)
    preps = {"catmus": lambda b: [S.catmus_line_tensor(c) for c in b],
             "parseq": S.parseq_batch,
             "sar": S.doctr_batch,
             "master": S.doctr_batch}
    for name in recs:
        batch = (crops * 2)[:8]
        med, _ = _time(lambda f=preps[name], b=batch: f(b))
        total = per_crop[name][8] * 8
        print(f"{name:<8} {med / 8:>7.3f} ms/crop   ({100 * med / total:>5.1f}% of the "
              f"batch-8 total)")

    # ---- CATMuS: what batching is actually available? ------------------------------
    print("\n" + "=" * 74)
    print("CATMuS WIDTH GROUPING -- the batch size it can really get")
    print("=" * 74)
    widths = {}
    for c in crops:
        w = S.catmus_line_tensor(c).shape[2]
        widths.setdefault(w, 0)
        widths[w] += 1
    groups = sorted(widths.values(), reverse=True)
    print(f"{len(crops)} crops -> {len(widths)} distinct widths; group sizes {groups}")
    print(f"mean achievable batch = {len(crops) / len(widths):.2f}")
    if "catmus" in recs:
        med, _ = _time(lambda: recs["catmus"].recognise(crops))
        print(f"realistic mixed-width call: {med:.2f} ms for {len(crops)} crops "
              f"= {med / len(crops):.3f} ms/crop")
        med1, _ = _time(lambda: [recs["catmus"].recognise([c]) for c in crops[:8]])
        print(f"forced one-at-a-time      : {med1 / 8:.3f} ms/crop")

    # ---- what the cascade would cost ----------------------------------------------
    print("\n" + "=" * 74)
    print("CASCADE BUDGET, RECOMPUTED")
    print("=" * 74)
    esc, t2, t3 = 0.37, 0.40, 0.30
    if "catmus" in recs:
        med, _ = _time(lambda: recs["catmus"].recognise(crops))
        c1 = med / len(crops)
    else:
        c1 = float("nan")
    spec = [per_crop[n][8] for n in ("parseq", "sar") if n in per_crop]
    c2 = sum(spec) / len(spec) if spec else float("nan")
    c3 = per_crop.get("master", {}).get(8, float("nan"))
    tier1, tier2, tier3 = esc * c1, esc * t2 * c2, esc * t2 * t3 * c3
    print(f"  Tier 1 CATMuS   {esc:.2f}            x {c1:7.3f} = {tier1:7.3f} ms/crop")
    print(f"  Tier 2 one spec {esc:.2f} x {t2:.2f}     x {c2:7.3f} = {tier2:7.3f} ms/crop")
    print(f"  Tier 3 MASTER   {esc:.2f} x {t2:.2f} x {t3:.2f} x {c3:7.3f} = {tier3:7.3f} ms/crop")
    print(f"  {'TOTAL':<48}{tier1 + tier2 + tier3:7.3f} ms/crop")
    allfive = sum(per_crop[n][8] for n in per_crop)
    print(f"  (running everything on every crop would be {allfive:.3f} ms/crop)")
    print("\n  Branch ratios are still the plan's assumptions; the per-model costs are "
          "now measured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
