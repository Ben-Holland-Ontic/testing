"""Verify the substitution end-to-end through the PATCHED cluster_ocr, with no paddle.

DIAGNOSTIC. Exercises the shipped code path — cluster_ocr._run_paddleocr_on_crops — not a
reimplementation of it, so it proves the edits work rather than that ONNX works.

    # paddle-free (the substituted stack)
    & 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe' substitution_probe.py

    # legacy runner, for a side-by-side baseline
    $env:OCULI_OCR_BACKEND='paddle'
    & 'C:\\OCULI.pd1\\TESTING\\.venv-release\\Scripts\\python.exe' substitution_probe.py
"""
from __future__ import annotations

import importlib.util
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

os.environ.setdefault("PADDLEOCR_USE_GPU", "1")
# The subprocess split is paddle-only; the ONNX path ignores this either way.
os.environ.setdefault("OCULI_PADDLE_SUBPROCESS", "0")

HAVE_PADDLE = importlib.util.find_spec("paddle") is not None
IMG = os.path.join(BK, "test_data", "01_original.png")
REF = r"C:\OCULI.pd1\TESTING\RESULTS\substitution_reference.json"


def stub_modelscope() -> None:
    """Only the PADDLE baseline needs this, and that is itself the point.

    paddlex imports modelscope, which imports torch. On the paddle path that is fatal —
    paddle has already loaded cuDNN, so torch's DLL load dies with
    WinError 127 ... torch\\lib\\cudnn_engines_precompiled64_9.dll (observed here). It is
    why paddle_ipc._server_init stubs modelscope out in the first place.

    The ONNX path has no such conflict; the stub is a no-op there. Applied unconditionally
    so both arms of the comparison run under identical conditions.
    """
    import types
    if "modelscope" in sys.modules:
        return

    class _Stub(types.ModuleType):
        __path__ = []

        def __getattr__(self, name):
            full = f"{self.__name__}.{name}"
            m = sys.modules.get(full)
            if m is None:
                m = _Stub(full)
                sys.modules[full] = m
            return m

    for nm in ("modelscope", "modelscope.hub", "modelscope.hub.errors",
               "modelscope.hub.snapshot_download", "modelscope.utils"):
        sys.modules.setdefault(nm, _Stub(nm))


def main() -> int:
    stub_modelscope()
    import numpy as np
    import cv2
    import ocr_backend
    import cluster_ocr

    print("=" * 74)
    print(f"backend       : {ocr_backend.active_backend()}")
    print(f"describe      : {ocr_backend.describe_backend()}")
    print(f"paddle present: {HAVE_PADDLE}")
    print(f"gpu_available : {cluster_ocr._gpu_available()}")
    print("=" * 74)

    det_dir, rec_dir = cluster_ocr._get_paddleocr_model_dirs()
    print(f"det dir (paddle-side): {det_dir}")
    print(f"onnx dir resolved    : {ocr_backend.onnx_model_dir_for(det_dir)}")

    t0 = time.perf_counter()
    engine = cluster_ocr._get_paddleocr()
    print(f"\nengine built in {time.perf_counter() - t0:.1f}s -> {type(engine).__name__}")

    # Crops in the shape _run_paddleocr_on_crops expects: rect_infos are
    # (gx, gy, rect_index, poly_id) 4-tuples, per cluster_ocr._run_batch.
    im = cv2.imread(IMG)
    if im is None:
        print(f"could not read {IMG}"); return 2
    h, w = im.shape[:2]
    tiles, infos = [], []
    idx = 0
    for gy in range(0, min(h, 1800), 600):
        for gx in range(0, min(w, 2400), 800):
            tile = im[gy:gy + 600, gx:gx + 800]
            if tile.shape[0] >= 32 and tile.shape[1] >= 32:
                tiles.append(tile)
                infos.append((gx, gy, idx, None))
                idx += 1
    print(f"tiles: {len(tiles)}")

    t0 = time.perf_counter()
    results = cluster_ocr._run_paddleocr_on_crops(tiles, infos, None)
    dt = time.perf_counter() - t0

    lines = [ln for grp in results for ln in (grp or [])]
    texts = [ln.get("text", "") for ln in lines]
    print(f"\n_run_paddleocr_on_crops: {len(lines)} lines in {dt:.2f}s")
    print(f"first 10: {texts[:10]}")

    payload = {"backend": ocr_backend.active_backend(), "n_lines": len(lines),
               "texts": texts, "seconds": dt}
    if HAVE_PADDLE and ocr_backend.active_backend() == "paddle":
        with open(REF, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print(f"\nwrote baseline -> {REF}")
    elif os.path.isfile(REF):
        with open(REF, encoding="utf-8") as fh:
            ref = json.load(fh)
        rt = ref["texts"]
        n = min(len(rt), len(texts))
        idx_same = sum(1 for i in range(n) if rt[i] == texts[i])
        set_same = sum(1 for x in sorted(rt) if x in sorted(texts))
        print("\n" + "=" * 74)
        print("vs the paddle baseline (same code path, OCULI_OCR_BACKEND=paddle)")
        print("=" * 74)
        print(f"  lines   : {ref['n_lines']} -> {len(lines)}")
        print(f"  by index: {idx_same}/{n} ({idx_same / max(n,1) * 100:.2f}%)")
        print(f"  as a set: {set_same}/{len(rt)} ({set_same / max(len(rt),1) * 100:.2f}%)")
        print(f"  seconds : {ref['seconds']:.2f} -> {dt:.2f}")
        shown = 0
        for i in range(n):
            if rt[i] != texts[i] and shown < 10:
                print(f"    [{i}] paddle={rt[i]!r}  onnx={texts[i]!r}")
                shown += 1
    else:
        print(f"\n(no baseline at {REF}; run with OCULI_OCR_BACKEND=paddle first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
