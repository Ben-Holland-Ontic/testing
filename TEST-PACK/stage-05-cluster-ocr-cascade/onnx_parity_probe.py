"""Step 3: numerical parity between the shipped paddle models and their ONNX conversions.

DIAGNOSTIC ONLY. Reads the release models read-only; changes nothing.

WHY TENSOR-LEVEL PARITY IS THE RIGHT GATE
-----------------------------------------
paddlex's preprocessing and postprocessing live ABOVE the runner - `runner_builder` swaps
only the inference call, so det/rec transforms, DB thresholds and the tuned det_limit /
unclip_ratio / box_thresh values are shared by both backends. If the raw det and rec
tensors agree for identical input, the text and boxes must agree too, because everything
downstream is the same Python code.

Comparing raw tensors is also far more sensitive than comparing decoded strings: a
recogniser can produce identical text from noticeably different logits, so string equality
would hide a real regression until some other document surfaced it.

WHAT IT DOES
------------
For det and rec, over several input shapes spanning the real workload:
  * runs the paddle inference predictor (the shipped .pdiparams)
  * runs onnxruntime on the converted .onnx
  * reports max/mean absolute and relative difference, and for rec also whether the
    argmax character sequence is identical (the thing that actually becomes text)

Run with the release venv:

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-release\\Scripts\\python.exe'
    & $PY onnx_parity_probe.py --device gpu
    & $PY onnx_parity_probe.py --device cpu      # if the GPU predictor crashes (phi.dll)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

MODELS = r"C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main\models\paddleocr"
ONNX = r"C:\OCULI.pd1\TESTING\onnx-models"

# Shapes chosen to span the real workload rather than a single convenient case:
#   det - square-ish and wide pages at the configured det_limit_side_len (960) and below
#   rec - fixed height 48 (the model's constraint), a range of widths incl. long callouts
DET_SHAPES = [(1, 3, 640, 640), (1, 3, 960, 960), (1, 3, 480, 1280)]
REC_SHAPES = [(1, 3, 48, 160), (1, 3, 48, 320), (4, 3, 48, 320)]


def paddle_run(model_dir: str, feeds: list, device: str) -> list:
    """Run the shipped paddle inference model on each input array."""
    import paddle
    from paddle import inference as pinf

    cfg = pinf.Config(os.path.join(model_dir, "inference.json"),
                      os.path.join(model_dir, "inference.pdiparams"))
    if device == "gpu":
        cfg.enable_use_gpu(256, 0)
    else:
        cfg.disable_gpu()
        cfg.set_cpu_math_library_num_threads(4)
    cfg.disable_glog_info()
    cfg.enable_memory_optim()
    predictor = pinf.create_predictor(cfg)

    in_names = predictor.get_input_names()
    out_names = predictor.get_output_names()
    handle = predictor.get_input_handle(in_names[0])

    outs, times = [], []
    for arr in feeds:
        def _one():
            handle.reshape(list(arr.shape))
            handle.copy_from_cpu(arr)
            t0 = time.perf_counter()
            predictor.run()
            # MUST fetch inside the timed region. predictor.run() dispatches the GPU work
            # and can return before it completes; copy_to_cpu is what forces the sync.
            # Timing run() alone made det look flat at ~8ms for 640x640, 960x960 AND
            # 480x1280 - i.e. independent of the work, which is the giveaway. ORT's
            # sess.run() already returns host arrays, so this makes the two comparable.
            _ = predictor.get_output_handle(out_names[0]).copy_to_cpu()
            return time.perf_counter() - t0
        _one()                      # warm up: match ORT's treatment exactly
        times.append(min(_one() for _ in range(3)))
        outs.append(predictor.get_output_handle(out_names[0]).copy_to_cpu())
    del predictor
    return outs, times


def onnx_run(onnx_path: str, feeds: list, device: str) -> list:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device == "gpu" else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(onnx_path, so, providers=providers)
    actual = sess.get_providers()
    # ORT falls back to CPU SILENTLY when the CUDA EP cannot load (wrong CUDA major).
    # Assert on the SESSION's providers, never on get_available_providers().
    if device == "gpu" and "CUDAExecutionProvider" not in actual:
        print(f"  !! ORT fell back to CPU (providers={actual}) - timings are not comparable")
    name = sess.get_inputs()[0].name

    outs, times = [], []
    for arr in feeds:
        # WARM UP FIRST. ORT compiles/selects CUDA kernels on the first call for a given
        # shape; without this the first measurement is tens of seconds and the comparison
        # is meaningless. Paddle's predictor is warmed the same way below.
        sess.run(None, {name: arr})
        best = min(_timed(sess, name, arr) for _ in range(3))
        times.append(best)
        outs.append(sess.run(None, {name: arr})[0])
    return outs, times, actual


def _timed(sess, name, arr) -> float:
    t0 = time.perf_counter()
    sess.run(None, {name: arr})
    return time.perf_counter() - t0


def compare(tag: str, a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"tag": tag, "shape_mismatch": (a.shape, b.shape)}
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    diff = np.abs(a64 - b64)
    denom = np.maximum(np.abs(a64), 1e-8)
    return {
        "tag": tag,
        "shape": a.shape,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float((diff / denom).max()),
        "paddle_range": (float(a64.min()), float(a64.max())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("=" * 76)
    print(f"ONNX vs paddle parity  (device={args.device})")
    print("=" * 76)

    overall_ok = True
    for name, sub, shapes in (
        ("det", "PP-OCRv6_medium_det_infer", DET_SHAPES),
        ("rec", "PP-OCRv6_medium_rec_infer", REC_SHAPES),
    ):
        mdir = os.path.join(MODELS, sub)
        opath = os.path.join(ONNX, sub + ".onnx")
        if not os.path.isfile(opath):
            print(f"\n[{name}] MISSING {opath} - run the WSL conversion first")
            overall_ok = False
            continue

        # Same normalised-image-like input for both backends.
        feeds = [rng.standard_normal(s).astype(np.float32) for s in shapes]

        print(f"\n{'-' * 76}\n[{name}] {sub}\n{'-' * 76}")
        try:
            p_out, p_t = paddle_run(mdir, feeds, args.device)
        except BaseException as exc:            # noqa: BLE001
            print(f"  paddle FAILED: {type(exc).__name__}: {exc}")
            print("  (the GPU predictor crashes ~20% of the time - see the investigation "
                  "doc; re-run, or use --device cpu)")
            overall_ok = False
            continue
        o_out, o_t, provs = onnx_run(opath, feeds, args.device)
        print(f"  ORT providers: {provs}")

        for i, s in enumerate(shapes):
            c = compare(str(s), p_out[i], o_out[i])
            if "shape_mismatch" in c:
                print(f"  {s}  SHAPE MISMATCH paddle={c['shape_mismatch'][0]} "
                      f"onnx={c['shape_mismatch'][1]}")
                overall_ok = False
                continue
            # det is a probability map -> tight tolerance is meaningful.
            # rec is (batch, timesteps, 18710) LOGITS: a few elements can differ by
            # ~1e-3 purely from kernel-order differences in a large matmul while every
            # decoded character is identical. Gate rec on argmax (below), not on this.
            tol = 1e-4 if name == "det" else 1e-1
            ok = c["max_abs"] < tol
            overall_ok &= ok
            print(f"  {str(s):<20} out={str(c['shape']):<22} "
                  f"max_abs={c['max_abs']:.3e}  mean_abs={c['mean_abs']:.3e}  "
                  f"{'OK' if ok else 'DIFFERS'}   "
                  f"paddle {p_t[i]*1000:6.1f}ms | onnx {o_t[i]*1000:6.1f}ms")

            # For rec, what actually becomes text is the per-timestep argmax.
            if name == "rec" and p_out[i].ndim == 3:
                pa = p_out[i].argmax(axis=-1)
                oa = o_out[i].argmax(axis=-1)
                same = bool((pa == oa).all())
                frac = float((pa == oa).mean())
                print(f"  {'':<20} argmax sequence identical: {same}  "
                      f"(agreement {frac*100:.2f}%)  <- this is what becomes text")
                overall_ok &= same

    print("\n" + "=" * 76)
    print("VERDICT:", "PARITY OK - ONNX matches the shipped paddle models"
          if overall_ok else "DIFFERENCES FOUND - investigate before trusting the swap")
    print("=" * 76)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
