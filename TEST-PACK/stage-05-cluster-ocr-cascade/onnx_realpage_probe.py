"""End-to-end validation on REAL pages: paddle backend vs ONNX backend.

DIAGNOSTIC ONLY. Release models/images are read read-only; ONNX model dirs are built
under TESTING/, never inside the release tree.

WHY THIS AND NOT THE TENSOR TEST
--------------------------------
The tensor probe compared raw det/rec outputs and passed. This runs the full paddlex
PREDICTOR for each model - preprocessing, DB post-processing, box assembly, CTC decode -
so it validates what actually ships rather than the layer underneath.

It targets `paddlex.inference.models.create_predictor(engine=...)`, which is the exact
seam a runner swap would use: predictors own the transforms, engines own only the
inference call. (PaddleOCR's own wrapper exposes no engine kwarg, so it is bypassed;
the predictors it builds internally are these.)

Three environment steps are required before any of this works, and each was discovered
the hard way here - see the comments on each:
  1. register_cuda_dll_dirs()          - or ORT silently runs on CPU
  2. stub_modelscope()                 - or paddlex drags torch in and paddle's cuDNN dies
  3. preload_cudnn_in_dependency_order() - or paddle's lazy cuDNN load fails WinError 127

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-release\\Scripts\\python.exe'
    & $PY onnx_realpage_probe.py
"""
from __future__ import annotations

import ctypes
import glob
import os
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ort_gpu_guard import register_cuda_dll_dirs  # noqa: E402

# (1) MUST precede `import onnxruntime`. onnxruntime-gpu does not bundle the CUDA DLLs it
# links; without these dirs the CUDA EP fails to load and ORT silently uses CPU. Note
# include_torch_lib stays False: adding torch\lib breaks paddle's cuDNN resolution.
register_cuda_dll_dirs()

import numpy as np  # noqa: E402

BK = r"C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main"
PADDLE_MODELS = os.path.join(BK, "models", "paddleocr")
ONNX_SRC = r"C:\OCULI.pd1\TESTING\onnx-models"
ONNX_DIRS = r"C:\OCULI.pd1\TESTING\onnx-model-dirs"
IMAGES = [os.path.join(BK, "test_data", "01_original.png"),
          os.path.join(BK, "test_data", "Screenshot_21-7-2026_101813_.jpeg")]

DET, REC = "PP-OCRv6_medium_det_infer", "PP-OCRv6_medium_rec_infer"
DET_NAME, REC_NAME = "PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"


def stub_modelscope() -> None:
    """(2) Mirror paddle_ipc._server_init. paddlex imports modelscope, which imports TORCH;
    once torch is loaded, torch\\lib is on the DLL search path and paddle's lazy cuDNN load
    resolves cudnn_engines_precompiled64_9.dll against TORCH's copy and dies WinError 127.
    Observed here verbatim before this stub was added."""
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


def preload_cudnn_in_dependency_order() -> int:
    """(3) Mirror _server_init's ordered preload. add_dll_directory alone is not enough:
    paddle loads cudnn_engines_precompiled64_9.dll LAZILY during GPU model creation and
    needs cudnn64_9 / cudnn_graph64_9 / cudart / cublas already resolved, base-first."""
    if os.name != "nt":
        return 0
    root = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    n = 0

    def _pre(pkg, names):
        nonlocal n
        for bn in names:
            for d in glob.glob(os.path.join(root, pkg, "bin")):
                p = os.path.join(d, bn)
                if os.path.isfile(p):
                    try:
                        ctypes.CDLL(p)
                        n += 1
                    except Exception:
                        pass
                    break

    _pre("cuda_runtime", ["cudart64_12.dll"])
    _pre("cublas", ["cublas64_12.dll", "cublasLt64_12.dll"])
    _pre("cudnn", ["cudnn_graph64_9.dll", "cudnn64_9.dll", "cudnn_ops64_9.dll",
                   "cudnn_cnn64_9.dll", "cudnn_adv64_9.dll", "cudnn_heuristic64_9.dll",
                   "cudnn_engines_runtime_compiled64_9.dll",
                   "cudnn_engines_precompiled64_9.dll"])
    return n


def build_onnx_model_dirs() -> tuple:
    """paddlex resolves an engine's model file as <MODEL_FILE_PREFIX>.onnx == inference.onnx.
    inference.yml is copied across because it carries the preprocessing config + charset."""
    out = {}
    for sub in (DET, REC):
        src = os.path.join(ONNX_SRC, sub + ".onnx")
        if not os.path.isfile(src):
            raise SystemExit(f"missing {src} - run the WSL conversion first")
        d = os.path.join(ONNX_DIRS, sub)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(src, os.path.join(d, "inference.onnx"))
        yml = os.path.join(PADDLE_MODELS, sub, "inference.yml")
        if os.path.isfile(yml):
            shutil.copy2(yml, os.path.join(d, "inference.yml"))
        out[sub] = d
    return out[DET], out[REC]


def make_predictor(model_name, model_dir, engine, extra=None):
    from paddlex.inference.models import create_predictor
    kw = dict(model_name=model_name, model_dir=model_dir, device="gpu:0")
    if engine:
        kw["engine"] = engine
        if engine == "onnxruntime":
            kw["engine_config"] = {"providers": ["CUDAExecutionProvider"]}
    kw.update(extra or {})
    t0 = time.perf_counter()
    p = create_predictor(**kw)
    return p, time.perf_counter() - t0


def run(pred, item, n=1):
    t0 = time.perf_counter()
    for _ in range(n):
        res = list(pred([item] if not isinstance(item, list) else item))
    return res, (time.perf_counter() - t0) / n


def polys_of(res):
    r = res[0]
    d = r if isinstance(r, dict) else getattr(r, "json", {}).get("res", r)
    for k in ("dt_polys", "polys", "boxes"):
        if k in d:
            return [np.asarray(b, dtype=np.float64).reshape(-1, 2) for b in d[k]]
    return []


def centre(b):
    return b.mean(axis=0)


def match(a, b, tol=10.0):
    used, pairs = set(), []
    for i, x in enumerate(a):
        cx = centre(x)
        best, bd = None, 1e18
        for j, y in enumerate(b):
            if j in used:
                continue
            dd = float(np.hypot(*(centre(y) - cx)))
            if dd < bd:
                best, bd = j, dd
        if best is not None and bd <= tol:
            used.add(best)
            pairs.append((i, best, bd))
    return pairs


def main() -> int:
    det_o, rec_o = build_onnx_model_dirs()
    stub_modelscope()
    n = preload_cudnn_in_dependency_order()
    print(f"env: preloaded {n} CUDA/cuDNN DLLs; modelscope stubbed; ORT DLL dirs registered")

    imgs = [p for p in IMAGES if os.path.isfile(p)]
    if not imgs:
        print("no test images"); return 2
    print(f"images: {[os.path.basename(p) for p in imgs]}\n")

    print("building det predictors")
    try:
        det_p, tp = make_predictor(DET_NAME, os.path.join(PADDLE_MODELS, DET), None)
        print(f"  paddle det  {tp:5.1f}s")
    except BaseException as exc:                        # noqa: BLE001
        print(f"  paddle det FAILED: {type(exc).__name__}: {str(exc)[:160]}")
        print("  (phi.dll access violation hits ~20% of GPU starts - re-run)")
        return 1
    try:
        det_x, tx = make_predictor(DET_NAME, det_o, "onnxruntime")
        print(f"  onnx   det  {tx:5.1f}s")
    except BaseException as exc:                        # noqa: BLE001
        print(f"  onnx det FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        return 1

    ok_all = True
    import cv2
    for img in imgs:
        print(f"\n{'=' * 76}\n{os.path.basename(img)}\n{'=' * 76}")
        rp, dtp = run(det_p, img)
        rx, dtx = run(det_x, img)
        bp, bx = polys_of(rp), polys_of(rx)
        print(f"  det boxes  paddle {len(bp):4d} ({dtp:5.2f}s)   "
              f"onnx {len(bx):4d} ({dtx:5.2f}s, {dtx/max(dtp,1e-9):.2f}x)")
        pairs = match(bp, bx)
        cov = len(pairs) / max(len(bp), len(bx), 1) * 100
        if pairs:
            offs = [d for _, _, d in pairs]
            print(f"  matched {len(pairs)} ({cov:.1f}%)  centre offset "
                  f"mean {sum(offs)/len(offs):.2f}px  max {max(offs):.2f}px")
        else:
            print("  NO boxes matched")
        box_ok = (abs(len(bp) - len(bx)) <= max(2, 0.02 * len(bp))) and cov >= 95.0
        ok_all &= box_ok
        print(f"  box parity: {'OK' if box_ok else 'DIFFERS'}")

    # ---- rec: the part that actually becomes text -----------------------------
    print(f"\n{'=' * 76}\nrecognition on REAL crops\n{'=' * 76}")
    try:
        rec_p, tp = make_predictor(REC_NAME, os.path.join(PADDLE_MODELS, REC), None)
        rec_x, tx = make_predictor(REC_NAME, rec_o, "onnxruntime")
        print(f"  paddle rec {tp:5.1f}s   onnx rec {tx:5.1f}s")
    except BaseException as exc:                        # noqa: BLE001
        print(f"  rec predictor FAILED: {type(exc).__name__}: {str(exc)[:200]}")
        return 1

    for img in imgs:
        im = cv2.imread(img)
        if im is None:
            continue
        boxes = polys_of(run(det_p, img)[0])
        crops = []
        for b in boxes:
            x1, y1 = np.floor(b.min(axis=0)).astype(int)
            x2, y2 = np.ceil(b.max(axis=0)).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(im.shape[1], x2), min(im.shape[0], y2)
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                crops.append(im[y1:y2, x1:x2])
        if not crops:
            continue

        t0 = time.perf_counter(); rp = list(rec_p(crops)); dtp = time.perf_counter() - t0
        t0 = time.perf_counter(); rx = list(rec_x(crops)); dtx = time.perf_counter() - t0

        def texts(res):
            out = []
            for r in res:
                d = r if isinstance(r, dict) else getattr(r, "json", {}).get("res", r)
                out.append(d.get("rec_text", d.get("text", "")))
            return out

        def scores(res):
            out = []
            for r in res:
                d = r if isinstance(r, dict) else getattr(r, "json", {}).get("res", r)
                out.append(float(d.get("rec_score", d.get("score", 0.0)) or 0.0))
            return out

        tp_, tx_ = texts(rp), texts(rx)
        sp_, sx_ = scores(rp), scores(rx)
        n = min(len(tp_), len(tx_))
        same = sum(1 for i in range(n) if tp_[i] == tx_[i])
        pct = same / n * 100 if n else 0.0
        print(f"\n  {os.path.basename(img)}: {n} crops   "
              f"paddle {dtp:5.2f}s | onnx {dtx:5.2f}s ({dtx/max(dtp,1e-9):.2f}x)")
        print(f"  IDENTICAL TEXT: {same}/{n}  ({pct:.2f}%)")
        if sp_ and sx_:
            ds = [abs(sp_[i] - sx_[i]) for i in range(n)]
            print(f"  confidence delta: mean {sum(ds)/len(ds):.5f} max {max(ds):.5f}")
        diffs = [(i, tp_[i], tx_[i]) for i in range(n) if tp_[i] != tx_[i]][:8]
        for i, a, b in diffs:
            print(f"    crop {i}: paddle={a!r}  onnx={b!r}")
        ok_all &= (same == n)

    print(f"\n{'=' * 76}")
    print("VERDICT:", "REAL-PAGE PARITY OK (det boxes + rec text)" if ok_all
          else "DIFFERENCES on real pages - inspect above")
    print("=" * 76)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
