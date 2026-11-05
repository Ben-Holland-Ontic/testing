"""Reusable guard for onnxruntime-gpu: make the CUDA EP load, and REFUSE to run on CPU.

This module exists because of two failure modes measured on the release stack, both of
which produce a silently CPU-bound pipeline that still looks healthy:

FAILURE 1 - wrong CUDA major.
    onnxruntime-gpu 1.29.0 requires CUDA 13. The release stack is CUDA 12.9. Symptom:

        Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 13.*
        session providers: ['CPUExecutionProvider']

    Pin the CUDA 12 build: onnxruntime-gpu==1.22.0.

FAILURE 2 - CUDA DLLs not on the search path.
    onnxruntime_providers_cuda.dll depends on cublasLt64_12.dll, cudart, cudnn etc., and
    onnxruntime-gpu does NOT bundle them. In a process that imported torch first they
    resolve by accident, because torch\\lib contains them. In a process that does not
    (the paddle-style OCR subprocess, or any tool run standalone) the EP fails to load:

        Error loading "...onnxruntime_providers_cuda.dll" which depends on
        "cublasLt64_12.dll" which is missing. (Error 126)

    The release already ships the nvidia-*-cu12 wheels, so the fix is to register their
    bin directories the same way pipeline._register_nvidia_dll_dirs() already does for
    paddle. Relying on torch having been imported first is not a design, it is luck.

IN BOTH CASES `get_available_providers()` STILL LISTS CUDAExecutionProvider. It reports
what the build supports, not what loaded. Only the SESSION's get_providers() tells the
truth, which is why check_session() below asserts on that.

Usage in any ONNX-backed runner:

    from ort_gpu_guard import register_cuda_dll_dirs, make_gpu_session
    register_cuda_dll_dirs()                      # before importing onnxruntime
    sess = make_gpu_session("model.onnx")         # raises if it would run on CPU
"""
from __future__ import annotations

import glob
import os
import sys

__all__ = ["register_cuda_dll_dirs", "make_gpu_session", "check_session",
           "OrtGpuUnavailable"]


class OrtGpuUnavailable(RuntimeError):
    """Raised when a session that must run on the GPU did not bind the CUDA EP."""


def register_cuda_dll_dirs(verbose: bool = False, include_torch_lib: bool = False) -> list:
    """Put the nvidia wheel bin dirs (and torch's lib, if present) on the DLL search path.

    Must run BEFORE `import onnxruntime`, because the provider DLL is resolved at first
    session creation and Windows caches the search path decisions made at load time.
    Mirrors pipeline._register_nvidia_dll_dirs(); safe to call more than once.
    """
    added = []
    if os.name != "nt":
        return added
    roots = []
    try:
        import site
        roots.extend(site.getsitepackages() or [])
    except Exception:
        pass
    roots.append(os.path.join(sys.prefix, "Lib", "site-packages"))

    seen = set()
    for sp in roots:
        nvidia = os.path.join(sp, "nvidia")
        if os.path.isdir(nvidia):
            for pkg in os.listdir(nvidia):
                for sub in ("bin", "lib"):
                    d = os.path.join(nvidia, pkg, sub)
                    if os.path.isdir(d) and d not in seen:
                        seen.add(d)
                        try:
                            os.add_dll_directory(d)
                            added.append(d)
                        except Exception:
                            pass
        # torch/lib also carries cublasLt/cudart/cudnn, but registering it is HARMFUL by
        # default and off unless explicitly requested. Measured: adding torch\lib to the
        # DLL search path makes a later `import paddle` resolve cuDNN against TORCH's copy
        # and die with the familiar
        #   WinError 127 ... nvidia\cudnn\bin\cudnn_engines_precompiled64_9.dll
        # even in a process where torch was never imported. That is the same ABI clash
        # `paddle_ipc._build_child_env` guards against by stripping torch from the child's
        # PATH. During a phased migration both engines may live in one process tree, so
        # this must stay opt-in.
        if include_torch_lib:
            tl = os.path.join(sp, "torch", "lib")
            if os.path.isdir(tl) and tl not in seen:
                seen.add(tl)
                try:
                    os.add_dll_directory(tl)
                    added.append(tl)
                except Exception:
                    pass
    if verbose:
        print(f"[ort-guard] registered {len(added)} CUDA DLL dir(s)")
        for d in added:
            print(f"[ort-guard]   {d}")
    return added


def check_session(sess, model: str = "") -> None:
    """Raise unless the session actually bound a GPU execution provider."""
    provs = sess.get_providers()
    gpu = [p for p in provs if p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")]
    if not gpu:
        import onnxruntime as ort
        raise OrtGpuUnavailable(
            f"onnxruntime session for {model or '<model>'} bound {provs} - it would run "
            f"on CPU.\n"
            f"  onnxruntime version : {ort.__version__}\n"
            f"  build advertises    : {ort.get_available_providers()}\n"
            f"  (that list is what the BUILD supports, not what LOADED - do not trust it)\n"
            f"Most likely: onnxruntime-gpu built for the wrong CUDA major (pin 1.22.0 for "
            f"CUDA 12), or the nvidia CUDA DLLs are not on the search path - call "
            f"register_cuda_dll_dirs() BEFORE importing onnxruntime."
        )


def make_gpu_session(model_path: str, cuda_options: dict | None = None,
                     enable_trt: bool = False, graph_optimize_all: bool = True):
    """Create an onnxruntime session that is guaranteed to be on the GPU, or raise."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    if graph_optimize_all:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = []
    if enable_trt:
        providers.append(("TensorrtExecutionProvider", {"trt_fp16_enable": True}))
    providers.append(("CUDAExecutionProvider", cuda_options or {}))
    providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(model_path, so, providers=providers)
    check_session(sess, os.path.basename(model_path))
    return sess


if __name__ == "__main__":
    # Self-test: prove the guard both fixes the load and catches the CPU fallback.
    print("registering CUDA DLL dirs...")
    dirs = register_cuda_dll_dirs(verbose=True)
    import onnxruntime as ort
    print(f"\nonnxruntime {ort.__version__}")
    print(f"build advertises: {ort.get_available_providers()}")
    m = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\OCULI.pd1\TESTING\onnx-models\PP-OCRv6_medium_det_infer.onnx"
    if os.path.isfile(m):
        try:
            s = make_gpu_session(m)
            print(f"\nSession bound: {s.get_providers()}  -> GPU confirmed")
        except OrtGpuUnavailable as exc:
            print(f"\nGUARD TRIPPED (correctly):\n{exc}")
    else:
        print(f"\n(model {m} not found; guard not exercised)")
