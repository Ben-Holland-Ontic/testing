"""Verify a packaged OCULI release installs and runs from a CLEAN location.

Run this against an EXTRACTED release tree, never against the working tree -- the whole
point is to catch things that only work because the dev machine happens to have them.

    $PY = 'C:\OCULI.pd1\TESTING\.venv-nopaddle\Scripts\python.exe'
    & $PY TESTING\RESOURCES\verify_fresh_install.py <install_root> [--input <dir>]

<install_root> is the directory holding bk-main\ and python\ (the runtime zip's layout).

Checks, in the order they would bite a real user:

  1  LAYOUT      bk-main\session_runtime.py and python\python.exe are siblings; the
                 LayoutML sidecar sits beside bk-main (session_runtime imports
                 synthetic_page_generator by bare name -- Classify/Verify dies without it).
  2  NO JUNK     no test_data, test_pipeline.py, output diagnostics, debug logs or
                 __pycache__ reached the payload.
  3  WHEELS      every '==' pin in requirements.txt resolves to a local wheel, and the
                 pip/setuptools/wheel bootstrap trio is vendored. The endpoint installs
                 with --no-index, so a missing wheel is an install-time failure on a
                 machine that may have no internet.
  4  MODELS      every model the OCR stack loads is present, by path.
  5  IMPORTS     the embedded interpreter can import the runtime modules.
  6  GPU         onnxruntime binds a GPU provider -- NOT a silent CPU fallback. This is
                 the failure that costs ~37x and reports nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

FAIL: list[str] = []
WARN: list[str] = []


def ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def bad(msg: str) -> None:
    FAIL.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    WARN.append(msg)
    print(f"  WARN  {msg}")


def check_layout(root: Path) -> tuple[Path, Path]:
    print("\n[1] LAYOUT")
    bk = root / "bk-main"
    py = root / "python" / "python.exe"
    sidecar = root / "LayoutML sidecar"
    (ok if (bk / "session_runtime.py").is_file() else bad)("bk-main/session_runtime.py")
    (ok if py.is_file() else bad)("python/python.exe (sibling of bk-main)")
    if (sidecar / "synthetic_page_generator.py").is_file():
        ok("LayoutML sidecar beside bk-main (Classify/Verify needs it)")
    else:
        bad("LayoutML sidecar MISSING -- build_annotations/finalize will fail at import")
    for rel in ("config/task_registry.json", "config/query_set.yaml",
                "workers/oculi_ocr/pipeline.py", "queue_manager.py", "pipeline.py"):
        (ok if (bk / rel).exists() else bad)(rel)
    return bk, py


def check_no_junk(bk: Path) -> None:
    print("\n[2] NO DEV PAYLOAD")
    for rel in ("test_data", "test_pipeline.py"):
        (bad if (bk / rel).exists() else ok)(f"{rel} absent")
    logs = list(bk.glob("debug-*.log")) + list(bk.glob("debug_*.jsonl"))
    (bad if logs else ok)(f"no debug logs ({len(logs)} found)" if logs else "no debug logs")
    pyc = list(bk.rglob("__pycache__"))
    (bad if pyc else ok)(f"no __pycache__ ({len(pyc)} found)" if pyc else "no __pycache__")
    out = bk / "output"
    if out.exists() and any(out.rglob("*")):
        warn(f"output/ ships with content ({sum(1 for _ in out.rglob('*'))} entries)")
    else:
        ok("output/ empty or absent")


def _norm(s: str) -> str:
    out = []
    for ch in s.lower():
        out.append("_" if ch in "-_." else ch)
    return "".join(out)


def check_wheels(bk: Path) -> None:
    print("\n[3] OFFLINE WHEELS")
    wd = bk / "wheels"
    if not wd.is_dir():
        bad("wheels/ MISSING -- --no-index install cannot work")
        return
    whls = list(wd.glob("*.whl"))
    if not whls:
        bad("wheels/ contains no .whl")
        return
    ok(f"{len(whls)} wheels, {sum(w.stat().st_size for w in whls) / 1048576:.0f} MB")
    # EXACT distribution match. A wheel is {distribution}-{version}-...whl, so compare only
    # the segment before the first '-'. Prefix matching gave a false pass:
    # 'tensorrt_cu12_bindings-...whl' appeared to satisfy the 'tensorrt-cu12' pin while the
    # real artifact was an 18 KB sdist that cannot install offline.
    dists = {_norm(w.name.split("-")[0]) for w in whls}
    sdists = {_norm(p.name.split("-")[0])
              for p in list(wd.glob("*.tar.gz")) + list(wd.glob("*.zip"))}
    names = [_norm(w.name) for w in whls]
    for boot in ("pip", "setuptools", "wheel"):
        (ok if boot in dists else bad)(f"bootstrap wheel: {boot}")
    req = bk / "requirements.txt"
    if not req.is_file():
        bad("requirements.txt missing")
        return
    missing, sdist_only, pins = [], [], 0
    for line in req.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#") or "==" not in t:
            continue
        pins += 1
        raw = t.split("==")[0].strip()
        pkg = _norm(raw)
        if pkg in dists:
            continue
        (sdist_only if pkg in sdists else missing).append(raw)
    # Source dists VERIFIED to build offline from the bundled wheels. Must match the
    # $sdistAllowed list in Package-OculiGitHubRelease.ps1.
    #   tensorrt-cu12 -- 17 KB pure-Python meta-package. Its setup.py would shell out to
    #   pypi.nvidia.com, but with NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=1 it declares
    #   bindings + libs as ordinary deps, both vendored. Verified: installs --no-index and
    #   `import tensorrt` returns 10.9.0.34.
    SDIST_OK = {"tensorrt-cu12", "langdetect"}
    sdist_bad = [s for s in sdist_only if s.lower() not in SDIST_OK]
    if missing:
        bad(f"{len(missing)} of {pins} pins have NO local wheel: {', '.join(missing[:12])}")
    if sdist_bad:
        bad(f"{len(sdist_bad)} pin(s) are SOURCE-ONLY (need network/toolchain to build, "
            f"so they break --no-index install): {', '.join(sdist_bad)}")
    if sdist_only and not sdist_bad:
        ok(f"{len(sdist_only)} allowed source dist(s): {', '.join(sdist_only)} (build offline)")
    if not missing and not sdist_bad:
        ok(f"all {pins} pinned requirements resolve locally")
    # onnxruntime-gpu needs the CUDA libs even though they are not pinned
    if any(n.startswith("onnxruntime_gpu_") for n in names):
        need = ["nvidia_cublas", "nvidia_cudnn", "nvidia_cufft", "nvidia_curand",
                "nvidia_cusolver", "nvidia_cusparse", "nvidia_nvjitlink",
                "nvidia_cuda_runtime"]
        absent = [p for p in need if not any(n.startswith(p) for n in names)]
        if absent:
            bad("onnxruntime-gpu present but CUDA libs missing: " + ", ".join(absent) +
                " -- the CUDA provider will SILENTLY fall back to CPU")
        else:
            ok("all 8 nvidia CUDA wheels present alongside onnxruntime-gpu")


def check_models(bk: Path) -> None:
    print("\n[4] MODELS")
    for rel in (
        "models/paddleocr/onnx/PP-OCRv6_medium_det_infer/inference.onnx",
        "models/paddleocr/onnx/PP-OCRv6_medium_rec_infer/inference.onnx",
        "models/paddleocr/onnx/PP-LCNet_x0_25_textline/inference.onnx",
        "models/paddleocr/onnx/Handwritting/inference.onnx",
        "models/supporting/parseq/parseq.onnx",
        "models/supporting/catmus/catmus.onnx",
        "models/supporting/sar_resnet31/sar_resnet31.onnx",
        "models/supporting/master/master.onnx",
        "models/doctr/models",
    ):
        (ok if (bk / rel).exists() else bad)(rel)


def check_imports_and_gpu(bk: Path, py: Path) -> None:
    print("\n[5] IMPORTS (embedded interpreter)")
    if not py.is_file():
        bad("no python.exe -- skipping import and GPU checks")
        return
    # Written to a temp FILE rather than passed with -c: escaping Windows paths through a
    # -c string silently produced a broken sys.path, which made onnxruntime resolve
    # elsewhere and report the wrong provider list. A file has no escaping to get wrong.
    probe = bk / "_verify_probe.py"
    probe.write_text(
        "import sys, os, json\n"
        # Set INSIDE the probe, not via the parent's env. Passing this script a copy of a
        # git-bash os.environ mangles PATH into /c/... form, which breaks Windows DLL
        # resolution -- onnxruntime then silently drops its CUDA provider and reports only
        # ['AzureExecutionProvider','CPUExecutionProvider'], which looks exactly like a
        # broken install but is an artifact of the harness.
        "os.environ['PADDLEOCR_USE_GPU'] = '1'\n"
        "sys.dont_write_bytecode = True\n"
        f"BK = {str(bk)!r}\n"
        "sys.path.insert(0, BK)\n"
        "sys.path.insert(0, os.path.join(BK, 'workers'))\n"
        "r = {}\n"
        "for m in ('ocr_backend','ocr_supporting','ocr_cascade','ocr_voting',"
        "'ocr_credibility','cluster_ocr','region_ocr'):\n"
        "    try:\n        __import__(m); r[m] = 'ok'\n"
        "    except Exception as e:\n        r[m] = type(e).__name__ + ': ' + str(e)[:90]\n"
        "try:\n"
        "    import ocr_backend as B\n"
        "    B.register_cuda_dll_dirs()\n"
        "    import onnxruntime as ort\n"
        "    r['_ort'] = ort.__version__\n"
        "    r['_avail'] = ort.get_available_providers()\n"
        "    r['_trt'] = B.trt_available()\n"
        "    r['_bound'] = B.gpu_available()\n"
        "except Exception as e:\n"
        "    r['_ort'] = type(e).__name__ + ': ' + str(e)[:90]\n"
        "print('@@' + json.dumps(r))\n",
        encoding="utf-8")
    try:
        # PYTHONNOUSERSITE is load-bearing. The bootstrapper appends "import site" to
        # python311._pth so pip-installed packages are importable -- but that also enables
        # the USER site directory, so a machine with its own Python 3.11 can shadow the
        # bundled packages. MEASURED here: without it the embedded interpreter imported
        # onnxruntime from the Store-Python user site (CPU build, providers
        # ['AzureExecutionProvider','CPUExecutionProvider']) instead of its own
        # GPU build -- a silent CPU fallback caused purely by what else is on the machine.
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"
        p = subprocess.run([str(py), str(probe)], capture_output=True, text=True,
                           timeout=300, cwd=str(bk), env=env)
    except Exception as exc:
        bad(f"interpreter failed to run: {exc}")
        return
    finally:
        probe.unlink(missing_ok=True)
        # Importing the runtime writes bytecode caches into the payload; this verifier
        # must not leave the tree dirtier than it found it.
        for pc in bk.rglob("__pycache__"):
            for f in pc.glob("*"):
                f.unlink(missing_ok=True)
            pc.rmdir()
    line = next((l for l in p.stdout.splitlines() if l.startswith("@@")), None)
    if not line:
        bad("no result from interpreter")
        print((p.stderr or p.stdout)[-700:])
        return
    r = json.loads(line[2:])
    for m in ("ocr_backend", "ocr_supporting", "ocr_cascade", "ocr_voting",
              "ocr_credibility", "cluster_ocr", "region_ocr"):
        (ok if r.get(m) == "ok" else bad)(f"import {m}" + ("" if r.get(m) == "ok" else f" -> {r.get(m)}"))
    print("\n[6] GPU BINDING (silent CPU fallback is the failure this catches)")
    print(f"       onnxruntime {r.get('_ort')}  (PADDLEOCR_USE_GPU=1)")
    print(f"       available   {r.get('_avail')}")
    if r.get("_bound") is True:
        ok("a real session BINDS a GPU provider")
    else:
        bad(f"gpu_available()={r.get('_bound')} -- OCR would run on CPU (~37x slower), silently")
    if r.get("_trt") is True:
        ok("TensorRT usable -- on sm_120 this is the difference between 1.75s and 65.6s")
    else:
        warn(f"trt_available()={r.get('_trt')} -- falls back to the CUDA EP (much slower on Blackwell)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("install_root")
    args = ap.parse_args()
    root = Path(args.install_root)
    print(f"Verifying fresh install at: {root}")
    if not root.is_dir():
        print("  install root does not exist")
        return 2
    bk, py = check_layout(root)
    check_no_junk(bk)
    check_wheels(bk)
    check_models(bk)
    check_imports_and_gpu(bk, py)
    print("\n" + "=" * 62)
    if FAIL:
        print(f"FAILED: {len(FAIL)} check(s)")
        for f in FAIL:
            print(f"   - {f}")
    else:
        print("ALL CHECKS PASSED")
    if WARN:
        print(f"{len(WARN)} warning(s):")
        for w in WARN:
            print(f"   - {w}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
