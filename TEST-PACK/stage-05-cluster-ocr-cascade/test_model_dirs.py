"""The model-directory gates must ask the ACTIVE backend, not the retired one.

WHY THIS EXISTS. Every page of an extraction failed at startup with

    RuntimeError: PaddleOCR local models not found. Required dirs (with
    inference.pdiparams): det=.../PP-OCRv6_medium_det_infer, ...

on a runtime that has no paddlepaddle, no paddleocr and no paddlex installed in it.

The runtime is ONNX-only. _build_ocr_engine passes every *_model_dir through
ocr_backend.onnx_model_dir_for(), which resolves <base>/<name> to
<base>/onnx/<name>/inference.onnx. The paddle-named directory is a NAMING KEY -- it is
never opened. But three gates -- _get_paddleocr_model_dirs, _get_paddleocr_cls_dir and
_handwriting_rec_dir_ok -- still demanded inference.pdiparams beside it before the mapping
ran. So ~600 MB of paddle weights that nothing reads were load-bearing for startup, and
the day they left the tree the whole console stopped extracting.

These tests pin the gate to the backend that does the loading.

    $PY = 'C:\OCULI.pd1\TESTING\.venv-nopaddle\Scripts\python.exe'
    & $PY TESTING\tests\test_model_dirs.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import add_runtime_to_path, check, check_eq, main  # noqa: E402

add_runtime_to_path()

import cluster_ocr as C  # noqa: E402
import ocr_backend as B  # noqa: E402


def test_the_shipped_tree_resolves():
    """The gates pass against the models this runtime actually ships."""
    det, rec = C._get_paddleocr_model_dirs()
    if not check("det and rec dirs resolve", bool(det) and bool(rec)):
        return
    check("det maps to a real inference.onnx",
          os.path.isfile(os.path.join(B.onnx_model_dir_for(det) or "", "inference.onnx")))
    check("rec maps to a real inference.onnx",
          os.path.isfile(os.path.join(B.onnx_model_dir_for(rec) or "", "inference.onnx")))


def test_no_gate_requires_paddle_weights_on_the_onnx_backend():
    """The regression itself: the ONNX tree alone must satisfy every gate.

    Asserted against the SHIPPED tree, which has no inference.pdiparams anywhere under
    models/paddleocr/. If paddle weights are ever reintroduced this still holds; if a gate
    is ever rewritten to demand them again, this fails.
    """
    if not check("backend is onnx", B.is_onnx()):
        return
    for name, d in (("det/rec", C._get_paddleocr_model_dirs()[0]),
                    ("cls", C._get_paddleocr_cls_dir()),
                    ("handwriting", C.PADDLE_HANDWRITING_REC_DIR)):
        if not d:
            check(f"{name} gate resolved", False, "gate returned nothing")
            continue
        check(f"{name} resolves WITHOUT paddle weights beside it",
              B.onnx_model_dir_for(d) is not None and not C._paddle_weights_present(d),
              d)


def test_an_onnx_only_directory_is_accepted():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "SOME_MODEL"
        d.mkdir()
        check("a bare directory is rejected", not C._model_dir_ok(str(d)))
        (d / "inference.onnx").write_bytes(b"not a real graph, but present")
        check("inference.onnx beside the name is enough", C._model_dir_ok(str(d)))


def test_the_parallel_onnx_tree_is_accepted():
    """The shipped layout: models/paddleocr/<name>/ does not exist at all."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        (base / "onnx" / "SOME_MODEL").mkdir(parents=True)
        (base / "onnx" / "SOME_MODEL" / "inference.onnx").write_bytes(b"x")
        missing = base / "SOME_MODEL"
        check("the paddle-named directory does not need to exist", not missing.exists())
        check("...and the gate still passes via the parallel onnx tree",
              C._model_dir_ok(str(missing)))


def test_a_genuinely_absent_model_is_still_refused():
    """Degrading gracefully must not become accepting anything."""
    with tempfile.TemporaryDirectory() as td:
        check("a directory with no model in it is refused",
              not C._model_dir_ok(os.path.join(td, "NOTHING_HERE")))
    check("empty input is refused", not C._model_dir_ok(""))
    check("None is refused", not C._model_dir_ok(None))


def test_paddle_weights_probe_is_honest():
    """_paddle_weights_present is what the retired backend would have asked."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "inference.json").write_bytes(b"{}")
        check("a graph alone is not weights", not C._paddle_weights_present(str(d)))
        (d / "inference.pdiparams").write_bytes(b"x")
        check("graph + params is weights", C._paddle_weights_present(str(d)))


if __name__ == "__main__":
    sys.exit(main(sys.modules[__name__]))
