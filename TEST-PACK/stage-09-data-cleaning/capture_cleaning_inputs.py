"""Capture the REAL DataCleaningInputsV2 a page hands to the classifier.

WHY. Every LLM test so far fed the model `unmatched_text` from the FINISHED document --
post-consolidation residue. That is not where Placement 1 sits. Data cleaning receives raw
geometry, raw text and raw signals: cluster_lines, rectangles, contours, line_segments,
unclosed_contours, and the per-cluster/block/line feature dicts. Testing the model on the
refined leftovers under-represents what it would actually see.

This intercepts FormatClassifier.classify, serialises its inputs, then calls through so the
run completes normally.

    $PY = 'C:\OCULI.pd1\TESTING\.venv-nopaddle\Scripts\python.exe'
    & $PY TESTING\RESOURCES\capture_cleaning_inputs.py <input_dir> <out.json>
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import fields, is_dataclass

BK = r"C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main"
for p in (BK, os.path.join(BK, "workers"), os.path.join(BK, "input"),
          os.path.join(BK, "cleaners")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = sys.argv[2] if len(sys.argv) > 2 else r"C:\OCULI.pd1\TESTING\RESULTS\cleaning_inputs.json"
INPUT = sys.argv[1]
_captured = []


def _plain(o, depth=0):
    """Serialise dataclasses / tuples / dicts far enough to inspect, without exploding."""
    if depth > 4:
        return "..."
    if is_dataclass(o):
        return {f.name: _plain(getattr(o, f.name, None), depth + 1) for f in fields(o)}
    if isinstance(o, dict):
        return {str(k): _plain(v, depth + 1) for k, v in list(o.items())[:5000]}
    if isinstance(o, (list, tuple)):
        return [_plain(v, depth + 1) for v in list(o)[:40000]]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)[:200]


def _install():
    """Patch the name BOUND IN pipeline's namespace.

    pipeline.py does `from data_cleaningv2 import classify_document_structure`, so the
    function object is bound into pipeline's globals at import time. Patching
    data_cleaningv2 (or cleaners.data_cleaningv2, which is a DIFFERENT module object
    because cleaners/ is itself on sys.path) leaves pipeline's reference untouched --
    which is why the first attempt captured nothing while classification demonstrably ran.
    """
    import pipeline as PL

    original_fn = PL.classify_document_structure

    def patched_free(inputs):
        _capture(inputs)
        return original_fn(inputs)

    PL.classify_document_structure = patched_free

    def _capture(inputs):
        try:
            page_id = str(getattr(getattr(inputs, "page", None), "page_id", "") or "")
            _captured.append({
                "page_id": page_id,
                "counts": {
                    "cluster_lines": len(getattr(inputs, "cluster_lines", []) or []),
                    "cluster_lines_raw": len(getattr(inputs, "cluster_lines_raw", []) or []),
                    "region_lines": len(getattr(inputs, "region_lines", []) or []),
                    "region_blocks": len(getattr(inputs, "region_blocks", []) or []),
                    "rectangles": len(getattr(inputs, "rectangles", []) or []),
                    "contours": len(getattr(inputs, "contours", []) or []),
                    "line_segments": len(getattr(inputs, "line_segments", []) or []),
                    "unclosed_contours": len(getattr(inputs, "unclosed_contours", []) or []),
                    "cluster_blocks": len(getattr(inputs, "cluster_blocks", []) or []),
                },
                "page": _plain(getattr(inputs, "page", None)),
                "cluster_lines": _plain(getattr(inputs, "cluster_lines", [])),
                "rectangles": _plain(getattr(inputs, "rectangles", [])),
                "cluster_features": _plain(getattr(inputs, "cluster_features", {})),
                "region_blocks": _plain(getattr(inputs, "region_blocks", [])),
                "region_block_features": _plain(getattr(inputs, "region_block_features", {})),
                "line_segments": _plain(getattr(inputs, "line_segments", [])),
            })
        except Exception as exc:  # capture must never break the run
            _captured.append({"capture_error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    _install()
    import session_runtime  # noqa: F401  (imports the pipeline)
    sys.argv = ["session_runtime.py", "extract", "--input-path", INPUT,
                "--run-id", "capture_inputs"]
    try:
        session_runtime.main()
    except SystemExit:
        pass
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_captured, f, indent=1)
    print(f"\nCAPTURED {len(_captured)} page(s) -> {OUT}")
    for c in _captured:
        print("  ", c.get("page_id"), c.get("counts"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
