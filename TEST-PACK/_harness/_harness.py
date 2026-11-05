"""Minimal test runner for the OCR unit tests.

There is no pytest anywhere in this project and the release venv must not grow a
test dependency, so these follow the convention the existing probes already use:
plain functions, a `check()` assertion helper, a non-zero exit code on failure.

A test module defines `test_*` functions taking no arguments and calls `check()`.
Run one with:

    $PY = 'C:\OCULI.pd1\TESTING\.venv-nopaddle\Scripts\python.exe'
    & $PY tests\test_phase0_routing.py

Run every module:

    & $PY tests\run_all.py
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Callable, List

BK = r"C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main"


def add_runtime_to_path() -> None:
    """Import the runtime the way pipeline.py does, so module names match."""
    for p in (BK, os.path.join(BK, "workers"), os.path.join(BK, "input"),
              os.path.join(BK, "cleaners")):
        if p not in sys.path:
            sys.path.insert(0, p)


class _State:
    failures: List[str] = []
    current: str = ""


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS  " if ok else "FAIL  "
    line = f"    {tag}{name}"
    if detail:
        line += f"   [{detail}]"
    print(line)
    if not ok:
        _State.failures.append(f"{_State.current}::{name}")
    return bool(ok)


def check_eq(name: str, got: Any, want: Any) -> bool:
    ok = got == want
    detail = "" if ok else f"got={got!r} want={want!r}"
    return check(name, ok, detail)


def run_module(mod) -> int:
    """Run every test_* function in `mod`. Returns the number of failures."""
    tests: List[Callable[[], None]] = [
        getattr(mod, n) for n in sorted(dir(mod))
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
    if not tests:
        print(f"  (no tests found in {mod.__name__})")
        return 0
    before = len(_State.failures)
    for fn in tests:
        _State.current = fn.__name__
        print(f"  {fn.__name__}")
        try:
            fn()
        except Exception as exc:
            _State.failures.append(f"{fn.__name__}::raised")
            print(f"    FAIL  raised {type(exc).__name__}: {exc}")
            traceback.print_exc()
    return len(_State.failures) - before


def main(mod) -> int:
    print(f"== {mod.__name__} ==")
    failed = run_module(mod)
    if failed:
        print(f"\n{failed} FAILED:")
        for f in _State.failures[-failed:]:
            print(f"  {f}")
        return 1
    print("\nall passed")
    return 0
