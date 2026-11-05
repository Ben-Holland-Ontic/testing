"""Run every test module in this directory.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY TESTING\\tests\\run_all.py

Exit code is the number of modules that had failures (0 = all green).
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))

import _harness  # noqa: E402


def main() -> int:
    modules = sorted(p.stem for p in HERE.glob("test_*.py"))
    if not modules:
        print("no test modules found")
        return 0

    failed_modules = []
    for name in modules:
        mod = importlib.import_module(name)
        print(f"== {name} ==")
        if _harness.run_module(mod):
            failed_modules.append(name)
        print()

    total = len(_harness._State.failures)
    if total:
        print(f"{total} check(s) failed across {len(failed_modules)} module(s):")
        for f in _harness._State.failures:
            print(f"  {f}")
        return len(failed_modules)
    print(f"all passed ({len(modules)} module(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
