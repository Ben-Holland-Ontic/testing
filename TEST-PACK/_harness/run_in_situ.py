# -*- coding: utf-8 -*-
"""Run a pack script from its ORIGINAL location, with the pipeline root on sys.path.

Six pack scripts resolve their project root and fixtures from __file__ (``_ROOT =
Path(__file__).resolve().parent.parent``). Collecting them into TEST-PACK changes their
depth, so _ROOT points at the pack and their imports and fixture paths break. This launcher
sidesteps that: the pack is the index, and execution uses the file still sitting in its
original tree.

    python _harness/run_in_situ.py --list
    python _harness/run_in_situ.py test_table_assembly_regression.py
    python _harness/run_in_situ.py test_table_assembly_regression.py -- --some-flag

Note the bundled interpreter ignores PYTHONPATH, because a python311._pth file is present.
That is why the root is injected here rather than set in the environment.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import runpy
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(PACK, '_manifest.json')

# Roots that make `workers.x`, `cleaners.x` and bare `oculi_ocr` resolve. Both trees are
# offered because the old-snapshot scripts were written against the snapshot's own layout.
ROOTS = [
    r'C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main',
    r'C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main\workers',
    r'C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main\cleaners',
    r'C:\Users\benja\Downloads\bk-main\bk-main',
]


def load():
    with open(MANIFEST, encoding='utf-8') as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('script', nargs='?', help='script file name, as it appears in the pack')
    ap.add_argument('--list', action='store_true', help='list runnable scripts and exit')
    ap.add_argument('rest', nargs=argparse.REMAINDER, help='arguments passed to the script')
    args = ap.parse_args()

    man = load()
    if args.list or not args.script:
        print('%-50s %-8s %s' % ('SCRIPT', 'ORIGIN', 'IN-SITU PATH EXISTS'))
        for e in sorted(man, key=lambda r: r['rel']):
            name = os.path.basename(e['rel'])
            print('  %-48s %-8s %s' % (name, e['tag'], os.path.isfile(e['src'])))
        return 0

    want = os.path.basename(args.script)
    hits = [e for e in man if os.path.basename(e['rel']) == want]
    if not hits:
        print('not in the pack: %s  (try --list)' % want)
        return 2
    entry = hits[0]
    src = entry['src']
    if not os.path.isfile(src):
        print('original is gone, cannot run in situ: %s' % src)
        return 2

    for r in reversed(ROOTS):
        if os.path.isdir(r) and r not in sys.path:
            sys.path.insert(0, r)
    # the script's own directory, as it would see it when run normally
    sys.path.insert(0, os.path.dirname(src))

    rest = [a for a in args.rest if a != '--']
    sys.argv = [src] + rest
    os.chdir(os.path.dirname(src))

    print('running %s' % src)
    print('  origin : %s tree' % entry['tag'])
    print('  cwd    : %s' % os.getcwd())
    print('-' * 70)
    try:
        runpy.run_path(src, run_name='__main__')
    except SystemExit as e:
        return int(e.code or 0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
