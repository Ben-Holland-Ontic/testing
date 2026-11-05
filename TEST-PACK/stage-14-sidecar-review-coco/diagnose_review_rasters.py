# -*- coding: utf-8 -*-
"""READ-ONLY diagnostic: can the classification-review page rasters be recovered?

Writes nothing. Reports, per page, whether the review raster exists, whether the original
source file is still reachable, and whether the target canvas dimensions are known - i.e.
whether a non-destructive backfill could rebuild the image without re-extracting.

Usage:
    python diagnose_review_rasters.py <session-dir> [--verbose]
    python diagnose_review_rasters.py --run-id <run-id>          (uses the default session root)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DEFAULT_ROOT = os.path.join(
    os.environ.get('LOCALAPPDATA', ''), 'DocumentExtractionConsole', 'sessions')


def bbox_extent(page):
    mx = my = 0.0
    def scan(lst):
        nonlocal mx, my
        for r in lst or []:
            bb = r.get('bbox') or r.get('bbox_global')
            if bb and len(bb) >= 4:
                try:
                    mx = max(mx, float(bb[2])); my = max(my, float(bb[3]))
                except (TypeError, ValueError):
                    pass
    ci = page.get('cleaning_inputs') or {}
    for k in ('rectangles', 'contours', 'region_lines', 'cluster_lines', 'line_segments'):
        scan(ci.get(k))
    scan(page.get('ocr_lines'))
    fb = page.get('frame_bbox')
    if fb and len(fb) >= 4:
        try:
            mx = max(mx, float(fb[2])); my = max(my, float(fb[3]))
        except (TypeError, ValueError):
            pass
    return mx, my


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', nargs='?')
    ap.add_argument('--run-id')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    sess = a.session or (os.path.join(DEFAULT_ROOT, a.run_id) if a.run_id else None)
    if not sess or not os.path.isdir(sess):
        print('session directory not found: %r' % sess)
        print('default root: %s' % DEFAULT_ROOT)
        if os.path.isdir(DEFAULT_ROOT):
            print('available runs:')
            for d in sorted(os.listdir(DEFAULT_ROOT))[:40]:
                print('   ', d)
        return 2

    rd = os.path.join(sess, 'review_data')
    files = sorted(glob.glob(os.path.join(rd, '*.json.gz')))
    print('session        : %s' % sess)
    print('review_data    : %d document(s)' % len(files))
    if not files:
        print('\nNo review_data. The review tab has nothing to work from and the run would '
              'have to be repeated.')
        return 1

    tot = have = recoverable = no_source = no_dims = 0
    sources_missing = {}
    for f in files:
        try:
            doc = json.loads(gzip.open(f, 'rt', encoding='utf-8').read())
        except Exception as exc:
            print('  !! unreadable %s: %s' % (os.path.basename(f), exc))
            continue
        for p in doc.get('pages', []):
            tot += 1
            pid = p.get('page_id')
            rip = p.get('render_image_path')
            sp = p.get('storage_path')
            w = p.get('render_width') or p.get('width')
            h = p.get('render_height') or p.get('height')
            mx, my = bbox_extent(p)
            raster_ok = bool(rip) and os.path.exists(rip)
            src_ok = bool(sp) and os.path.exists(sp)
            dims_ok = bool(w) and bool(h) and float(w) > 1 and float(h) > 1

            if raster_ok:
                have += 1
                state = 'HAVE RASTER'
            elif not dims_ok:
                no_dims += 1
                state = 'NO DIMS'
            elif src_ok:
                recoverable += 1
                state = 'RECOVERABLE'
            else:
                no_source += 1
                state = 'SOURCE GONE'
                sources_missing[sp or '(none)'] = sources_missing.get(sp or '(none)', 0) + 1

            if a.verbose or state not in ('HAVE RASTER',):
                print('  %-13s %-46s canvas=%sx%s  maxbox=%.0fx%.0f  src=%s'
                      % (state, (pid or '?')[:46], w, h, mx, my,
                         (os.path.basename(sp) if sp else None)))

    print('\n--- summary ---')
    print('  pages total                  : %d' % tot)
    print('  raster already present       : %d' % have)
    print('  recoverable by re-render     : %d' % recoverable)
    print('  source file gone             : %d' % no_source)
    print('  canvas dims unknown          : %d' % no_dims)

    if sources_missing:
        print('\n  missing sources (top 10):')
        for k, v in sorted(sources_missing.items(), key=lambda t: -t[1])[:10]:
            print('    %4d x  %s' % (v, k))

    ann = sorted(glob.glob(os.path.join(sess, 'annotations', '*.json')))
    print('\n  annotation graphs already built: %d' % len(ann))
    stale = 0
    for g in ann:
        try:
            gg = json.loads(open(g, encoding='utf-8').read())
        except Exception:
            continue
        for p in gg.get('pages', []):
            ip = p.get('image_path')
            if not ip or not os.path.exists(ip):
                stale += 1
    if ann:
        print('  graph pages with no usable image_path: %d'
              '   (these need image_path patched after a backfill)' % stale)

    print('\n--- verdict ---')
    if tot and have == tot:
        print('  All rasters present. The review tab should display images; if it does not, the '
              'problem is elsewhere.')
    elif recoverable and not no_source:
        print('  RECOVERABLE. Every page without a raster still has its source file and known '
              'canvas dimensions, so a backfill can rebuild the images without re-extracting.')
    elif recoverable:
        print('  PARTIALLY RECOVERABLE. %d page(s) can be rebuilt; %d have lost their source '
              'file and would need re-extraction (or the source restoring to its old path).'
              % (recoverable, no_source))
    else:
        print('  NOT RECOVERABLE from review_data alone - sources are gone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
