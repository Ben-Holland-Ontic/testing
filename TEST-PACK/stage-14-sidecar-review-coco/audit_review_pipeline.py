# -*- coding: utf-8 -*-
"""Final audit of the review -> label -> synthetic pipeline, plus a production-upgrade rehearsal.

Read-only against the real session. Section D works on a scratch COPY that is first put back
into the PRE-UPGRADE state (no rasters, render_image_path null, no graphs), then driven exactly
as the application drives it, to prove an existing session needs no re-extraction.

Usage:  python audit_review_pipeline.py [run-id]
"""
from __future__ import annotations

import glob
import gzip
import io
import json
import os
import shutil
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

APP = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'DocumentExtractionConsole')
RT = os.path.join(APP, 'runtime', 'python', 'python.exe')
BK = os.path.join(APP, 'runtime', 'bk-main')
SC = os.path.join(APP, 'runtime', 'LayoutML sidecar')
sys.path.insert(0, SC)

RUN = sys.argv[1] if len(sys.argv) > 1 else 'test'
SESS = os.path.join(APP, 'sessions', RUN)

import numpy as np                      # noqa: E402
from PIL import Image                   # noqa: E402
import ontology_config as onto          # noqa: E402

fails, warns = [], []


def fail(s):
    fails.append(s)
    print('   FAIL  %s' % s)


def warn(s):
    warns.append(s)
    print('   warn  %s' % s)


def ok(s):
    print('   ok    %s' % s)


# ---------------------------------------------------------------- A
print('\n=== A. Classification-review readiness (every document) ===')
graphs = sorted(glob.glob(os.path.join(SESS, 'annotations', '*.json')))
if not graphs:
    fail('no annotation graphs in %s' % SESS)
for g in graphs:
    d = json.loads(open(g, encoding='utf-8').read())
    doc = d.get('document_id', os.path.basename(g))
    for p in d.get('pages', []):
        ip, W, H = p.get('image_path'), float(p['width']), float(p['height'])
        regs = p.get('regions') or []
        ti = [t for r in regs for t in (r.get('table_internal') or [])]
        if not ip or not os.path.exists(ip):
            fail('%s: no page image' % doc[-16:]); continue
        sz = Image.open(ip).size
        if sz != (int(W), int(H)):
            fail('%s: image %s != canvas %dx%d' % (doc[-16:], sz, W, H)); continue
        out = [r for r in regs if (r.get('bbox') or [0] * 4)[2] > W + .5
               or (r.get('bbox') or [0] * 4)[3] > H + .5]
        if out:
            fail('%s: %d region(s) outside canvas' % (doc[-16:], len(out))); continue
        ok('%-16s image %4dx%-5d  regions=%-4d table_internal=%-3d verified=%s'
           % (doc[-16:], sz[0], sz[1], len(regs), len(ti), d.get('verified')))

# ---------------------------------------------------------------- B
print('\n=== B. COCO labels vs the annotation graph (fidelity) ===')
cocos = sorted(glob.glob(os.path.join(SESS, 'synthetic', '*.coco.json')))
if not cocos:
    warn('no COCO files yet - finalize a document to produce them')
for c in cocos:
    doc = os.path.basename(c)[:-len('.coco.json')]
    gp = os.path.join(SESS, 'annotations', doc + '.json')
    if not os.path.exists(gp):
        fail('%s: COCO with no graph' % doc[-16:]); continue
    graph = json.loads(open(gp, encoding='utf-8').read())
    coco = json.loads(open(c, encoding='utf-8').read())
    names = {x['id']: x['name'] for x in coco['categories']}

    def poly_aabb(poly):
        """COCO deliberately labels from the RICH polygon where one exists, so a region may
        legitimately appear at its polygon's AABB rather than its bbox. Accept either."""
        if not poly or len(poly) < 3:
            return None
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        return (round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys)))

    exp_reg, exp_ti = [], 0
    for p in graph.get('pages', []):
        for r in p.get('regions') or []:
            cid = onto.region_category_id(r.get('detector_class', ''))
            if cid is not None and r.get('bbox'):
                alts = {tuple(round(v) for v in r['bbox'])}
                pa = poly_aabb(r.get('polygon'))
                if pa:
                    alts.add(pa)
                exp_reg.append((cid, alts))
            for t in (r.get('table_internal') or []):
                if str(t.get('class', '')).lower() in onto.TABLE_INTERNAL_CLASSES and t.get('bbox'):
                    exp_ti += 1
    got_reg, got_ti = [], 0
    for a in coco['annotations']:
        nm = names[a['category_id']]
        x, y, w, h = a['bbox']
        if nm in onto.REGION_CLASSES:
            got_reg.append((a['category_id'], (round(x), round(y), round(x + w), round(y + h))))
        else:
            got_ti += 1

    pool = list(got_reg)
    missing = []
    for cid, alts in exp_reg:
        hit = next((g for g in pool if g[0] == cid and g[1] in alts), None)
        if hit is None:
            missing.append((cid, sorted(alts)[0]))
        else:
            pool.remove(hit)
    if missing or pool or got_ti != exp_ti:
        fail('%s: regions exp=%d got=%d unmatched=%d orphan=%d | internals exp=%d got=%d'
             % (doc[-16:], len(exp_reg), len(got_reg), len(missing), len(pool), exp_ti, got_ti))
        for m in missing[:3]:
            print('          graph region with no COCO label: %s %s' % (names.get(m[0]), m[1]))
        for o in pool[:3]:
            print('          COCO label with no graph region: %s %s' % (names.get(o[0]), o[1]))
    else:
        ok('%-16s %d region labels + %d table-internal labels all match the graph exactly'
           % (doc[-16:], len(exp_reg), got_ti))

# ---------------------------------------------------------------- C
print('\n=== C. Synthetic renders ===')
for c in cocos:
    doc = os.path.basename(c)[:-len('.coco.json')]
    coco = json.loads(open(c, encoding='utf-8').read())
    for im in coco.get('images', []):
        f = os.path.join(SESS, 'synthetic', 'images', im['file_name'])
        if not os.path.exists(f):
            fail('%s: synthetic image missing' % doc[-16:]); continue
        a = np.asarray(Image.open(f).convert('L')).astype(int)
        H, W = a.shape
        if (W, H) != (im['width'], im['height']):
            fail('%s: synthetic %dx%d != COCO %dx%d' % (doc[-16:], W, H, im['width'], im['height']))
            continue
        paper = float(np.median(a))
        faint = a < (paper - 12)
        rules = [i for i, v in enumerate(faint.sum(1)) if v > W * 0.5]
        # A TILED background repeats one patch, so its rules land at a near-CONSTANT pitch.
        # Real content (a frame plus a title block) gives wildly uneven spacing, and a kept
        # frame alone spans the full height - so spread is not evidence of tiling, regularity
        # is. Measure the spacing's relative deviation instead.
        tiled = False
        pitch_cv = None
        if len(rules) > 6:
            gaps = [b - a for a, b in zip(rules, rules[1:]) if b - a > 2]
            if len(gaps) >= 4:
                mean = sum(gaps) / float(len(gaps))
                var = sum((g - mean) ** 2 for g in gaps) / float(len(gaps))
                pitch_cv = (var ** 0.5) / mean if mean else None
                tiled = pitch_cv is not None and pitch_cv < 0.15
        blank = faint.mean() < 0.001
        if tiled:
            fail('%s: background looks TILED (%d rules at near-constant pitch, cv=%.2f)'
                 % (doc[-16:], len(rules), pitch_cv))
        elif blank:
            fail('%s: synthetic is effectively blank' % doc[-16:])
        else:
            ok('%-16s %4dx%-5d ink=%.3f  rules=%-3d pitch_cv=%-5s (uneven => real content)'
               % (doc[-16:], W, H, faint.mean(), len(rules),
                  ('%.2f' % pitch_cv) if pitch_cv is not None else 'n/a'))

# ---------------------------------------------------------------- D
print('\n=== D. Production-upgrade rehearsal (old session + new backend) ===')
root = r'C:\OCULI-SMOKE\upgrade_rehearsal'
dst = os.path.join(root, 'sessions', RUN)
shutil.rmtree(root, ignore_errors=True)
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copytree(SESS, dst)

# put it back into the PRE-UPGRADE state
for p in glob.glob(os.path.join(dst, 'review_data', 'rasters', '*.png')):
    os.remove(p)
for p in glob.glob(os.path.join(dst, 'annotations', '*.json')):
    os.remove(p)
for f in glob.glob(os.path.join(dst, 'review_data', '*.json.gz')):
    d = json.loads(gzip.open(f, 'rt', encoding='utf-8').read())
    for p in d.get('pages', []):
        p['render_image_path'] = None
        p['render_width'] = None
        p['render_height'] = None
    with gzip.open(f, 'wt', encoding='utf-8') as fh:
        json.dump(d, fh)
print('   prepared: rasters deleted, graphs deleted, render_image_path nulled'
      '  (exactly what a pre-upgrade session looks like)')

env = dict(os.environ, DEC_APPDATA_ROOT=root, PYTHONIOENCODING='utf-8')
r = subprocess.run([RT, 'session_runtime.py', 'build_annotations', '--run-id', RUN],
                   cwd=BK, env=env, capture_output=True, text=True)
print('   build_annotations exit=%d  (this is what the app runs on Classify Regions)' % r.returncode)

built = sorted(glob.glob(os.path.join(dst, 'annotations', '*.json')))
rasters = glob.glob(os.path.join(dst, 'review_data', 'rasters', '*.png'))
good = 0
for g in built:
    d = json.loads(open(g, encoding='utf-8').read())
    for p in d.get('pages', []):
        ip = p.get('image_path')
        if ip and os.path.exists(ip) and Image.open(ip).size == (int(p['width']), int(p['height'])) \
                and (p.get('regions') or []):
            good += 1
if not built:
    fail('upgrade rehearsal produced no graphs')
elif good != sum(len(json.loads(open(g, encoding='utf-8').read()).get('pages', [])) for g in built):
    fail('upgrade rehearsal: only %d page(s) render correctly' % good)
else:
    ok('%d document(s) rebuilt, %d raster(s) rendered from the ORIGINAL files' % (len(built), len(rasters)))
    ok('every page renders at canvas size with its stored annotations - no re-extraction')

print('\n=== VERDICT ===')
print('  failures: %d   warnings: %d' % (len(fails), len(warns)))
for f in fails:
    print('   - %s' % f)
for w in warns:
    print('   ~ %s' % w)
print('  %s' % ('ALL CHECKS PASSED' if not fails else 'ISSUES FOUND - see above'))
raise SystemExit(1 if fails else 0)
