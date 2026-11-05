"""The cascade, wired the way the plan actually specified: per detected line region.

DIAGNOSTIC. This replaces an earlier measurement that handed the supporting models whole
crops and concluded they were too weak to ensemble. That was a plumbing bug: on the
primary's own detected line regions the same models jump from 16-17% exact to 58-65%.

So this runs the real comparison:

  * for every crop, ask the PRIMARY where the text is and what it says, line by line;
  * crop to each detected line region and ask the four supporting models about THOSE
    pixels -- same input unit, 1:1 comparable;
  * vote per line, apply donor spacing, and report what changed.

Two groups are measured separately because they are different questions:

  REGRESSION   crops the primary read at conf >= 0.95. Changes here are almost certainly
               damage; the target is zero.
  OPPORTUNITY  crops the primary read at conf < 0.85. These are real text read badly --
               'AUIRUMKEU KEPKE JERIAIIVE UP' for AUTHORIZED REPRESENTATIVE OF -- and are
               where the gain has to come from, if there is one.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY line_vote_probe.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import warnings

warnings.filterwarnings("ignore")

BK = r"C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main"
for p in (BK, os.path.join(BK, "workers"), os.path.join(BK, "input"),
          os.path.join(BK, "cleaners")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("OCULI_SUPPORTING_MODEL_DIR", r"C:\OCULI.pd1\TESTING\onnx-models")
os.environ.setdefault("PADDLEOCR_USE_GPU", "1")

OUT = r"C:\OCULI.pd1\TESTING\RESULTS\hard_crops"
CROP_DIR = os.path.join(OUT, "crops")
MANIFEST = os.path.join(OUT, "manifest.json")
REPORT = os.path.join(OUT, "line_vote.json")
CHUNK = 32
LIMIT = 260          # per group, keeps the experiment quick and memory flat


def key(t: str) -> str:
    return "".join(t.split()).casefold()


def cer(a: str, b: str) -> float:
    if not b:
        return 0.0 if not a else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(b)


def collect_lines(metas, engine, cluster_ocr, cv2):
    """[(meta, line_text, line_conf, line_image)] using the primary's own detection."""
    rows = []
    for m in metas:
        img = cv2.imread(os.path.join(CROP_DIR, m["file"]))
        if img is None:
            continue
        res = cluster_ocr._normalize_paddle_list_result(engine.ocr(img))
        for ln in cluster_ocr._parse_paddle_result_to_lines(res, 0, 0, 0, "PROBE"):
            text = (ln.get("text") or "").strip()
            bbox = ln.get("bbox") or []
            if not text or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
            x1 = max(0, x1 - 2); y1 = max(0, y1 - 2)
            x2 = min(img.shape[1], x2 + 2); y2 = min(img.shape[0], y2 + 2)
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            rows.append((m, text, float(ln.get("confidence", 0.0) or 0.0),
                         img[y1:y2, x1:x2]))
    return rows


def run_group(name, metas, engine, modules):
    cv2, cluster_ocr, Cr, S, V, stats = modules
    cred = lambda t: Cr.credibility(t, stats).total  # noqa: E731

    rows = collect_lines(metas, engine, cluster_ocr, cv2)
    print(f"\n{name}: {len(metas)} crops -> {len(rows)} detected line regions")
    if not rows:
        return []

    changes = []
    for off in range(0, len(rows), CHUNK):
        part = rows[off:off + CHUNK]
        imgs = [r[3] for r in part]
        reads = {}
        for n in S.MODEL_NAMES:
            rec = S.get_recogniser(n)
            reads[n] = rec.recognise(imgs) if rec else [("", 0.0)] * len(imgs)
        for i, (m, ref, conf, _im) in enumerate(part):
            cands = [V.Candidate("primary", ref, conf)]
            cands += [V.Candidate(n, reads[n][i][0], reads[n][i][1])
                      for n in S.MODEL_NAMES]
            donor = S.choose_donor(reads["catmus"][i][0], ref)
            d = V.decide(cands, donor=donor, credibility_of=cred)
            if d.changed:
                changes.append({
                    "file": m["file"], "conf": round(conf, 3),
                    "primary": ref, "vote": d.text, "reason": d.reason,
                    "models": {n: reads[n][i][0] for n in S.MODEL_NAMES},
                    "cer_delta": None,
                })
        del imgs, reads

    print(f"  votes that CHANGED the primary: {len(changes)} / {len(rows)} lines "
          f"({100 * len(changes) / len(rows):.1f}%)")
    return changes


def main() -> int:
    import cv2
    import cluster_ocr
    import ocr_backend
    import ocr_credibility as Cr
    import ocr_supporting as S
    import ocr_voting as V

    ocr_backend.register_cuda_dll_dirs()
    engine = cluster_ocr._get_paddleocr()

    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    confident = [m for m in manifest if m["primary_text"] and m["primary_conf"] >= 0.95]
    weak = [m for m in manifest if m["primary_text"] and m["primary_conf"] < 0.85]

    corpus = [m["primary_text"] for m in confident]
    stats = Cr.build_statistics((t, 1.0) for t in corpus)
    print(f"batch statistics: {stats.n_chars} chars from {len(corpus)} confident readings")

    modules = (cv2, cluster_ocr, Cr, S, V, stats)
    reg = run_group("REGRESSION  (primary conf >= 0.95)", confident[:LIMIT], engine, modules)
    opp = run_group("OPPORTUNITY (primary conf <  0.85)", weak[:LIMIT], engine, modules)

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump({"regressions": reg, "opportunities": opp}, f, indent=2,
                  ensure_ascii=False)

    print("\n" + "=" * 74)
    print("CHANGES ON CONFIDENT CROPS -- every one of these is probably damage")
    print("=" * 74)
    for r in reg[:15]:
        print(f"  {r['file']} conf={r['conf']}")
        print(f"      {r['primary']!r}  ->  {r['vote']!r}")
        print(f"      {'  '.join(f'{k}={v!r}' for k, v in r['models'].items() if v)}")

    print("\n" + "=" * 74)
    print("CHANGES ON WEAK CROPS -- judge these by eye; this is the whole case")
    print("=" * 74)
    for r in opp[:30]:
        print(f"  {r['file']} conf={r['conf']}")
        print(f"      {r['primary']!r}  ->  {r['vote']!r}")
        print(f"      {'  '.join(f'{k}={v!r}' for k, v in r['models'].items() if v)}")
    print(f"\nfull report -> {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
