"""What would ROVER actually decide, on real crops, with the models we have?

DIAGNOSTIC. Answers three questions with measurement rather than argument:

  1. Does voting recover the cases any single model gets wrong?
  2. Does it ever DISPLACE a correct primary reading -- the regression that matters most?
  3. Does the credibility layer add anything on top of the vote, or is it redundant here?

Votes are taken SPACE-FREE (three of four supporting models cannot emit a space) and
case-insensitively (CATMuS returns lowercase on uppercase drawing labels, which is a
property of its training set, not a disagreement about the characters).

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY vote_simulation_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

BK = r"C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main"
for p in (BK, os.path.join(BK, "workers"), os.path.join(BK, "input"),
          os.path.join(BK, "cleaners")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("OCULI_SUPPORTING_MODEL_DIR", r"C:\OCULI.pd1\TESTING\onnx-models")
os.environ.setdefault("PADDLEOCR_USE_GPU", "1")

FIX = r"C:\OCULI.pd1\TESTING\RESOURCES\fixtures"
CROPS = os.path.join(FIX, "crops")
BASELINE = r"C:\OCULI.pd1\TESTING\RESULTS\parity\phase0_baseline.json"


def key(text: str) -> str:
    """The form candidates are compared in: space-free, case-folded."""
    return "".join(text.split()).casefold()


def vote(candidates):
    """[(model, text)] -> (winning_text, support, total, tally). Frequency only (alpha=1)."""
    tally = Counter(key(t) for _, t in candidates if t)
    if not tally:
        return "", 0, 0, tally
    best, support = tally.most_common(1)[0]
    # Tie -> no winner; the caller keeps the primary.
    if sum(1 for _, c in tally.items() if c == support) > 1:
        return "", support, sum(tally.values()), tally
    # Return the winner in the casing of the highest-priority model that produced it.
    for _, text in candidates:
        if text and key(text) == best:
            return text, support, sum(tally.values()), tally
    return best, support, sum(tally.values()), tally


def main() -> int:
    import cv2
    import ocr_backend
    import ocr_credibility as C
    import ocr_supporting as S

    ocr_backend.register_cuda_dll_dirs()

    manifest = json.load(open(os.path.join(FIX, "crops.json"), encoding="utf-8"))
    items = [(m, cv2.imread(os.path.join(CROPS, m["file"]))) for m in manifest]
    items = [(m, i) for m, i in items if i is not None]

    reads = {}
    for name in S.MODEL_NAMES:
        rec = S.get_recogniser(name)
        reads[name] = ([t for t, _ in rec.recognise([i for _, i in items])]
                       if rec else [""] * len(items))

    # Batch statistics from the real 8-page corpus -- what the credibility layer would
    # actually have available at this point in a run.
    corpus = [l["text"].strip() for d in json.load(open(BASELINE, encoding="utf-8"))["documents"]
              for l in d["unmatched_lines"] if (l.get("text") or "").strip()]
    stats = C.build_statistics((t, 1.0) for t in corpus)

    print(f"{len(items)} real crops; batch statistics from {len(corpus)} corpus lines "
          f"({stats.n_chars} chars)\n")

    truth_n = agree_n = 0
    primary_right = vote_right = 0
    displaced = []
    rescued = []

    for idx, (m, _img) in enumerate(items):
        primary = m["primary_text"]
        catmus = reads["catmus"][idx]
        cands = [("primary", primary)] + [(n, reads[n][idx]) for n in S.MODEL_NAMES]
        winner, support, total, tally = vote(cands)

        # Spacing is layout, decided separately, donated by CATMuS then the primary.
        donor = S.choose_donor(catmus, primary)
        spaced = S.apply_donor_spacing(winner, donor) if winner else ""

        truth = m.get("expected")
        if truth:
            truth_n += 1
            p_ok = key(primary) == key(truth)
            v_ok = key(spaced) == key(truth)
            primary_right += p_ok
            vote_right += v_ok
            if p_ok and not v_ok:
                displaced.append((m["file"], truth, spaced))
            if v_ok and not p_ok:
                rescued.append((m["file"], truth, primary, spaced))

        if key(spaced) == key(primary):
            agree_n += 1

        cred_p = C.credibility(primary, stats)
        cred_w = C.credibility(spaced, stats) if spaced else None
        mark = "=" if key(spaced) == key(primary) else "*"
        print(f"{mark} {m['file']}")
        print(f"    primary {primary!r}  (credibility {cred_p.total if cred_p.total is None else round(cred_p.total,2)})")
        for n in S.MODEL_NAMES:
            print(f"    {n:<10}{reads[n][idx]!r}")
        print(f"    VOTE -> {spaced!r}   support {support}/{total}"
              f"   credibility {cred_w.total if (cred_w is None or cred_w.total is None) else round(cred_w.total,2)}")

    print("\n" + "=" * 70)
    print(f"vote agrees with primary on {agree_n}/{len(items)} crops")
    if truth_n:
        print(f"on the {truth_n} hand-verified crops: primary correct {primary_right},"
              f" vote correct {vote_right}")
    print(f"DISPLACED (primary was right, vote wrong): {len(displaced)}")
    for f, t, g in displaced:
        print(f"    {f}  truth={t!r} vote={g!r}")
    print(f"RESCUED  (primary wrong, vote right): {len(rescued)}")
    for f, t, p, g in rescued:
        print(f"    {f}  truth={t!r} primary={p!r} vote={g!r}")

    # Would credibility have added anything the vote did not already settle?
    print("\ncredibility as a tiebreak: how it ranks each crop's candidate set")
    unanimous = disputed = 0
    for idx, (m, _) in enumerate(items):
        ks = {key(t) for t in [m["primary_text"]] + [reads[n][idx] for n in S.MODEL_NAMES] if t}
        if len(ks) == 1:
            unanimous += 1
        else:
            disputed += 1
    print(f"  unanimous {unanimous}/{len(items)}, disputed {disputed}/{len(items)} "
          f"-- credibility only has a job on the disputed ones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
