"""Phase 2 - the per-batch credibility layer.

Judges whether a string looks like text from THIS batch, using only this batch's own
high-confidence readings. No predefined patterns, no persistent state, no model.

The corpus for these tests is real: the 742 unmatched_text lines the pipeline actually
produced on the 8-page reference run (TESTING/RESULTS/parity/phase0_baseline.json). The
garbage strings are also real -- they are what the bundled handwriting recogniser
returned on this corpus, at high confidence.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY TESTING\\tests\\test_phase2_credibility.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import add_runtime_to_path, check, check_eq, main  # noqa: E402

add_runtime_to_path()

BASELINE = Path(r"C:\OCULI.pd1\TESTING\RESULTS\parity\phase0_baseline.json")

# Measured outputs of the handwriting recogniser on this corpus, at high confidence.
# These are the failure the whole voting design exists to catch.
#
# Split in two, because MEASUREMENT says they are different problems. Every string in
# TRUE_GARBAGE is beaten by ~100% of real corpus lines; the NEAR_MISS strings are not.
# Garbage that keeps real word SHAPE is invisible to a character model, by construction.
# Asking the n-gram signal to catch it would be testing a thing it cannot do; that class
# belongs to the lexicon and to repetition.
#
# PROVENANCE of every string below, because a fabricated "measurement" is worse than no
# measurement:
#   'pote hivme', 'vaveassome'      handwriting engine, recorded in cluster_ocr.py
#   'pite oe ...', 'paotat- | ...'  handwriting engine, recorded in the plan
#   'body Cop serew'                CATMuS, fixtures/reference.json (Ball_Valve_dwg_1_08)
#   'Thnest Washer deonng'          CATMuS, fixtures/reference.json (Ball_Valve_dwg_1_02)
#   '3oay Cap Screw'                PP-OCRv6 itself, in the 8-page corpus
TRUE_GARBAGE = [
    "pite oe ovvsovse ome toteove",
    "paotat- | pie ooivvette",
    "pote hivme",
    "vaveassome",
    "Thnest Washer deonng",
]
NEAR_MISS_GARBAGE = ["body Cop serew", "3oay Cap Screw"]
GARBAGE = TRUE_GARBAGE + NEAR_MISS_GARBAGE

REAL = [
    "INTERPRET DIMENSIONS AND TOLERANCES PER",
    "FASTENER TORQUE TABLE",
    "UNLESS OTHERWISE SPECIFIED",
    "MATERIAL DESCRIPTION",
    "Plain Carbon Steel",
]


def _corpus():
    """Every line of real OCR text from the 8-page reference run."""
    if not BASELINE.is_file():
        return []
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    out = []
    for doc in data["documents"]:
        for line in doc["unmatched_lines"]:
            text = (line.get("text") or "").strip()
            if text:
                out.append(text)
    return out


def _stats_from_corpus():
    import ocr_credibility as C
    # The digest does not carry per-line confidence, so these are observed at 1.0. That
    # is honest for a fixture: they are the pipeline's actual accepted output.
    return C.build_statistics((t, 1.0) for t in _corpus())


def test_ngram_rejects_garbage():
    """Real garbage scores below real text -- measured as separation, not as a threshold.

    An absolute cutoff would be a magic number tuned to one corpus. What generalises is
    the ORDERING: garbage below text.
    """
    corpus = _corpus()
    if not check("real corpus available", len(corpus) > 100, f"{len(corpus)} lines"):
        return
    stats = _stats_from_corpus()
    check(f"corpus is big enough for an opinion ({stats.n_chars} chars)",
          stats.has_ngram_opinion)

    scored = [stats.ngram_logprob(t) for t in corpus]
    for text in TRUE_GARBAGE:
        score = stats.ngram_logprob(text)
        above = sum(1 for s in scored if s > score) / len(scored)
        check(f"beaten by {above:.1%} of the corpus: {text[:34]!r} = {score:.3f}",
              above >= 0.95, f"expected >=95%")

    worst_real = min(stats.ngram_logprob(t) for t in REAL)
    for text in TRUE_GARBAGE:
        check(f"below every known-real string: {text[:34]!r}",
              stats.ngram_logprob(text) < worst_real,
              f"worst real = {worst_real:.3f}")


def test_ngram_does_not_claim_to_catch_near_misses():
    """The documented blind spot, asserted so it cannot be forgotten.

    'Body Cap Screw' misread as '3oay Cop serew' keeps real word shape, so a character
    model cannot reject it. This test exists to pin that limit down: if a future change
    makes the n-gram appear to catch it, the reason needs checking, not celebrating.
    """
    stats = _stats_from_corpus()
    if not check("corpus available", stats.has_ngram_opinion):
        return
    corpus = _corpus()
    scored = [stats.ngram_logprob(t) for t in corpus]
    for text in NEAR_MISS_GARBAGE:
        score = stats.ngram_logprob(text)
        above = sum(1 for s in scored if s > score) / len(scored)
        check(f"near-miss garbage is NOT separated by n-grams alone: {text!r} "
              f"beaten by only {above:.1%}", above < 0.95)

    # ...and the lexicon is what catches it, given a batch that knows the right form.
    import ocr_credibility as C
    trained = C.build_statistics(
        [("BODY CAP SCREW", 1.0)] * 5 + [("PART NUMBER", 1.0)] * 5
        + [("MATERIAL DESCRIPTION", 1.0)] * 5 + [("SHEET SIZE", 1.0)] * 5
        + [("REVISION", 1.0)] * 5)
    check("the lexicon has not seen 'serew'", trained.token_support("serew") == 0)
    check("the lexicon has seen 'screw'", trained.token_support("screw") >= 3)
    good = C.credibility("BODY CAP SCREW", trained)
    bad = C.credibility("3oay Cop serew", trained)
    check("lexicon separates the near-miss where n-grams cannot",
          good.lexicon is not None and bad.lexicon is not None
          and good.lexicon > bad.lexicon,
          f"good={good.lexicon} bad={bad.lexicon}")


def test_ngram_accepts_real():
    stats = _stats_from_corpus()
    if not check("corpus available", stats.has_ngram_opinion):
        return

    ceiling = max(stats.ngram_logprob(g) for g in TRUE_GARBAGE)
    for text in REAL:
        score = stats.ngram_logprob(text)
        check(f"real string beats the true-garbage ceiling: {text[:34]!r} = "
              f"{score:.3f}", score > ceiling, f"ceiling {ceiling:.3f}")

    # Held-out: corpus lines themselves, sampled across the corpus.
    corpus = _corpus()
    sample = corpus[:: max(1, len(corpus) // 40)][:40]
    below = [t for t in sample if (stats.ngram_logprob(t) or -99) <= ceiling]
    check(f"{len(sample) - len(below)}/{len(sample)} sampled corpus lines beat the "
          f"true-garbage ceiling", len(below) <= len(sample) // 10,
          "" if not below else f"worst: {below[:3]}")


def test_thresholds_are_where_measurement_put_them():
    """Guards the numbers the docstring justifies, so a silent edit is caught."""
    import ocr_credibility as C

    check(f"MIN_NGRAM_CHARS={C.MIN_NGRAM_CHARS} is above the unreliable band (<2.5K)",
          C.MIN_NGRAM_CHARS >= 2500)
    stats = _stats_from_corpus()
    check(f"the 8-page corpus ({stats.n_chars} chars) clears it",
          stats.has_ngram_opinion)

    # ...and a corpus in the measured-unreliable band does not.
    corpus = _corpus()
    small = C.build_statistics((t, 1.0) for t in corpus[:80])
    check(f"a small corpus ({small.n_chars} chars) abstains",
          not small.has_ngram_opinion)


def test_cold_start_abstains():
    """Below the evidence threshold it must say nothing, not say something quietly wrong.

    A weak opinion is worse than none: a low score on a correct reading would displace it.
    """
    import ocr_credibility as C

    empty = C.BatchStatistics()
    result = C.credibility("INTERPRET DIMENSIONS", empty)
    check("empty statistics have no opinion", not result.has_opinion)
    check_eq("total is None, not 0.0", result.total, None)
    check_eq("ngram is None", empty.ngram_logprob("anything"), None)
    check("the reason is stated", "no opinion" in result.detail, result.detail)

    # A handful of strings is still not enough.
    thin = C.build_statistics([("PART NUMBER", 1.0), ("REV", 1.0), ("SHEET", 1.0)])
    check(f"3 strings ({thin.n_chars} chars) is still no ngram opinion",
          not thin.has_ngram_opinion)
    check("...and no lexicon opinion", not thin.has_lexicon_opinion)
    check("credibility abstains", not C.credibility("PART NUMBER", thin).has_opinion)

    # Crossing the threshold turns the opinion on, and only then.
    full = _stats_from_corpus()
    if full.n_chars:
        check("the real corpus does have an opinion",
              C.credibility("PART NUMBER", full).has_opinion)


def test_lexicon_edit_repair():
    import ocr_credibility as C

    # A batch where 'ISO 2768-1' is frequent and 'ZQX-9931' appears once.
    readings = [("ISO 2768-1", 1.0)] * 6 + [("ZQX-9931", 1.0)]
    readings += [("MATERIAL DESCRIPTION", 1.0)] * 4
    readings += [("PART NUMBER", 1.0)] * 4 + [("SHEET SIZE", 1.0)] * 4
    readings += [("REVISION HISTORY", 1.0)] * 4 + [("TORQUE TABLE", 1.0)] * 4
    stats = C.build_statistics(readings)
    check(f"lexicon has an opinion ({stats.n_tokens} tokens)",
          stats.has_lexicon_opinion)

    repaired, edits = stats.lexicon_repair("SO 2706-1")
    # The plan's example wanted 'ISO 2768-1'. Only the numeric token is repaired, ON
    # PURPOSE: the edit budget is 0 for tokens of three characters or fewer, because one
    # wrong character in a two-character token IS a different token. Drawings are full of
    # real short tokens -- ID, OD, NO, QTY, REV -- and repairing those towards whatever
    # frequent token happens to be one edit away would corrupt correct readings to fix
    # rare ones. The conservative half of the repair is the half worth having.
    check_eq("the long near-miss token is corrected", repaired, "SO 2768-1")
    check_eq("the edit is reported", edits, [("2706-1", "2768-1")])
    check_eq("a 2-character token is deliberately NOT repaired",
             stats.lexicon_repair("SO")[0], "SO")
    check_eq("...nor a 3-character one", stats.lexicon_repair("PRT")[0], "PRT")
    check_eq("edit budget: 0 below 4 chars", C.BatchStatistics.edit_budget("abc"), 0)
    check_eq("edit budget: 1 at 4-5 chars", C.BatchStatistics.edit_budget("abcd"), 1)
    check_eq("edit budget: 2 at 6+ chars", C.BatchStatistics.edit_budget("abcdef"), 2)

    # A rare token with no frequent neighbour is left alone: repair must not invent.
    untouched, edits2 = stats.lexicon_repair("ZQX-9931")
    check_eq("an unrelated rare token is left alone", untouched, "ZQX-9931")
    check_eq("...with no edits claimed", edits2, [])

    # A token already in the lexicon is never "repaired".
    same, edits3 = stats.lexicon_repair("ISO 2768-1")
    check_eq("a known token is untouched", same, "ISO 2768-1")
    check_eq("...with no edits", edits3, [])

    # Ambiguity must abstain rather than guess.
    ambiguous = C.build_statistics(
        [("CAT", 1.0)] * 4 + [("BAT", 1.0)] * 4 + [("PART NUMBER", 1.0)] * 8
        + [("SHEET SIZE", 1.0)] * 8 + [("REVISION", 1.0)] * 8)
    got, edits4 = ambiguous.lexicon_repair("XAT")
    check_eq("two equally close targets => no repair", got, "XAT")
    check_eq("...and no edits", edits4, [])

    # Case is restored, so a repair cannot silently change an uppercase drawing label.
    cased, _ = stats.lexicon_repair("MATERIAL DESCRIPTIONN")
    check("repairs keep the original case", cased.isupper(), cased)


def test_repetition_corroborates():
    import ocr_credibility as C

    # Needs a real batch: below MIN_REPETITION_STRINGS the repetition signal abstains,
    # because in a batch of three every string is either unique or a third of the corpus.
    readings = [("TORQUE TABLE", 1.0)] * 3 + [("ODDMENT", 1.0)]
    readings += [("PART NUMBER", 1.0)] * 10 + [("SHEET SIZE", 1.0)] * 10
    readings += [("REVISION", 1.0)] * 10
    stats = C.build_statistics(readings)
    check(f"batch is big enough for a repetition opinion ({stats.n_strings} strings)",
          stats.has_repetition_opinion)

    thin = C.build_statistics([("A B", 1.0), ("C D", 1.0)])
    check("a tiny batch has NO repetition opinion", not thin.has_repetition_opinion)
    check_eq("...and returns None rather than a count", thin.repetition("A B"), None)

    check_eq("a string seen 3x is counted 3x", stats.repetition("TORQUE TABLE"), 3.0)
    check_eq("a hapax is counted once", stats.repetition("ODDMENT"), 1.0)
    check_eq("an unseen string is counted zero", stats.repetition("NEVER SEEN"), 0.0)
    check("repeated outscores hapax",
          C.credibility("TORQUE TABLE", stats).repetition
          > C.credibility("ODDMENT", stats).repetition)
    check("counting ignores case and spacing",
          stats.repetition("torque   table") == 3.0)


def test_deterministic():
    """Same input and same statistics, same score. Every time, in any order."""
    import ocr_credibility as C

    corpus = _corpus()
    if not check("corpus available", len(corpus) > 100):
        return

    a = C.build_statistics((t, 1.0) for t in corpus)
    b = C.build_statistics((t, 1.0) for t in corpus)
    for text in REAL + GARBAGE:
        check(f"stable across rebuilds: {text[:30]!r}",
              C.credibility(text, a).total == C.credibility(text, b).total)

    # Insertion order must not change the answer: the statistics are counts, not a
    # sequence, and a threaded batch will not arrive in a fixed order.
    c = C.build_statistics((t, 1.0) for t in reversed(corpus))
    for text in REAL + GARBAGE:
        check(f"stable across input order: {text[:30]!r}",
              C.credibility(text, a).total == C.credibility(text, c).total)

    # Repeated scoring of the same string is stable (no mutation while reading).
    first = C.credibility(REAL[0], a).total
    for _ in range(5):
        C.credibility(GARBAGE[0], a)
    check("scoring does not mutate the statistics",
          C.credibility(REAL[0], a).total == first)


def test_learns_only_from_trusted_readings():
    """The circularity mitigation: low-confidence and disputed text teaches nothing.

    Without this gate a systematic error would train the model that judges it.
    """
    import ocr_credibility as C

    stats = C.BatchStatistics()
    check("a high-confidence agreed reading is learned",
          stats.observe("PART NUMBER", confidence=0.99, agreed=True))
    check("a low-confidence reading is not learned",
          not stats.observe("pite oe ovvsovse", confidence=0.40, agreed=True))
    check("a disputed reading is not learned",
          not stats.observe("pote hivme", confidence=0.99, agreed=False))
    check("an empty reading is not learned", not stats.observe("   ", 1.0, True))
    check_eq("only the trusted reading was counted", stats.n_strings, 1)
    check_eq("garbage never entered the lexicon", stats.token_support("ovvsovse"), 0)


if __name__ == "__main__":
    sys.exit(main(sys.modules[__name__]))
