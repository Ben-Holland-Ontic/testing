"""Phase 3 - ROVER voting and the decision policy.

Votes are per position, so consensus can produce a string no single model returned. NULL
is a first-class vote so an omission registers as a deletion rather than as agreement.
Spaces and case are not voted on (three of four supporting models cannot emit a space at
all) -- characters are voted, spacing is re-applied from a donor afterwards.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY TESTING\\tests\\test_phase3_voting.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import add_runtime_to_path, check, check_eq, main  # noqa: E402

add_runtime_to_path()


def C(model, text, conf=0.0):
    import ocr_voting as V
    return V.Candidate(model, text, conf)


def test_alignment_basic():
    import ocr_voting as V

    slots = V.build_network([C("primary", "CAT"), C("parseq", "CAT"), C("sar", "CAT")])
    check_eq("identical candidates give one slot per character", len(slots), 3)
    check("every model votes in every slot",
          all(set(s.votes) == {"primary", "parseq", "sar"} for s in slots))
    check_eq("slot characters keep their ORIGINAL case",
             "".join(s.votes["primary"] for s in slots), "CAT")

    # A substitution aligns rather than creating extra slots.
    slots = V.build_network([C("primary", "CAT"), C("parseq", "COT"), C("sar", "CAT")])
    check_eq("a substitution keeps the slot count", len(slots), 3)
    check_eq("the disputed slot holds both characters",
             {slots[1].votes["primary"], slots[1].votes["parseq"]}, {"A", "O"})

    # An insertion creates a slot, and earlier models vote NULL in it.
    slots = V.build_network([C("primary", "CAT"), C("parseq", "CAAT")])
    check_eq("an insertion adds a slot", len(slots), 4)
    nulls = [s for s in slots if s.votes.get("primary") is None]
    check_eq("the primary votes NULL in exactly the inserted slot", len(nulls), 1)

    check_eq("no candidates gives no slots", V.build_network([]), [])
    check_eq("empty strings are ignored", V.build_network([C("primary", "")]), [])


def test_null_is_a_vote():
    """An omitted character must register as a deletion, not as agreement.

    Without this a model that simply stopped early would appear to agree with every
    character it never read.
    """
    import ocr_voting as V

    # 'ABC' vs 'BC': the leading 'A' is missing from one candidate.
    slots = V.build_network([C("primary", "ABC"), C("parseq", "BC")])
    check_eq("the union of characters is covered", len(slots), 3)
    check_eq("the omitting model votes NULL in the first slot",
             slots[0].votes["parseq"], None)
    check_eq("...and the primary votes its character there",
             slots[0].votes["primary"], "A")

    # With two models against one, NULL loses and the character survives.
    r = V.vote([C("primary", "ABC"), C("parseq", "ABC"), C("sar", "BC")])
    check_eq("a character two models read survives one omission", r.text, "ABC")

    # With two models omitting, NULL wins -- a deletion is a real outcome.
    r = V.vote([C("primary", "ABC"), C("parseq", "BC"), C("catmus", "BC")])
    check_eq("a character two models omit is deleted", r.text, "BC")

    check("NULL being a vote is visible in the slots",
          any(ch is None for s in V.build_network(
              [C("primary", "ABC"), C("parseq", "BC")]) for ch in s.votes.values()))


def test_consensus_beyond_any_input():
    """The point of voting per position: a correct string no single model returned."""
    import ocr_voting as V

    # Each model gets a different character wrong; the majority is right at every slot.
    r = V.vote([
        C("primary", "XBCD"),
        C("parseq", "AXCD"),
        C("sar", "ABXD"),
        C("catmus", "ABCD"),
    ])
    check_eq("the vote reconstructs a string most candidates got wrong", r.text, "ABCD")
    check("...which three of the four never produced",
          sum(1 for t in ("XBCD", "AXCD", "ABXD") if t == r.text) == 0)

    # A real shape: one model right on the front, another on the back.
    r = V.vote([
        C("primary", "TORQUX TABLX"),
        C("parseq", "TORQUE TABLE"),
        C("sar", "TXRQUE TABLE"),
    ])
    check_eq("front and back are recovered from different models",
             r.text, "TORQUETABLE")


def test_alpha_extremes():
    import ocr_voting as V

    # Two models say 'A' with low confidence; one says 'B' with high confidence.
    cands = [C("primary", "A", 0.10), C("parseq", "A", 0.10), C("catmus", "B", 0.99)]
    check_eq("alpha=1.0 ignores confidence: frequency wins",
             V.vote(cands, alpha=1.0).text, "A")
    check_eq("alpha=0.0 reduces to confidence: the confident minority wins",
             V.vote(cands, alpha=0.0).text, "B")
    check_eq("the default is frequency-only", V.DEFAULT_ALPHA, 1.0)

    # ...and the default really behaves like alpha=1.0.
    check_eq("default matches alpha=1.0",
             V.vote(cands).text, V.vote(cands, alpha=1.0).text)


def test_independence_weighting():
    """Correlated models must not be able to outvote independent ones by duplication.

    SAR and MASTER are both doctr attention decoders over an identical 126-character
    vocabulary, so their agreement is largely the same evidence twice.
    """
    import ocr_voting as V

    w = V.model_weights(["primary", "catmus", "sar", "master"])
    check_eq("sar and master share one vote", (w["sar"], w["master"]), (0.5, 0.5))
    check_eq("primary keeps a full vote", w["primary"], 1.0)
    check_eq("catmus keeps a full vote", w["catmus"], 1.0)
    check("sar+master together weigh less than primary+catmus",
          w["sar"] + w["master"] < w["primary"] + w["catmus"])

    # A model alone in its group is not penalised for an absent sibling.
    w2 = V.model_weights(["primary", "sar"])
    check_eq("sar alone gets a full vote", w2["sar"], 1.0)

    # The behavioural consequence: the correlated pair cannot overturn the independent pair.
    r = V.vote([C("primary", "A"), C("catmus", "A"), C("sar", "B"), C("master", "B")])
    check_eq("primary+catmus (2.0) beat sar+master (1.0)", r.text, "A")


def test_tie_keeps_primary():
    import ocr_voting as V

    r = V.vote([C("primary", "A"), C("parseq", "B")])
    check_eq("a one-all tie keeps the primary's character", r.text, "A")

    r = V.vote([C("primary", "AB"), C("parseq", "AC"), C("catmus", "AB"),
                C("sar", "AC"), C("master", "AC")])
    # primary+catmus = 2.0 for 'b'; parseq 1.0 + sar/master 1.0 = 2.0 for 'c'. Exact tie.
    check_eq("an exact weighted tie keeps the primary", r.text, "AB")

    d = V.decide([C("primary", "AB"), C("parseq", "AC"), C("catmus", "AB"),
                  C("sar", "AC"), C("master", "AC")])
    check("...and the decision reports no change", not d.changed, d.reason)


def test_votes_are_space_free():
    """Candidates differing only in spacing or case must register as AGREEMENT."""
    import ocr_voting as V

    r = V.vote([C("primary", "Thrust Washer Bearing"),
                C("parseq", "ThrustWasherBearing"),
                C("sar", "ThrustWasherBearing")])
    check("spacing differences are not disagreements", r.unanimous, r.detail)
    check_eq("the vote is over characters only", r.text, "ThrustWasherBearing")

    r = V.vote([C("primary", "PLANNER"), C("catmus", "planner"), C("sar", "PLANNER")])
    check("case differences are not disagreements", r.unanimous, r.detail)

    # A genuine character disagreement still shows up.
    r = V.vote([C("primary", "Thrust Washer Bearing"),
                C("parseq", "ThrnstWasherBearing"),
                C("sar", "ThrustWasherBearing")])
    check("a real character difference is still a disagreement", not r.unanimous)
    check_eq("...and the majority character wins", r.text, "ThrustWasherBearing")


def test_donor_spacing_is_applied_to_the_winner():
    import ocr_voting as V

    # The primary is wrong by one character; three models agree on the correction, so
    # the vote changes it -- and the winner comes back with the donor's spacing.
    d = V.decide([C("primary", "ThrustWasherBearinq"),
                  C("parseq", "ThrustWasherBearing"),
                  C("sar", "ThrustWasherBearing"),
                  C("catmus", "Thrust Washer Bearing")],
                 donor="Thrust Washer Bearing")
    check("the vote changed the primary", d.changed, d.reason)
    check_eq("the winning characters come back spaced",
             d.text.casefold(), "thrust washer bearing")
    check("the characters themselves are unchanged",
          d.text.replace(" ", "").casefold() == "thrustwasherbearing")

    # A primary the vote agrees with is returned untouched -- donor spacing is for the
    # winner, not a licence to rewrite the primary's own formatting.
    d = V.decide([C("primary", "ThrustWasherBearing"),
                  C("parseq", "ThrustWasherBearing"),
                  C("sar", "ThrustWasherBearing")],
                 donor="Thrust Washer Bearing")
    check_eq("an agreeing primary is left exactly as it was",
             d.text, "ThrustWasherBearing")


def test_empty_primary_is_not_recovered():
    """The empty-primary path is OFF, and that is a measured decision, not caution.

    On 1083 real crops, 259 were empty with ink present. Voting on them produced 105
    strings of which TWO were real text ('41.0', '10.0'); 79% were single characters
    hallucinated from table borders and leader lines -- three models "agreeing" on 'L'
    because a right angle looks like an L. An empty primary means the detector found no
    line region, so the supporting models can only be shown the whole crop, which is the
    configuration measured to be worst.
    """
    import ocr_voting as V

    d = V.decide([C("primary", ""), C("parseq", "TORQUE"), C("sar", "TORQUE"),
                  C("master", "TORQUE")])
    check_eq("nothing is promoted from an empty primary", d.text, "")
    check("...reported as no change", not d.changed)
    check("...and the reason says why", "recovery disabled" in d.reason, d.reason)

    # The mechanism still exists for re-running the experiment, explicitly.
    d = V.decide([C("primary", ""), C("parseq", "TORQUE"), C("sar", "TORQUE"),
                  C("master", "TORQUE")], recover_empty_primary=True)
    check_eq("the flag re-enables it for experiments", d.text.casefold(), "torque")
    check("...and reports the change", d.changed, d.reason)

    # Even then, a single uncorroborated voice is not promoted.
    d = V.decide([C("primary", ""), C("parseq", "TORQUE")],
                 recover_empty_primary=True)
    check_eq("one voice alone still promotes nothing", d.text, "")


def test_blend_when_candidates_agree():
    """Close candidates are blended per position -- this is where ROVER's gain lives."""
    import ocr_voting as V

    cands = [C("primary", "XBCD"), C("parseq", "AXCD"),
             C("sar", "ABXD"), C("catmus", "ABCD")]
    r = V.vote(cands)
    check(f"divergence is low ({r.divergence:.2f})",
          r.divergence <= V.MAX_BLEND_DIVERGENCE, f"{r.divergence:.2f}")
    d = V.decide(cands)
    check_eq("the blend reconstructs a string most candidates got wrong",
             d.text.casefold(), "abcd")
    check("...and says it blended", "blend" in d.reason, d.reason)


def test_select_when_candidates_diverge():
    """Far-apart candidates are SELECTED whole, never blended.

    MEASURED: PARSeq read 'PROJECTION' correctly on the THIRD ANGLE PROJECTION symbol,
    the primary read 'THRO CTIOLE S', and per-character blending produced 'proctions' --
    a chimera worse than the best single candidate. Blending assumes the candidates are
    variations on one string; when they are not, it invents.
    """
    import ocr_voting as V

    cands = [C("primary", "THRO CTIOLE S"), C("catmus", " S"),
             C("parseq", "PROJECTION"), C("sar", "PASSENGE"),
             C("master", "PERRECHENE")]
    r = V.vote(cands)
    check(f"divergence is high ({r.divergence:.2f})",
          r.divergence > V.MAX_BLEND_DIVERGENCE, f"{r.divergence:.2f}")

    check_eq("...and blending WOULD have produced the chimera", r.text, "PROCTIONS")

    d = V.decide(cands)
    check("select mode was used", "select" in d.reason, d.reason)
    check("the chimera is never emitted", d.text.casefold() != "proctions", d.text)
    check_eq("no candidate is corroborated, so the primary is kept",
             d.text, "THRO CTIOLE S")
    check("...and the primary was not changed", not d.changed, d.reason)


def test_select_picks_the_corroborated_candidate():
    import ocr_voting as V

    # Three models read 'DATE'; primary and catmus disagree with it and each other.
    cands = [C("primary", "DAE"), C("catmus", "DUE"), C("parseq", "DATE"),
             C("sar", "DATE"), C("master", "DATE")]
    d = V.decide(cands)
    check_eq("a corroborated whole candidate wins", d.text.casefold(), "date")
    check("...and reports the change", d.changed, d.reason)

    # parseq is one group; sar+master are ONE group between them -> 2 independent voices.
    text, support, why = V.select_best(cands)
    check_eq("selection counts independent groups", support, 2)
    check_eq("...and picks the right string", text.casefold(), "date")


def test_select_ties_go_to_the_primary():
    import ocr_voting as V

    # sar+master agree but are one group; the primary is one group. A tie.
    cands = [C("primary", "ABCD"), C("sar", "WXYZ"), C("master", "WXYZ")]
    text, support, _why = V.select_best(cands)
    check_eq("a tie on support goes to the primary", text, "ABCD")
    d = V.decide(cands)
    check_eq("...and the primary is kept", d.text, "ABCD")
    check("...with no change claimed", not d.changed)


def test_credibility_only_breaks_ties_in_select_mode():
    import ocr_voting as V

    plausible = lambda t: 5.0 if t == "TORQUE" else -5.0  # noqa: E731

    # Between two equally-supported NON-primary candidates, credibility decides.
    text, support, _ = V.select_best(
        [C("parseq", "TORQUE"), C("catmus", "TQRQUX")], credibility_of=plausible)
    check_eq("credibility breaks a tie between equals", text, "TORQUE")
    check_eq("...but support is still only one voice", support, 1)

    # The primary outranks credibility on a tie, deliberately: a correct reading is worth
    # more than a marginally more plausible-looking one.
    text, _s, _w = V.select_best(
        [C("primary", "QQQQ"), C("parseq", "TORQUE")], credibility_of=plausible)
    check_eq("the primary wins a tie even when credibility prefers the other",
             text, "QQQQ")

    # ...and one voice never displaces the primary, however credible.
    d = V.decide([C("primary", "QQQQ"), C("parseq", "TORQUE"), C("catmus", "TQRQUX")],
                 credibility_of=plausible)
    check_eq("credibility cannot promote a single voice", d.text, "QQQQ")


def test_decision_never_displaced_by_one_voice():
    import ocr_voting as V

    # Four agree with the primary, one dissents: the primary must hold.
    d = V.decide([C("primary", "Body Cap Screw"), C("parseq", "BodyCapScrew"),
                  C("sar", "BodyCapScrew"), C("master", "BodyCapScrew"),
                  C("catmus", "body Cop serew")])
    check_eq("the real Body Cap Screw case holds", d.text, "Body Cap Screw")
    check("...reported as no change", not d.changed, d.reason)

    # A lone dissenter against a lone primary is a tie, and a tie keeps the primary.
    d = V.decide([C("primary", "REV"), C("catmus", "REY")])
    check_eq("one voice cannot overturn the primary", d.text, "REV")
    check("...reported as no change", not d.changed)


def test_credibility_can_only_block_never_force():
    """Credibility is the weakest, least calibrated signal: it gets no veto in favour.

    MEASURED reason: on the 8-page corpus it scores the correct 'PLANNER' at -4.96 and
    the correct 'Body Cap Screw' at -0.75, below plausible-looking garbage. A layer that
    wrong must not be able to promote anything on its own.
    """
    import ocr_voting as V

    cands = [C("primary", "AAA"), C("parseq", "BBB"), C("sar", "BBB"),
             C("master", "BBB"), C("catmus", "BBB")]

    d = V.decide(cands)
    check_eq("without credibility, the corroborated vote wins", d.text.casefold(), "bbb")

    # Credibility dislikes the winner -> the primary is kept.
    d = V.decide(cands, credibility_of=lambda t: 0.0 if t.casefold() == "aaa" else -9.0)
    check_eq("credibility can block a change", d.text, "AAA")
    check("...and says why", "less credible" in d.reason, d.reason)

    # Credibility loves a candidate nobody corroborated -> still not promoted.
    d = V.decide([C("primary", ""), C("parseq", "ZZZ")],
                 credibility_of=lambda t: 99.0)
    check_eq("credibility cannot promote an uncorroborated candidate", d.text, "")

    # "No opinion" must behave as no evidence, not as a low score.
    d = V.decide(cands, credibility_of=lambda t: None)
    check_eq("no opinion does not block", d.text.casefold(), "bbb")


def test_deterministic():
    import ocr_voting as V

    cands = [C("primary", "ABCD", 0.5), C("parseq", "ABXD", 0.7),
             C("sar", "ABCD", 0.6), C("master", "AXCD", 0.9)]
    first = V.vote(cands).text
    for _ in range(10):
        check_eq("repeat votes agree", V.vote(cands).text, first)

    # Candidate order must not change the answer: the dispatch is threaded and results
    # do not arrive in a fixed order.
    for perm in ([3, 1, 0, 2], [2, 0, 3, 1], [1, 3, 2, 0]):
        reordered = [cands[i] for i in perm]
        check_eq(f"order {perm} gives the same answer", V.vote(reordered).text, first)


if __name__ == "__main__":
    sys.exit(main(sys.modules[__name__]))


def test_case_is_folded_for_the_vote_but_restored_for_the_answer():
    """Case must never be an output of the vote, only an input to the comparison.

    REGRESSION TEST. The blend used to return its own comparison key, so
    'FREEDOM OF INFORMATION ACT (5 USC 552) AND' came back as
    'freedomofinformationact(5usc552)and' -- every capital destroyed. Selection preserved
    case (it returns a candidate verbatim) so the unit tests passed; only real corpus text
    exposed it. CATMuS lowercases because of its training set, which is not a disagreement
    about which letters are present.
    """
    import ocr_voting as V

    r = V.vote([C("primary", "DATE"), C("catmus", "date"), C("sar", "DATE")])
    check("case differences are not disagreements", r.unanimous, r.detail)
    check_eq("the answer keeps the primary's case", r.text, "DATE")

    # The primary's spelling wins the case even when it is outvoted on count.
    r = V.vote([C("primary", "Date"), C("catmus", "date"), C("sar", "date")])
    check_eq("the primary supplies the spelling", r.text, "Date")

    # A blended correction keeps the case of the models that supplied each character.
    d = V.decide([C("primary", "DAE", 0.5), C("catmus", "DUE"), C("parseq", "DATE"),
                  C("sar", "DATE"), C("master", "DATE")])
    check_eq("a corrected word is not lowercased", d.text, "DATE")

    # Real shape, from the corpus that exposed the bug.
    long_text = "FREEDOM OF INFORMATION ACT (5 USC 552) AND"
    d = V.decide([C("primary", long_text, 0.5),
                  C("parseq", "FREEDOMOFINFORMATIONACT(5USC552)AND"),
                  C("sar", "FREEDOMOFINFORMATIONACT(5USC552)AND")],
                 donor=long_text)
    check_eq("a long confident line survives intact", d.text, long_text)


def test_gates_keep_the_vote_away_from_what_it_cannot_help():
    """Two fitted gates, both measured on 941 real line regions."""
    import ocr_voting as V

    # A confident primary is not second-guessed: that is where the damage was.
    d = V.decide([C("primary", "AISI Type 316L", 0.99), C("parseq", "ALSIType316L"),
                  C("sar", "ALSIType316L"), C("master", "ALSIType316L")])
    check_eq("a confident primary is left alone", d.text, "AISI Type 316L")
    check("...and says why", "confident" in d.reason, d.reason)

    # ...and the same candidates DO change it once the primary is unsure.
    d = V.decide([C("primary", "AISI Type 316L", 0.40), C("parseq", "ALSIType316L"),
                  C("sar", "ALSIType316L"), C("master", "ALSIType316L")],
                 donor="AISI Type 316L")
    check("an unsure primary is open to correction", d.changed, d.reason)

    # A string too short to carry signal is never voted on.
    d = V.decide([C("primary", "N", 0.5), C("parseq", "2"), C("sar", "2"),
                  C("master", "2"), C("catmus", "2")])
    check_eq("a single character is left alone", d.text, "N")
    check("...and says why", "too short" in d.reason, d.reason)

    check_eq("the fitted gates are where measurement put them",
             (V.MAX_PRIMARY_CONFIDENCE, V.MIN_VOTABLE_CHARS, V.MAX_BLEND_DIVERGENCE),
             (0.85, 3, 0.20))
