"""Phase 4 - the cascade, wired into the dispatch behind a default-off flag.

The cascade runs as a post-pass over the primary's results, inside the one dispatch both
cluster OCR and region OCR go through. It only ever sees lines the primary already
detected, which is what lets it show the supporting models a LINE REGION rather than a
whole crop -- measured as the difference between 17% and 65% exact for PARSeq.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY TESTING\\tests\\test_phase4_cascade.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import BK, add_runtime_to_path, check, check_eq, main  # noqa: E402

os.environ.setdefault("OCULI_SUPPORTING_MODEL_DIR", r"C:\OCULI.pd1\TESTING\onnx-models")
os.environ.setdefault("PADDLEOCR_USE_GPU", "1")

add_runtime_to_path()

FIX = Path(r"C:\OCULI.pd1\TESTING\RESOURCES\fixtures")


def _fake_line(text, conf, bbox):
    return {"text": text, "bbox": bbox, "rect_index": 0, "poly_id": "P", "confidence": conf}


def _crop(name):
    import cv2
    return cv2.imread(str(FIX / "crops" / name))


def test_flag_off_is_inert():
    """With the flag off the cascade must be a no-op, not a cheap path.

    This is what makes the whole feature safe to ship dark: byte-identical output, no
    model loaded, no GPU touched.
    """
    import ocr_cascade

    saved = os.environ.get("OCULI_OCR_CASCADE")
    try:
        os.environ["OCULI_OCR_CASCADE"] = "0"
        check("explicitly disabled by '0'", not ocr_cascade.enabled())

        img = _crop("01_original_03.png")
        if img is None:
            check("fixture crop present", False)
            return
        results = [[_fake_line("Honeywell", 0.10, [0, 0, img.shape[1], img.shape[0]])]]
        before = json.dumps(results, sort_keys=True)
        out = ocr_cascade.refine([img], [(0, 0, 0, "P")], results)
        check_eq("results are returned untouched", json.dumps(out, sort_keys=True), before)
        check("no cascade provenance was added", "cascade" not in out[0][0])
        check_eq("prewarm does nothing when disabled", ocr_cascade.prewarm(), {})

        for value in ("", "no", "off", "false"):
            os.environ["OCULI_OCR_CASCADE"] = value
            check(f"{value!r} leaves it disabled", not ocr_cascade.enabled())
        os.environ["OCULI_OCR_CASCADE"] = "1"
        check("'1' enables it", ocr_cascade.enabled())
        # ...and it is ON when nothing is set at all.
        os.environ.pop("OCULI_OCR_CASCADE", None)
        check("ON by default when the variable is unset", ocr_cascade.enabled())
    finally:
        if saved is None:
            os.environ.pop("OCULI_OCR_CASCADE", None)
        else:
            os.environ["OCULI_OCR_CASCADE"] = saved


def test_both_paths_still_covered():
    """region_ocr reaches the cascade because it shares the dispatch. Asserted, because
    if that import is ever refactored away region OCR loses the feature silently."""
    import ast
    src = Path(BK, "workers", "region_ocr.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imports = [n.lineno for n in ast.walk(tree)
               if isinstance(n, ast.ImportFrom) and n.module == "cluster_ocr"
               and "_run_paddleocr_on_crops" in {a.name for a in n.names}]
    check("region_ocr still imports the shared dispatch", bool(imports), str(imports))

    cluster = Path(BK, "workers", "cluster_ocr.py").read_text(encoding="utf-8")
    check("the dispatch calls the cascade", "ocr_cascade.refine" in cluster)
    check("...inside _run_paddleocr_on_crops_impl",
          cluster.index("ocr_cascade.refine")
          > cluster.index("def _run_paddleocr_on_crops_impl"))
    check("prewarm is wired into prewarm_cluster_ocr",
          "ocr_cascade.prewarm" in cluster)


def test_escalation_gates():
    """Models must not be RUN on lines whose result could not be used anyway."""
    import ocr_cascade as K
    import ocr_voting as V

    check("a confident line is not escalated",
          not K.should_escalate("MATERIAL", 0.99))
    check("an unsure line is escalated", K.should_escalate("MATERIAL", 0.50))
    check("an empty line is never escalated", not K.should_escalate("", 0.10))
    check("a one-character line is not escalated", not K.should_escalate("N", 0.10))
    check("a two-character line is not escalated", not K.should_escalate("AB", 0.10))
    check("three characters is enough", K.should_escalate("ABC", 0.10))
    check_eq("the gates are ocr_voting's, not a second copy",
             (V.MAX_PRIMARY_CONFIDENCE, V.MIN_VOTABLE_CHARS), (0.85, 3))


def test_specialist_routing_is_geometric():
    """Tier 2 picks ONE model from measured geometry, never from an inferred category."""
    import numpy as np
    import ocr_cascade as K

    wide = np.zeros((20, 400, 3), dtype=np.uint8)     # aspect 20:1
    normal = np.zeros((30, 120, 3), dtype=np.uint8)   # aspect 4:1

    check_eq("a wide line goes to CATMuS (dynamic width is the point)",
             K.choose_specialist(wide), "catmus")
    check_eq("a normal line goes to SAR", K.choose_specialist(normal), "sar")
    # Phase 6 changed this deliberately: a handwriting-labelled line now goes to the
    # handwriting model as a VOTER, where before the label swapped the primary engine.
    check_eq("a handwriting-labelled line goes to the handwriting specialist",
             K.choose_specialist(normal, "handwriting"), K.HANDWRITING)
    check("exactly one specialist is chosen, never a list",
          isinstance(K.choose_specialist(normal), str))


def test_tier1_agreement_stops_the_cascade():
    """Agreement is the fast path: when Tier 1 agrees, no other model is constructed."""
    import ocr_cascade as K

    img = _crop("01_original_03.png")
    if img is None:
        check("fixture crop present", False)
        return

    called = []
    real_read = K._read

    def spy(model, regions):
        called.append((model, len(regions)))
        if model == K.TIER1:
            # Agree with the primary exactly.
            return [("Honeywell", 0.9)] * len(regions)
        return real_read(model, regions)

    saved = os.environ.get("OCULI_OCR_CASCADE")
    K._read = spy
    os.environ["OCULI_OCR_CASCADE"] = "1"
    try:
        counts = K.TierCounts()
        results = [[_fake_line("Honeywell", 0.50, [0, 0, img.shape[1], img.shape[0]])]]
        K.refine([img], [(0, 0, 0, "P")], results, counts=counts)
        models = [m for m, _n in called]
        check_eq("only Tier 1 ran", models, [K.TIER1])
        check_eq("tier 2 was never entered", counts.tier2, 0)
        check_eq("tier 3 was never entered", counts.tier3, 0)
        check_eq("nothing changed", counts.changed, 0)
        check_eq("the line is untouched", results[0][0]["text"], "Honeywell")
    finally:
        K._read = real_read
        if saved is None:
            os.environ.pop("OCULI_OCR_CASCADE", None)
        else:
            os.environ["OCULI_OCR_CASCADE"] = saved


def test_tier_accounting_matches_what_ran():
    """Tier counters must equal the models actually invoked, or cost is unknowable."""
    import ocr_cascade as K

    img = _crop("Ball_Valve_dwg_1_02.png")
    if img is None:
        check("fixture crop present", False)
        return

    called = []
    real_read = K._read

    def spy(model, regions):
        called.append(model)
        return [("ZZZZZZZ", 0.5)] * len(regions)      # always disagree, force all tiers

    saved = os.environ.get("OCULI_OCR_CASCADE")
    K._read = spy
    os.environ["OCULI_OCR_CASCADE"] = "1"
    try:
        counts = K.TierCounts()
        results = [[_fake_line("Thrust Washer Bearing", 0.50,
                               [0, 0, img.shape[1], img.shape[0]])]]
        K.refine([img], [(0, 0, 0, "P")], results, counts=counts)
        check_eq("one line seen", counts.seen, 1)
        check_eq("one line escalated", counts.escalated, 1)
        check_eq("tier 1 ran once", counts.tier1, 1)
        check_eq("tier 2 ran once", counts.tier2, 1)
        check("tier 1 was PARSeq, the cheapest model", called[0] == K.TIER1, str(called))
        check("no model ran more than the tiers say",
              len(called) <= 3, str(called))
        check_eq("accounting is reportable", set(counts.as_dict()),
                 {"lines", "escalated", "tier1", "tier2", "tier3", "changed"})
    finally:
        K._read = real_read
        if saved is None:
            os.environ.pop("OCULI_OCR_CASCADE", None)
        else:
            os.environ["OCULI_OCR_CASCADE"] = saved


def test_a_missing_model_never_costs_a_page():
    """Graceful degradation: a supporting model is an extra opinion, not a prerequisite."""
    import ocr_cascade as K
    import ocr_supporting as S

    saved_dir = os.environ.get("OCULI_SUPPORTING_MODEL_DIR")
    saved_flag = os.environ.get("OCULI_OCR_CASCADE")
    saved_cache = getattr(S._TLS, "cache", None)
    img = _crop("01_original_03.png")
    if img is None:
        check("fixture crop present", False)
        return
    try:
        S._TLS.cache = {}
        os.environ["OCULI_SUPPORTING_MODEL_DIR"] = os.path.join(
            os.environ.get("TEMP", "."), "oculi-no-models-at-all")
        os.environ["OCULI_OCR_CASCADE"] = "1"

        results = [[_fake_line("Honeywell", 0.50, [0, 0, img.shape[1], img.shape[0]])]]
        out = K.refine([img], [(0, 0, 0, "P")], results)
        check("refine survives every model being absent", out is results)
        check_eq("the primary's reading is kept", out[0][0]["text"], "Honeywell")
        check("nothing was silently changed", out[0][0]["text"] == "Honeywell")
        check("no provenance is stamped when no model could speak",
              "cascade" not in out[0][0])
    finally:
        if saved_dir is None:
            os.environ.pop("OCULI_SUPPORTING_MODEL_DIR", None)
        else:
            os.environ["OCULI_SUPPORTING_MODEL_DIR"] = saved_dir
        if saved_flag is None:
            os.environ.pop("OCULI_OCR_CASCADE", None)
        else:
            os.environ["OCULI_OCR_CASCADE"] = saved_flag
        S._TLS.cache = saved_cache if saved_cache is not None else {}


def test_a_change_is_never_silent():
    """Any line the cascade rewrites carries what it was and why it changed."""
    import ocr_cascade as K

    img = _crop("Ball_Valve_dwg_1_02.png")
    if img is None:
        check("fixture crop present", False)
        return

    real_read = K._read
    K._read = lambda model, regions: [("ThrustWasherBearing", 0.9)] * len(regions)
    saved = os.environ.get("OCULI_OCR_CASCADE")
    os.environ["OCULI_OCR_CASCADE"] = "1"
    try:
        results = [[_fake_line("ThrustWasherBearinq", 0.50,
                               [0, 0, img.shape[1], img.shape[0]])]]
        K.refine([img], [(0, 0, 0, "P")], results)
        line = results[0][0]
        if check("the line was changed", line["text"] != "ThrustWasherBearinq",
                 line["text"]):
            prov = line.get("cascade") or {}
            check_eq("the previous reading is recorded", prov.get("was"),
                     "ThrustWasherBearinq")
            check("the reason is recorded", bool(prov.get("reason")), str(prov.get("reason")))
            check("every candidate is recorded", bool(prov.get("candidates")))
    finally:
        K._read = real_read
        if saved is None:
            os.environ.pop("OCULI_OCR_CASCADE", None)
        else:
            os.environ["OCULI_OCR_CASCADE"] = saved


def test_provenance_recorded_on_every_escalated_line():
    """Phase 5: a line the cascade looked at carries what it saw and why it decided.

    Recorded whether or not the line CHANGED. A line deliberately left alone is evidence
    too -- it is how Phase 6's retirement gets measured, and it is what lets a decision be
    argued with after the fact instead of taken on trust.
    """
    import ocr_cascade as K

    img = _crop("Ball_Valve_dwg_1_02.png")
    if img is None:
        check("fixture crop present", False)
        return

    real_read = K._read
    saved = os.environ.get("OCULI_OCR_CASCADE")
    os.environ["OCULI_OCR_CASCADE"] = "1"
    try:
        # Case 1: everything agrees with the primary -> no change, but still recorded.
        K._read = lambda m, r: [("Thrust Washer Bearing", 0.9)] * len(r)
        results = [[_fake_line("Thrust Washer Bearing", 0.50,
                               [0, 0, img.shape[1], img.shape[0]])]]
        K.refine([img], [(0, 0, 0, "P")], results)
        prov = results[0][0].get("cascade")
        if check("an unchanged line still carries provenance", isinstance(prov, dict)):
            check("...marked as not changed", prov.get("changed") is False)
            check_eq("...with the primary as winner", prov.get("winner"), "primary")
            check_eq("...and the tier it reached", prov.get("tier"), 1)
            check("...and the candidate set", "parseq" in (prov.get("candidates") or {}))
            check("...and divergence", prov.get("divergence") is not None)
            check("...and corroboration", prov.get("corroboration") is not None)

        # Case 2: a real correction -> full attribution.
        K._read = lambda m, r: [("ThrustWasherBearing", 0.9)] * len(r)
        results = [[_fake_line("ThrustWasherBearinq", 0.50,
                               [0, 0, img.shape[1], img.shape[0]])]]
        K.refine([img], [(0, 0, 0, "P")], results)
        prov = results[0][0].get("cascade") or {}
        check("a changed line is marked changed", prov.get("changed") is True)
        check_eq("the previous reading is kept", prov.get("was"), "ThrustWasherBearinq")
        check("the winner is attributed to a model, not left blank",
              bool(prov.get("winner")), str(prov.get("winner")))
        check("the reason is recorded", bool(prov.get("reason")))
        cands = prov.get("candidates") or {}
        check("every candidate is recorded with its confidence",
              all(isinstance(v, list) and len(v) == 2 for v in cands.values()),
              str(cands))
    finally:
        K._read = real_read
        if saved is None:
            os.environ.pop("OCULI_OCR_CASCADE", None)
        else:
            os.environ["OCULI_OCR_CASCADE"] = saved


def test_ocr_meta_chosen_slots_are_populated():
    """The `chosen` / `chosen_conf` slots stop being inert placeholders."""
    import cluster_ocr

    # No cascade involvement at all -> unchanged meaning, plus a real confidence.
    lines = [{"text": "MATERIAL", "confidence": 0.97},
             {"text": "DESCRIPTION", "confidence": 0.93}]
    meta = cluster_ocr._summarise_cascade(lines)
    check_eq("untouched rectangles still say 'paddle'", meta["chosen"], "paddle")
    check_eq("...and chosen_conf stays None, so flag-off output is unchanged",
             meta["chosen_conf"], None)
    check("no cascade summary when it never ran", "cascade" not in meta)

    # Looked at, left alone.
    lines = [{"text": "MATERIAL", "confidence": 0.5,
              "cascade": {"changed": False, "tier": 1, "winner": "primary"}}]
    meta = cluster_ocr._summarise_cascade(lines)
    check_eq("a line examined but kept still reads 'paddle'", meta["chosen"], "paddle")
    check_eq("...and records that it was escalated",
             meta["cascade"]["escalated"], 1)
    check("...and NOW reports a real confidence", meta["chosen_conf"] is not None)
    check_eq("...with zero changes", meta["cascade"]["changed"], 0)

    # Changed.
    lines = [{"text": "DATE", "confidence": 0.5,
              "cascade": {"changed": True, "tier": 2, "winner": "parseq",
                          "was": "DAE", "text": "DATE", "reason": "blend: corroborated"}}]
    meta = cluster_ocr._summarise_cascade(lines)
    check_eq("a changed rectangle attributes the winner",
             meta["chosen"], "cascade:parseq")
    check_eq("...and reports the tier reached", meta["cascade"]["max_tier"], 2)
    check_eq("...and keeps the before/after", meta["cascade"]["changes"][0]["was"], "DAE")

    check_eq("empty input is handled", cluster_ocr._summarise_cascade([])["chosen"],
             "paddle")
    check_eq("...with no confidence to report",
             cluster_ocr._summarise_cascade([])["chosen_conf"], None)

    # The exact shape the pre-cascade build produced, for any rectangle it never saw.
    check_eq("flag-off ocr_meta is byte-identical to the original placeholders",
             cluster_ocr._summarise_cascade([{"text": "X", "confidence": 0.9}]),
             {"chosen": "paddle", "chosen_conf": None})


def test_phase6_router_retired_only_when_the_cascade_runs():
    """The handwriting model is DEMOTED, not removed -- and only with the flag on."""
    import cluster_ocr
    import ocr_cascade as K

    saved = os.environ.get("OCULI_OCR_CASCADE")
    try:
        os.environ["OCULI_OCR_CASCADE"] = "0"
        check("flag off: the legacy router stays", not K.retires_legacy_router())
        os.environ["OCULI_OCR_CASCADE"] = "1"
        check("flag on: the router is retired", K.retires_legacy_router())
    finally:
        if saved is None:
            os.environ.pop("OCULI_OCR_CASCADE", None)
        else:
            os.environ["OCULI_OCR_CASCADE"] = saved

    # The routing function itself is unchanged; retirement works by passing None, so the
    # legacy path is provably identical rather than merely "believed to be".
    default, hw = object(), object()
    labels = ["handwriting", "normal", "handwriting"]
    with_hw = cluster_ocr._select_engine_groups(3, labels, default, hw, 8)
    without = cluster_ocr._select_engine_groups(3, labels, default, None, 8)
    check("with the router, handwriting crops are routed away from the primary",
          any(use for use, _g in with_hw))
    check("retired, every crop goes to the general engine",
          all(not use for use, _g in without))
    check_eq("...and all crops are still covered exactly once",
             sorted(i for _u, g in without for i in g), [0, 1, 2])


def test_phase6_handwriting_is_a_voter_not_a_replacement():
    """It gets a say on the same crops -- but only one, and it cannot win alone."""
    import numpy as np
    import ocr_cascade as K
    import ocr_voting as V

    normal = np.zeros((30, 120, 3), dtype=np.uint8)
    check_eq("a handwriting-labelled line routes to the handwriting SPECIALIST",
             K.choose_specialist(normal, "handwriting"), K.HANDWRITING)
    check_eq("the specialist is one model, not a replacement engine",
             isinstance(K.choose_specialist(normal, "handwriting"), str), True)

    # It votes on exactly the same terms as every other model: full weight, its own
    # independent group, one voice toward corroboration.
    w = V.model_weights(["primary", "parseq", "sar", "master", K.HANDWRITING])
    check_eq("it carries a full vote, like parseq and the primary",
             w[K.HANDWRITING], 1.0)
    check_eq("its correlation group is declared, not defaulted",
             V.CORRELATION_GROUPS.get(K.HANDWRITING), "handwriting")
    check("it is independent of the doctr pair",
          V.CORRELATION_GROUPS[K.HANDWRITING] != V.CORRELATION_GROUPS["sar"])

    # The historical failure: the handwriting model confidently destroying real text.
    # As a lone voice it can no longer do that.
    d = V.decide([V.Candidate("primary", "Part Number", 0.50),
                  V.Candidate(K.HANDWRITING, "pote hivme", 0.99)])
    check_eq("'pote hivme' cannot overwrite 'Part Number'", d.text, "Part Number")
    check("...reported as no change", not d.changed, d.reason)

    # ...and it still counts when it is right and others agree.
    d = V.decide([V.Candidate("primary", "Pait Numbet", 0.50),
                  V.Candidate(K.HANDWRITING, "Part Number", 0.90),
                  V.Candidate("parseq", "PartNumber", 0.90)],
                 donor="Part Number")
    check("a corroborated handwriting reading does win", d.changed, d.reason)
    check_eq("...with the right text", d.text.casefold(), "part number")


def test_phase6_preclassifier_is_untouched():
    """The label still drives preprocessing. THIS is the regression that would hurt most.

    The plan is explicit: remove the ENGINE ROUTER, keep the pre-classifier. Its label
    feeds seven branch points inside _preprocess_for_ocr and persists downstream as
    ocr_label / text_kind.
    """
    import ast
    import cluster_ocr

    check("_preprocess_for_ocr still exists", hasattr(cluster_ocr, "_preprocess_for_ocr"))

    src = Path(BK, "workers", "cluster_ocr.py").read_text(encoding="utf-8")
    # The classifier is INLINE in process_rectangles rather than a named function, so it
    # is asserted by the labels it assigns. Every one of these must survive: they are what
    # _preprocess_for_ocr branches on.
    for assignment in ('label = "numeric"', 'label = "small"', 'label = "blurry"',
                       'label = "handwriting"', 'label = "normal"'):
        check(f"the pre-classifier still assigns {assignment.split(chr(34))[1]!r}",
              assignment in src)
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_preprocess_for_ocr"), None)
    if check("_preprocess_for_ocr found in source", fn is not None):
        labels_used = {n.value for n in ast.walk(fn)
                       if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        for expected in ("numeric", "small", "blurry"):
            check(f"preprocessing still branches on {expected!r}",
                  expected in labels_used)

    # The label must still reach the output.
    check("the label still reaches ocr_meta", '"label": label' in src)


def test_phase6_flag_off_path_is_untouched():
    """Nothing above changes anything while the cascade is off."""
    import cluster_ocr

    saved = os.environ.get("OCULI_OCR_CASCADE")
    try:
        os.environ["OCULI_OCR_CASCADE"] = "0"
        default, hw = object(), object()
        labels = ["handwriting", "normal", "handwriting", "handwriting"]
        groups = cluster_ocr._select_engine_groups(4, labels, default, hw, 8)
        check("the handwriting engine is still selected with the flag off",
              any(use for use, _g in groups))
        check_eq("arbitration is still enabled by default",
                 cluster_ocr._HW_ARBITRATE, True)
        check_eq("ocr_meta is still the original placeholder pair",
                 cluster_ocr._summarise_cascade([{"text": "X", "confidence": 0.9}]),
                 {"chosen": "paddle", "chosen_conf": None})
    finally:
        if saved is None:
            os.environ.pop("OCULI_OCR_CASCADE", None)
        else:
            os.environ["OCULI_OCR_CASCADE"] = saved


if __name__ == "__main__":
    sys.exit(main(sys.modules[__name__]))
