"""Phase 0 - the engine-selection seam is extracted without changing a decision.

Phase 0 of the multi-engine plan is a pure refactor: the inline grouping loop in
`_run_paddleocr_on_crops_impl` becomes a named function, `_select_engine_groups`,
so the routing decision has somewhere to live and something to test. Nothing else
moves and no output changes.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY TESTING\\tests\\test_phase0_routing.py
"""
from __future__ import annotations

import ast
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import BK, add_runtime_to_path, check, check_eq, main  # noqa: E402

add_runtime_to_path()

PARITY_DIR = Path(r"C:\OCULI.pd1\TESTING\RESULTS\parity")
BASELINE_DIGEST = PARITY_DIR / "phase0_baseline.json"
CANDIDATE_DIGEST = PARITY_DIR / "phase0_candidate.json"


# ---------------------------------------------------------------------------------
# Frozen copy of the pre-refactor loop. This is the golden reference: the extracted
# function must reproduce it exactly, so the reference must NOT be refactored to
# share code with it. Copied verbatim from cluster_ocr.py before Phase 0.
# ---------------------------------------------------------------------------------
def _reference_groups(n_imgs, labels, ocr_default, ocr_handwriting, batch_size):
    use_labels = labels is not None and len(labels) == n_imgs
    groups = []
    i = 0
    while i < n_imgs:
        use_hw = use_labels and labels[i] == "handwriting" and ocr_handwriting is not None
        ocr = ocr_handwriting if use_hw else ocr_default
        group = [i]
        i += 1
        while i < n_imgs and len(group) < batch_size:
            next_hw = use_labels and labels[i] == "handwriting" and ocr_handwriting is not None
            if (ocr_handwriting if next_hw else ocr_default) is not ocr:
                break
            group.append(i)
            i += 1
        groups.append((use_hw, group))
    return groups


# Labels the pre-classifier actually emits. Only "handwriting" steers the router,
# but the others must be exercised so a mistyped comparison would show up.
LABELS = ["normal", "handwriting", "numeric", "small", "blurry"]


def _label_cases():
    """~50 label vectors: hand-picked edges first, then deterministic random."""
    cases = [
        None,                                  # labels omitted entirely
        [],                                    # empty
        ["normal"],
        ["handwriting"],
        ["handwriting"] * 9,                   # one run longer than batch_size
        ["normal"] * 9,
        ["handwriting", "normal"],
        ["normal", "handwriting"],
        ["handwriting", "normal"] * 6,         # alternating: worst case for grouping
        ["normal"] * 4 + ["handwriting"] * 4 + ["normal"] * 4,
        ["numeric", "small", "blurry", "handwriting", "normal"],
        ["HANDWRITING"],                       # case sensitivity is load-bearing
        ["hand writing", "handwriting "],      # near-miss labels must NOT route to hw
        [None, "handwriting", None],
    ]
    rng = random.Random(20260911)
    while len(cases) < 50:
        n = rng.randint(1, 40)
        cases.append([rng.choice(LABELS) for _ in range(n)])
    return cases


def test_routing_golden():
    import cluster_ocr

    if not check("function exists", hasattr(cluster_ocr, "_select_engine_groups")):
        return
    select = cluster_ocr._select_engine_groups

    # Stand-ins for the engines. Identity is all the routing logic looks at, so
    # sentinels exercise the real comparison without building a model.
    default = object()
    handwriting = object()

    mismatches = []
    cases = 0
    for labels in _label_cases():
        n = 0 if labels is None else len(labels)
        # Both engine configurations: handwriting present, and handwriting absent
        # (the model failed to load), which must collapse every crop onto default.
        for hw in (handwriting, None):
            for batch_size in (1, 2, 6, 8, 64):
                want = _reference_groups(n, labels, default, hw, batch_size)
                got = select(n, labels, default, hw, batch_size)
                cases += 1
                if got != want:
                    mismatches.append((labels, hw is not None, batch_size, got, want))

    check(
        f"decisions match the pre-refactor loop over {cases} configurations",
        not mismatches,
        "" if not mismatches else f"{len(mismatches)} mismatch(es), first={mismatches[0]}",
    )

    # Structural invariants the caller relies on, checked independently of the
    # reference so a bug copied into both would still be caught.
    labels = ["normal", "handwriting"] * 10
    groups = select(len(labels), labels, default, handwriting, 6)
    flat = [i for _, g in groups for i in g]
    check_eq("every crop appears exactly once, in order", flat, list(range(len(labels))))
    check("no group exceeds batch_size", all(len(g) <= 6 for _, g in groups))
    check("no group is empty", all(len(g) > 0 for _, g in groups))
    check(
        "use_hw agrees with the labels in its group",
        all(all((labels[i] == "handwriting") == use_hw for i in g) for use_hw, g in groups),
    )

    # A crop labelled handwriting must never be routed to the handwriting engine
    # when that engine is absent.
    groups = select(4, ["handwriting"] * 4, default, None, 8)
    check("handwriting engine absent => no hw group", all(not use_hw for use_hw, _ in groups))


def test_both_paths_covered():
    """region_ocr must still reach OCR through cluster_ocr's shared dispatch.

    The plan hooks `_run_paddleocr_on_crops`; region OCR gets the cascade for free
    only because it imports that exact symbol. If the import is ever refactored
    away, region OCR would silently keep the old single-engine behaviour and
    nothing else in the suite would notice.
    """
    region_src = Path(BK, "workers", "region_ocr.py").read_text(encoding="utf-8")
    tree = ast.parse(region_src)

    importers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "cluster_ocr":
            if "_run_paddleocr_on_crops" in {a.name for a in node.names}:
                importers.append(node.lineno)
    check(
        "region_ocr imports _run_paddleocr_on_crops from cluster_ocr",
        bool(importers),
        f"line(s) {importers}" if importers else "import not found",
    )

    # ...and actually calls it, rather than importing it and using something else.
    calls = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_run_paddleocr_on_crops"
    ]
    check("region_ocr calls it", bool(calls), f"line(s) {calls}")

    import cluster_ocr
    check("dispatch is present on cluster_ocr", hasattr(cluster_ocr, "_run_paddleocr_on_crops"))
    check("impl is present on cluster_ocr", hasattr(cluster_ocr, "_run_paddleocr_on_crops_impl"))

    # The dispatch must stay a single funnel: one function the cascade can hook.
    cluster_tree = ast.parse(Path(BK, "workers", "cluster_ocr.py").read_text(encoding="utf-8"))
    fn = next(
        (n for n in ast.walk(cluster_tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_run_paddleocr_on_crops"),
        None,
    )
    if check("dispatch function found in source", fn is not None):
        forwards = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_run_paddleocr_on_crops_impl"
        ]
        check("dispatch forwards to impl (the hook point stays one function)", bool(forwards))


def _stable(entries):
    """Entries minus the fields two identical runs of the same build disagree on."""
    return [{k: v for k, v in e.items() if k != "unstable_table_label"} for e in entries]


def test_pipeline_output_identical():
    """Full 8-image extract, byte-equal to the pre-refactor build.

    Both digests come from RESOURCES/extract_parity_digest.py. Regenerate the
    candidate after any change to the OCR path -- see that file's docstring for
    the exact commands.
    """
    if not check("baseline digest exists", BASELINE_DIGEST.is_file(), str(BASELINE_DIGEST)):
        return
    if not check("candidate digest exists", CANDIDATE_DIGEST.is_file(), str(CANDIDATE_DIGEST)):
        return
    base = json.loads(BASELINE_DIGEST.read_text(encoding="utf-8"))
    cand = json.loads(CANDIDATE_DIGEST.read_text(encoding="utf-8"))

    check_eq("document count", cand.get("document_count"), base.get("document_count"))

    by_base = {d["relative_path"]: d for d in base["documents"]}
    by_cand = {d["relative_path"]: d for d in cand["documents"]}
    check_eq("same documents", sorted(by_cand), sorted(by_base))

    for key in sorted(set(by_base) & set(by_cand)):
        b, c = by_base[key], by_cand[key]
        check_eq(f"{key}: unmatched_text line count",
                 c["unmatched_total_lines"], b["unmatched_total_lines"])
        check(f"{key}: unmatched_text lines",
              c["unmatched_lines"] == b["unmatched_lines"],
              "" if c["unmatched_lines"] == b["unmatched_lines"] else "content differs")
        # TABLE entries carry a representative query_name/anchor_bbox that two
        # identical runs disagree on (measured; see extract_parity_digest.py).
        # Compare the stable view, so the test tracks recognised text rather than
        # a pre-existing tie-break.
        cs, bs = _stable(c["summary_entries"]), _stable(b["summary_entries"])
        check(f"{key}: summary entries", cs == bs,
              "" if cs == bs else "content differs")
        check_eq(f"{key}: page statuses", c["pages"], b["pages"])

    check_eq("whole-run sha256", cand.get("sha256"), base.get("sha256"))


if __name__ == "__main__":
    sys.exit(main(sys.modules[__name__]))
