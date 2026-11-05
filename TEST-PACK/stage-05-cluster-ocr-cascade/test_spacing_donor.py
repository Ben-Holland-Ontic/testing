"""Donor spacing - giving three space-blind recognisers their spaces back.

PARSeq, SAR and MASTER have no space in their charsets: they are word-level scene-text
models and a space never appeared in their training labels. Rather than fight that, the
cascade decides CHARACTERS and SPACING separately -- candidates are compared space-free,
and spacing is applied once at the end from a model that has it.

CATMuS is the donor of record. The measurements that chose it over the alternatives are
recorded in ocr_supporting.py's Spacing section, and reproduced by
RESOURCES/space_recovery_probe.py.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY TESTING\\tests\\test_spacing_donor.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import add_runtime_to_path, check, check_eq, main  # noqa: E402

os.environ.setdefault("OCULI_SUPPORTING_MODEL_DIR", r"C:\OCULI.pd1\TESTING\onnx-models")
os.environ.setdefault("PADDLEOCR_USE_GPU", "1")

add_runtime_to_path()

FIX = Path(r"C:\OCULI.pd1\TESTING\RESOURCES\fixtures")
CROPS = FIX / "crops"
SPACELESS = ("parseq", "sar", "master")


def _multiword():
    import cv2
    manifest = json.loads((FIX / "crops.json").read_text(encoding="utf-8"))
    out = []
    for m in manifest:
        if " " not in m["primary_text"]:
            continue
        img = cv2.imread(str(CROPS / m["file"]))
        if img is not None:
            out.append((m, img))
    return out


def test_emits_space_is_read_from_the_charset():
    """The capability comes from the shipped metadata, not from a hardcoded list.

    If a model is ever re-converted with a different charset, this must follow it rather
    than keep asserting what used to be true.
    """
    import ocr_supporting as S

    check("catmus emits spaces", S.emits_space("catmus"))
    for name in SPACELESS:
        check(f"{name} does not emit spaces", not S.emits_space(name))

    # ...and that really is what the metadata says.
    for name in S.MODEL_NAMES:
        _, meta_path = S.model_paths(name)
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        charset = meta.get("charset") or meta.get("vocab") or "".join(
            meta.get("idx2ch", {}).values())
        check_eq(f"{name}: emits_space agrees with metadata",
                 S.emits_space(name), " " in charset)


def test_catmus_is_the_donor_of_record():
    """CATMuS outranks the primary, structurally.

    The cascade escalates because the primary looked wrong; borrowing its spacing would
    mean trusting the reading we just rejected. Getting this order wrong is silent, so
    it is asserted rather than left to call-site convention.
    """
    import ocr_supporting as S

    check_eq("CATMuS wins when both have spacing",
             S.choose_donor("A B", "C D"), "A B")
    check_eq("primary is used when CATMuS has no spacing",
             S.choose_donor("AB", "C D"), "C D")
    check_eq("primary is used when CATMuS is empty",
             S.choose_donor("", "C D"), "C D")
    check_eq("primary is used when CATMuS is None",
             S.choose_donor(None, "C D"), "C D")
    check_eq("no donor when neither has spacing", S.choose_donor("AB", "CD"), "")
    check_eq("no donor at all", S.choose_donor(), "")

    # Keyword calls must give the same answer as positional ones -- a caller reaching for
    # clarity must not silently invert the priority.
    check_eq("keyword call keeps the priority",
             S.choose_donor(primary_text="C D", catmus_text="A B"), "A B")


def test_donor_never_changes_characters():
    """Insert-only. Whatever the donor says, every character survives in order.

    This is the property that makes a wrong donor safe: the worst it can do is put a
    space in an odd place, never turn a correct reading into a wrong one.
    """
    import ocr_supporting as S

    cases = [
        ("ThrustWasherBearing", "Thrust Washer Bearing"),
        ("ThrustWasherBearing", "Thrnst Wosher Beoring"),   # donor misreads characters
        ("ThrustWasherBearing", "T W B"),                   # donor much shorter
        ("ABC", "A B C D E F G"),                           # donor much longer
        ("SolidWorksStudentEdition.", "SolidWorkes Student Edition."),
        ("X", "A B"),                                       # single character
        ("", "A B"),                                        # empty candidate
        ("ALLCAPS", "no spaces here"),
        ("12345", "1 2 3 4 5"),
    ]
    for candidate, donor in cases:
        got = S.apply_donor_spacing(candidate, donor)
        check(f"characters preserved: {candidate!r} + {donor!r} -> {got!r}",
              S.strip_spaces(got) == S.strip_spaces(candidate),
              f"got {S.strip_spaces(got)!r}")
        check(f"no leading/trailing space: {got!r}", got == got.strip() or not got)


def test_donor_is_a_noop_when_it_should_be():
    import ocr_supporting as S

    check_eq("donor without spaces leaves the candidate alone",
             S.apply_donor_spacing("ThrustWasherBearing", "ThrustWasherBearing"),
             "ThrustWasherBearing")
    check_eq("empty donor leaves the candidate alone",
             S.apply_donor_spacing("ABC", ""), "ABC")
    check_eq("a candidate that already has spaces is untouched",
             S.apply_donor_spacing("Thrust Washer Bearing", "A B C D"),
             "Thrust Washer Bearing")
    check_eq("empty candidate stays empty", S.apply_donor_spacing("", "A B"), "")


def test_donor_recovers_spaces_on_real_crops():
    """End to end: the three space-blind models, spaced by CATMuS, on real crops."""
    import ocr_supporting as S

    items = _multiword()
    if not check("multi-word fixture crops present", bool(items), f"{len(items)}"):
        return

    catmus = S.get_recogniser("catmus")
    if not check("catmus loaded", catmus is not None):
        return
    catmus_reads = [t for t, _ in catmus.recognise([img for _, img in items])]

    donors_available = sum(1 for t in catmus_reads if " " in t)
    check(f"CATMuS supplies a donor for {donors_available}/{len(items)} crops",
          donors_available == len(items),
          "" if donors_available == len(items) else "some crops fall back to the primary")

    for name in SPACELESS:
        rec = S.get_recogniser(name)
        if rec is None:
            check(f"{name}: loaded", False)
            continue
        raw = [t for t, _ in rec.recognise([img for _, img in items])]

        # Before: not one of them can produce the reference, because none has a space.
        before = sum(1 for r, (m, _) in zip(raw, items) if r == m["primary_text"])
        check_eq(f"{name}: exact matches WITHOUT a donor", before, 0)

        after = 0
        for r, c, (m, _) in zip(raw, catmus_reads, items):
            donor = S.choose_donor(c, m["primary_text"])
            spaced = S.apply_donor_spacing(r, donor)
            check(f"{name}: {m['file']} characters preserved",
                  S.strip_spaces(spaced) == S.strip_spaces(r))
            if spaced == m["primary_text"]:
                after += 1
        check(f"{name}: donor recovers {after}/{len(items)} exactly", after >= before,
              f"{before} -> {after}")
        if name == "parseq":
            check_eq("parseq: donor recovers every multi-word crop", after, len(items))


def test_space_free_comparison_makes_models_comparable():
    """Candidates differing only in spacing must compare EQUAL for voting.

    Without this every multi-word crop registers a spurious disagreement between the
    space-bearing models and the space-blind ones, and the vote is decided by an artefact
    of the charsets rather than by what the models read.
    """
    import ocr_supporting as S

    primary = "Thrust Washer Bearing"
    parseq = "ThrustWasherBearing"
    check("space-free forms are equal",
          S.strip_spaces(primary) == S.strip_spaces(parseq))
    check("raw forms are not equal (so the normalisation is load-bearing)",
          primary != parseq)
    check_eq("multiple/odd whitespace collapses too",
             S.strip_spaces("  Thrust  Washer\tBearing "), "ThrustWasherBearing")
    check("a genuine character disagreement still shows up",
          S.strip_spaces("Thrust Washer Bearing") != S.strip_spaces("Thrnst Wosher Bearing"))


if __name__ == "__main__":
    sys.exit(main(sys.modules[__name__]))
