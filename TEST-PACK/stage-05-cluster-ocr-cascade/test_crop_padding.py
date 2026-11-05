"""Cell-crop margin: engine-aware padding (Phase 3).

The legacy margin (5-8% white) exists because PP-OCRv6 runs its own DB detector
inside .ocr() and DB misses text touching a border. The cascade detects first and
recognises tight sub-crops, so it needs only a token margin - and a large one is
actively harmful, since PARSeq hallucinates on loosely-cropped words.
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "workers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cluster_ocr as co  # noqa: E402


NMA = co.NUMERIC_MIN_AREA


def test_active_engine_defaults_to_paddle(monkeypatch):
    monkeypatch.delenv("OCULI_OCR_ENGINE", raising=False)
    assert co.active_engine() == "paddle"


def test_active_engine_kill_switch(monkeypatch):
    monkeypatch.setenv("OCULI_OCR_ENGINE", "cascade")
    assert co.active_engine() == "cascade"
    monkeypatch.setenv("OCULI_OCR_ENGINE", "paddle")
    assert co.active_engine() == "paddle"
    # anything unrecognised must fall back to the safe default, not crash
    monkeypatch.setenv("OCULI_OCR_ENGINE", "banana")
    assert co.active_engine() == "paddle"


# ------------------------------------------------------------------- legacy ---

# The small-crop branch triggers on AREA < 2*numeric_min_area (=800), not on side
# length: 40x40 is 1600px so it takes the large branch despite being a small cell.
@pytest.mark.parametrize("h,w,expected", [
    (20, 20, 15),      # area 400 < 800 -> small branch, 8% of 20 = 1.6, floor 15 wins
    (40, 40, 10),      # area 1600 >= 800 -> large branch, 5% of 40 = 2, floor 10 wins
    (400, 400, 20),    # large branch, 5% of 400
    (1000, 200, 10),   # large branch, 5% of 200
])
def test_legacy_pad_values_unchanged(h, w, expected):
    """These are the PaddleOCR-tuned values the parity gate depends on."""
    assert co._legacy_crop_pad(h, w, NMA) == expected


def test_legacy_small_crops_get_more_margin():
    small = co._legacy_crop_pad(15, 15, NMA)
    large = co._legacy_crop_pad(600, 600, NMA)
    assert small == 15 and large == 30
    # relative margin is much larger for small cells - the point of the branch
    assert small / 15 > large / 600


def test_engine_dispatch_uses_legacy_by_default(monkeypatch):
    monkeypatch.delenv("OCULI_OCR_ENGINE", raising=False)
    for h, w in ((20, 20), (400, 400), (1000, 200)):
        assert co._crop_pad_for_engine(h, w, NMA) == co._legacy_crop_pad(h, w, NMA)


# ------------------------------------------------------------------ cascade ---

def test_cascade_pad_scales_with_structure_size(monkeypatch):
    """A fixed pixel margin swamps small cells and vanishes on large ones."""
    monkeypatch.setenv("OCULI_OCR_ENGINE", "cascade")
    small = co._crop_pad_for_engine(10, 10, NMA)
    mid = co._crop_pad_for_engine(50, 50, NMA)
    large = co._crop_pad_for_engine(400, 400, NMA)
    assert small < mid < large, (small, mid, large)


def test_cascade_pad_calibration_point():
    """~50px cell -> 4px, the value the earlier bench was run at."""
    assert co.cascade_crop_pad(50, 50) == 4


def test_cascade_pad_uses_smaller_side_not_width():
    """A wide, short cell must get a margin sized to its TEXT height."""
    wide_short = co.cascade_crop_pad(10, 400)
    square_tall = co.cascade_crop_pad(400, 400)
    assert wide_short < square_tall
    assert wide_short == co.cascade_crop_pad(10, 10)


def test_cascade_pad_never_swamps_tiny_cells():
    """Measured cells go down to 3px tall; the margin must not exceed the cell."""
    for side in (2, 3, 5, 8):
        assert co.cascade_crop_pad(side, side) < side or co.cascade_crop_pad(side, side) <= 1


def test_cascade_pad_is_clamped():
    assert co.cascade_crop_pad(10_000, 10_000) == co.CASCADE_PAD_MAX_PX
    assert co.cascade_crop_pad(1, 1) >= co.CASCADE_PAD_MIN_PX


def test_cascade_pad_much_smaller_than_legacy(monkeypatch):
    """The whole point: stop inflating crops for a detector that no longer runs."""
    monkeypatch.setenv("OCULI_OCR_ENGINE", "cascade")
    for h, w in ((400, 400), (1000, 200), (20, 20)):
        assert co._crop_pad_for_engine(h, w, NMA) < co._legacy_crop_pad(h, w, NMA)


def test_cascade_pad_is_nonzero():
    """Some margin is still wanted so glyphs don't touch the border."""
    assert co.cascade_crop_pad(50, 50) > 0


def test_cascade_pad_env_override(monkeypatch):  # noqa
    monkeypatch.setenv("OCULI_CASCADE_PAD_RATIO", "0.2")
    import importlib
    importlib.reload(co)
    try:
        assert co.CASCADE_PAD_RATIO == 0.2
        # use a small cell so the MAX clamp doesn't mask the ratio change
        assert co.cascade_crop_pad(20, 20) == 4
    finally:
        monkeypatch.delenv("OCULI_CASCADE_PAD_RATIO", raising=False)
        importlib.reload(co)
