from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "workers"))

import region_ocr as ro  # noqa: E402


def _line(x0: int, y0: int, x1: int, y1: int) -> dict:
    return {"bbox": [x0, y0, x1, y1], "text": "t"}


def test_rhythm_profile_two_line_small_gap_gets_partial_bonus_cv():
    lines = [
        _line(100, 0, 500, 20),
        _line(100, 28, 490, 48),  # gap=8, line_h=20 => gap < 3*line_h
    ]
    gap, cv, median_width, gaps = ro._compute_rhythm_profile(lines)
    assert gap == pytest.approx(8.0)
    assert cv == pytest.approx(0.3)
    assert median_width > 0
    assert len(gaps) == 1


def test_rhythm_profile_two_line_large_gap_is_explicit_no_bonus():
    lines = [
        _line(100, 0, 500, 20),
        _line(100, 140, 490, 160),  # gap=120, line_h=20 => gap >= 3*line_h
    ]
    gap, cv, _, gaps = ro._compute_rhythm_profile(lines)
    assert len(gaps) == 1
    assert cv == pytest.approx(1.0)
    assert gap == pytest.approx(0.0)


def test_rhythm_profile_header_body_uses_p25_and_trimmed_cv():
    lines = [
        _line(100, 0, 400, 20),     # header
        _line(100, 100, 500, 120),  # gap 80
        _line(100, 145, 490, 165),  # gap 25
        _line(100, 190, 480, 210),  # gap 25
    ]
    gap, cv, _, gaps = ro._compute_rhythm_profile(lines)
    assert sorted(gaps) == [25.0, 25.0, 80.0]
    assert gap == pytest.approx(25.0)
    assert cv == pytest.approx(0.0, abs=1e-6)


def test_merge_overlapping_boxes_avoids_tiny_overlap_chain():
    # Tiny 1px overlaps should not create transitive single giant merge.
    boxes = [
        [0, 0, 100, 100],
        [99, 0, 199, 100],   # 1px overlap with first
        [198, 0, 298, 100],  # 1px overlap with second
    ]
    merged = ro.merge_overlapping_boxes(boxes, iou_thresh=0.5)
    assert len(merged) == 3


def test_merge_overlapping_boxes_merges_meaningful_overlap():
    boxes = [
        [0, 0, 100, 100],
        [60, 0, 160, 100],  # 40% overlap width
    ]
    merged = ro.merge_overlapping_boxes(boxes, iou_thresh=0.5)
    assert len(merged) == 1
    assert merged[0][0] == 0 and merged[0][2] == 160


def test_merge_overlapping_boxes_respects_whitespace_corridor():
    boxes = [
        [50, 100, 250, 160],
        [280, 100, 480, 160],  # clear horizontal corridor
    ]
    merged = ro.merge_overlapping_boxes(boxes, iou_thresh=0.5)
    assert len(merged) == 2


def test_group_boxes_whitespace_cuts_rejects_large_horizontal_same_band_gap():
    boxes = [
        [50, 100, 260, 160],
        [500, 106, 710, 166],  # near same-band with large horizontal whitespace gap
    ]
    grouped = ro._group_boxes_by_whitespace_cuts(boxes, page_h=1200, page_w=1000, page_id="test")
    assert len(grouped) == 2

