"""
Grid index consistency (Bugs 1–4): fast-path table grid, band assignment, EC4 overlap, adaptive tolerance.

Run: py -3 -m pytest tests/test_grid_index_consistency.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "cleaners"))

import relationship_resolver_v2_enhanced as rr  # noqa: E402
from data_cleaningv2 import (  # noqa: E402
    DataCleaningInputsV2,
    TableConsolidator,
    _ec4_bbox_overlaps,
)


def _canonical_v1(entries):
    """Match ConsolidatedTable.to_dict()['canonical_grid'] shape for resolver tests."""
    return {"schema": "CanonicalCell/v1", "immutable": True, "cells": entries}


def test_build_table_cell_grid_fast_path():
    table = {
        "table_id": "T0",
        "grid_shape": [2, 1],
        "cells": [
            {"cell_id": "r0_c0", "text": "a", "bbox": [0, 0, 10, 10], "row_idx": 0, "col_idx": 0},
            {"cell_id": "r1_c0", "text": "b", "bbox": [0, 12, 10, 22], "row_idx": 1, "col_idx": 0},
        ],
        "canonical_grid": _canonical_v1([
            {"id": "r0_c0", "text": "a", "bbox": [0, 0, 10, 10], "row_idx": 0, "col_idx": 0},
            {"id": "r1_c0", "text": "b", "bbox": [0, 12, 10, 22], "row_idx": 1, "col_idx": 0},
        ]),
    }
    grid = rr._build_table_cell_grid(table)
    assert len(grid) == 2
    assert grid[0]["row_idx"] == 0 and grid[0]["col_idx"] == 0
    assert grid[1]["row_idx"] == 1 and grid[1]["col_idx"] == 0


def test_build_table_cell_grid_partial_indexing_slow_path():
    """Resolver reads canonical_grid only; a peer without indices must not appear in the lattice."""
    table = {
        "table_id": "T1",
        "grid_shape": [2, 1],
        "cells": [
            {"cell_id": "r0_c0", "text": "a", "bbox": [0, 0, 10, 10], "row_idx": 0, "col_idx": 0},
            {"cell_id": "r1_c0", "text": "b", "bbox": [0, 12, 10, 22]},
        ],
        "canonical_grid": _canonical_v1([
            {"id": "r0_c0", "text": "a", "bbox": [0, 0, 10, 10], "row_idx": 0, "col_idx": 0},
        ]),
    }
    grid = rr._build_table_cell_grid(table)
    assert len(grid) == 1
    assert grid[0]["cell_id"] == "r0_c0"


def test_build_table_cell_grid_stale_grid_shape_falls_through():
    """Stale grid_shape metadata must not hide two indexed columns if canonical_grid is correct."""
    table = {
        "table_id": "T2",
        "grid_shape": [1, 1],
        "cells": [
            {"cell_id": "a", "text": "", "bbox": [0, 0, 10, 10], "row_idx": 0, "col_idx": 0},
            {"cell_id": "b", "text": "", "bbox": [12, 0, 22, 10], "row_idx": 0, "col_idx": 1},
        ],
        "canonical_grid": _canonical_v1([
            {"id": "a", "text": "", "bbox": [0, 0, 10, 10], "row_idx": 0, "col_idx": 0},
            {"id": "b", "text": "", "bbox": [12, 0, 22, 10], "row_idx": 0, "col_idx": 1},
        ]),
    }
    grid = rr._build_table_cell_grid(table)
    assert len(grid) == 2


def test_find_cluster_index_midpoint_bands():
    tc = TableConsolidator(DataCleaningInputsV2([], [], [], [], [], None), {})
    assert tc._find_cluster_index(49.9, [0.0, 100.0]) == 0
    assert tc._find_cluster_index(50.0, [0.0, 100.0]) == 1
    assert tc._find_cluster_index(50.1, [0.0, 100.0]) == 1


def test_ec4_bbox_overlaps():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (100.0, 100.0, 110.0, 110.0)
    assert not _ec4_bbox_overlaps(a, b, 5.0)
    b2 = (8.0, 0.0, 20.0, 10.0)
    assert _ec4_bbox_overlaps(a, b2, 5.0)


def test_compute_adaptive_tolerance_p25_not_min():
    tc = TableConsolidator(DataCleaningInputsV2([], [], [], [], [], None), {})
    out = tc._compute_adaptive_tolerance([0.0, 2.0, 27.0], 10.0)
    old_style = 2.0 * 0.4
    clamped_old = max(10.0 * 0.2, min(old_style, 10.0 * 0.7))
    assert out > clamped_old
    assert out == pytest.approx(3.1, rel=1e-5)
