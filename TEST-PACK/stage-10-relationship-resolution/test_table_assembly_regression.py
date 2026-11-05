"""
Regression tests for:
1. known_query_names missing from _build_table_level_anchor_dicts (header row misidentified
   when data row has more non-empty cells than header row)
2. known_query_aliases missing so 'REV' cell was not canonicalized to 'Revision' via alias 'Rev.'
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "cleaners"))

import relationship_resolver_v2_enhanced as rr  # noqa: E402
from relationship_resolver_v2_enhanced import (
    _build_table_level_anchor_dicts,
    _build_table_cell_grid,
    assemble_table_records,
    Tolerances,
)


def _make_tol():
    return Tolerances(
        tau_x=30.0, tau_y=20.0,
        close_gap=6.0, max_gap=40.0, coord_tol=25.0,
    )


def _cell(cell_id, text, row_idx, col_idx, x0=0, y0=0, x1=50, y1=20):
    return {
        "id": cell_id,
        "cell_id": cell_id,
        "text": text,
        "bbox": [x0, y0, x1, y1],
        "row_idx": row_idx,
        "col_idx": col_idx,
        "row_span": 1,
        "col_span": 1,
    }


def _canonical(cells):
    return {"schema": "CanonicalCell/v1", "immutable": True, "cells": cells}


# ──────────────────────────────────────────────────────────────────
# Test 1: known_query_names fixes header-row selection
#   Scenario: 1520 roller assembly - 7x5 grid with header at BOTTOM (row 6)
#   Bug: without known_query_names, row 0 (5 non-empty cells) beats row 6 (4 non-empty)
# ──────────────────────────────────────────────────────────────────
def _make_1520_table():
    """Minimal replica of fc2bae5c953f24fe Table grid=[7,5]."""
    cells = []
    # Row 0 - data (5 non-empty)
    for col, text in enumerate(["A", "HexBolt", "Steel,Mild", "4", "ANSIB18.2.3.5M-M10"]):
        cells.append(_cell(f"r0c{col}", text, 0, col, col*60, 0, col*60+55, 18))
    # Rows 1-5 - more data
    for row in range(1, 6):
        for col, text in enumerate(["1", "Component", "Material", str(row), f"RA-0{row}"]):
            cells.append(_cell(f"r{row}c{col}", text, row, col, col*60, row*20, col*60+55, row*20+18))
    # Row 6 - HEADER (col 0 empty, 4 non-empty)
    for col, text in enumerate(["", "DESCRIPTION", "MATERIAL 1", "TQrY", "PARTNUMBER"]):
        cells.append(_cell(f"r6c{col}", text, 6, col, col*60, 120, col*60+55, 138))
    return {
        "table_id": "T_1520",
        "bbox": [0, 0, 300, 140],
        "cells": cells,
        "canonical_grid": _canonical(cells),
        "ec_type": "STANDARD",
        "reading_path": "cell_grid",
    }


def test_known_query_names_selects_correct_header_row():
    """Without known_query_names, row 0 (more cells) beats row 6 (header). Fix: pass names."""
    table = _make_1520_table()
    known_names = ["Part Number", "Item Number", "Quantity", "Drawing Number",
                   "Revision", "Drawing Description", "CAGE Code", "Material"]

    # One anchor pointing to PARTNUMBER at [6,4]
    anchor = {
        "query_name": "Part Number",
        "source_type": "table_cell",
        "source_id": f"T_1520/r6c4",
        "block_id": "T_1520",
        "bbox": [240, 120, 295, 138],
        "anchor_text": "PARTNUMBER",
        "text_norm": "partnumber",
        "score": 0.9,
        "alias_matched": "Part Number",
        "fuzz_score": 0.9,
        "cosine_score": 0.9,
    }
    second_anchor = {
        "query_name": "Drawing Description",
        "source_type": "table_cell",
        "source_id": f"T_1520/r6c1",
        "block_id": "T_1520",
        "bbox": [60, 120, 115, 138],
        "anchor_text": "DESCRIPTION",
        "text_norm": "description",
        "score": 0.85,
        "alias_matched": "DESCRIPTION",
        "fuzz_score": 0.85,
        "cosine_score": 0.85,
    }

    # WITHOUT fix: no known_query_names → _band_score uses only geometry
    anchors_without = _build_table_level_anchor_dicts(
        table_id="T_1520",
        table_dict=table,
        query_anchor_dicts=[anchor, second_anchor],
        tol=_make_tol(),
        known_query_names=None,
    )
    # WITH fix: known_query_names passed → qscore tips balance toward row 6
    anchors_with = _build_table_level_anchor_dicts(
        table_id="T_1520",
        table_dict=table,
        query_anchor_dicts=[anchor, second_anchor],
        tol=_make_tol(),
        known_query_names=known_names,
    )

    # With fix: every synthetic anchor must point to row 6 (header row)
    assert anchors_with, "Expected synthetic anchors with known_query_names"
    rows_with = {a["source_id"].split("/")[1] for a in anchors_with}
    # All header anchors should be from row 6 cells (source_id ends in r6c*)
    assert all(sid.startswith("r6") for sid in rows_with), (
        f"Expected all anchors from row 6, got source_ids: {rows_with}"
    )
    print(f"  [PASS] With known_query_names: header anchors from row 6: {rows_with}")

    # Without fix: may select row 0 (more non-empty cells) since anchor_hits push row 6
    # (anchor_hits still helps here, but test documents the behaviour was broken without query names
    # when anchor_hits alone are not present from _enrich path)
    print(f"  [INFO] Without known_query_names: source_ids = "
          f"{[a['source_id'] for a in anchors_without[:4]]}")


# ──────────────────────────────────────────────────────────────────
# Test 2: alias canonicalization - 'REV' → 'Revision' via alias 'Rev.'
# ──────────────────────────────────────────────────────────────────
def _make_revision_table():
    """Minimal replica of 06b71112bcf9c673 revision block [4,14]."""
    cells = []
    # Row 0: sheet numbers + REV label at col 13
    for col in range(13):
        cells.append(_cell(f"r0c{col}", str(col + 1) if col < 3 else "", 0, col, col*60, 0, col*60+55, 18))
    cells.append(_cell("r0c13", "REV", 0, 13, 780, 0, 835, 18))
    # Row 1: more sheet numbers + SHEET label at col 13
    for col in range(13):
        cells.append(_cell(f"r1c{col}", str(26 - col), 1, col, col*60, 20, col*60+55, 38))
    cells.append(_cell("r1c13", "SHEET", 1, 13, 780, 20, 835, 38))
    # Row 2: revision letters
    for col in range(13):
        cells.append(_cell(f"r2c{col}", "B", 2, col, col*60, 40, col*60+55, 58))
    cells.append(_cell("r2c13", "", 2, 13, 780, 40, 835, 58))
    # Row 3: more revision letters
    for col in range(13):
        cells.append(_cell(f"r3c{col}", "B", 3, col, col*60, 60, col*60+55, 78))
    cells.append(_cell("r3c13", "", 3, 13, 780, 60, 835, 78))
    return {
        "table_id": "T_REV",
        "bbox": [0, 0, 840, 80],
        "cells": cells,
        "canonical_grid": _canonical(cells),
        "ec_type": "STANDARD",
        "reading_path": "cell_grid",
    }


def test_alias_canonicalization_rev_to_revision():
    """'REV' header cell must canonicalize to 'Revision' via alias 'Rev.'"""
    table = _make_revision_table()
    known_names = ["Revision", "Sheet Number", "Drawing Number", "Part Number"]
    known_aliases = {
        "Revision": ["Rev.", "Rev No.", "Revision Number", "Rev Code", "Version", "Rey"],
        "Sheet Number": ["Sheet", "Sheet No.", "Sheet #"],
    }

    rev_anchor = {
        "query_name": "Revision",
        "source_type": "table_cell",
        "source_id": "T_REV/r0c13",
        "block_id": "T_REV",
        "bbox": [780, 0, 835, 18],
        "anchor_text": "REV",
        "text_norm": "rev",
        "score": 0.85,
        "alias_matched": "Rev.",
        "fuzz_score": 0.85,
        "cosine_score": 0.85,
    }
    sheet_anchor = {
        "query_name": "Sheet Number",
        "source_type": "table_cell",
        "source_id": "T_REV/r1c13",
        "block_id": "T_REV",
        "bbox": [780, 20, 835, 38],
        "anchor_text": "SHEET",
        "text_norm": "sheet",
        "score": 0.85,
        "alias_matched": "Sheet",
        "fuzz_score": 0.85,
        "cosine_score": 0.85,
    }

    anchors = _build_table_level_anchor_dicts(
        table_id="T_REV",
        table_dict=table,
        query_anchor_dicts=[rev_anchor, sheet_anchor],
        tol=_make_tol(),
        known_query_names=known_names,
        known_query_aliases=known_aliases,
    )

    assert anchors, "Expected synthetic anchors for revision table"
    names = [a["query_name"] for a in anchors]
    print(f"  [INFO] Synthetic anchor names: {names}")

    # 'REV' cell should be canonicalized to 'Revision' via alias 'Rev.'
    assert "Revision" in names, (
        f"Expected 'Revision' in anchor names via alias 'Rev.', got: {names}"
    )
    # 'SHEET' should be canonicalized to 'Sheet Number' via alias 'Sheet'
    assert "Sheet Number" in names, (
        f"Expected 'Sheet Number' in anchor names via alias 'Sheet', got: {names}"
    )
    print("  [PASS] REV -> 'Revision', SHEET -> 'Sheet Number'")


# ──────────────────────────────────────────────────────────────────
# Test 3: without aliases, 'REV' scores < 72 against 'Revision' alone
# ──────────────────────────────────────────────────────────────────
def test_rev_below_threshold_without_aliases():
    """Confirm 'REV' alone does NOT reach 72 against 'Revision' — aliases are needed."""
    from rapidfuzz import fuzz
    score = fuzz.token_set_ratio("rev", "revision")
    assert score < 72, f"Expected score < 72 (confirming aliases needed), got {score}"
    alias_score = fuzz.token_set_ratio("rev", "rev.")
    assert alias_score >= 72, f"Expected 'rev' vs 'rev.' >= 72, got {alias_score}"
    print(f"  [PASS] 'rev' vs 'revision'={score} (<72, needs alias); 'rev' vs 'rev.'={alias_score} (>=72)")


if __name__ == "__main__":
    print("\n=== test_known_query_names_selects_correct_header_row ===")
    test_known_query_names_selects_correct_header_row()
    print("\n=== test_alias_canonicalization_rev_to_revision ===")
    test_alias_canonicalization_rev_to_revision()
    print("\n=== test_rev_below_threshold_without_aliases ===")
    test_rev_below_threshold_without_aliases()
    print("\nAll tests passed.")
