"""
Regression guard for rectangle line-mask stage (1759857039776 grid / BOM style pages).

Fails if morph/path/merge changes drop enclosed regions below the known-good floor.
Run: py -3 -m pytest tests/test_rectangle_detection_regression.py -q
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _line_mask_rect_count(gray):
    from workers import rectangle_detection as rd

    roi_h, roi_w = gray.shape[:2]
    lt = rd._estimate_line_thickness(gray)
    binary, dilation_applied = rd._binarize_robust(gray, True, line_thickness=lt)
    sk = rd._skeletonize_binary(binary)
    paths = rd._trace_skeleton_paths(sk)
    eff_t = lt + 1.5 if dilation_applied else lt
    tol = (
        rd.APPROX_POLYGON_TOLERANCE_THIN
        if eff_t < 2.5
        else rd.APPROX_POLYGON_TOLERANCE
    )
    legacy_poly = not rd._segment_style_is_lowdpi_canvas(roi_h, roi_w)
    segs = rd._paths_to_hv_segments(
        paths, tolerance=tol, legacy_poly_simplify=legacy_poly
    )
    med = rd._median_vertical_spacing_horizontal_segments(segs)
    morph = rd._morphological_line_segments(
        binary, roi_h, roi_w, median_vertical_spacing=med
    )
    merged = rd._merge_coincident_segments(list(segs) + list(morph))
    rects = rd._identify_rectangles_from_segments(merged, roi_h, roi_w)
    return len(rects)


def test_1759857039776_line_mask_rectangles_min():
    path = _ROOT / "test_data" / "1759857039776.jpg"
    if not path.is_file():
        pytest.skip(f"missing fixture: {path}")
    img = cv2.imread(str(path))
    assert img is not None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n = _line_mask_rect_count(gray)
    # Floor from legacyrect.PY A/B on this asset (34); allow small upside drift.
    assert n >= 34, f"line-mask rects {n}, expected >= 34 (regression vs legacy grid)"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
