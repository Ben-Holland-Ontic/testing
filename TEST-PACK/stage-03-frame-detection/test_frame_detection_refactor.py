from __future__ import annotations

import cv2
import numpy as np

from workers.frame_detection import detect_frame


def _make_page(width=2400, height=3400, margin_x=100, margin_y=120, thickness=5):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(
        img,
        (margin_x, margin_y),
        (width - margin_x, height - margin_y),
        (0, 0, 0),
        thickness=thickness,
    )
    return img


def _margin_ratios(bbox, page_w, page_h):
    x, y, w, h = bbox
    left = x
    top = y
    right = page_w - (x + w)
    bottom = page_h - (y + h)
    return (
        left / page_w,
        right / page_w,
        top / page_h,
        bottom / page_h,
    )


def test_frame_detects_valid_border():
    img = _make_page()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result, _ = detect_frame(img, page_id="p1", gray=gray, scan_branch="cg_normal_legacy")
    assert "bbox" in result, result
    assert result["strategy"] in {"A", "C", "B"}
    assert result["score"] >= 0.0

    ratios = _margin_ratios(result["bbox"], img.shape[1], img.shape[0])
    assert all(r <= 0.08 for r in ratios), ratios


def test_frame_rejects_asymmetric_non_frame():
    img = np.full((3200, 2200, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (80, 100), (1300, 3100), (0, 0, 0), thickness=5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result, _ = detect_frame(img, page_id="p2", gray=gray, scan_branch="cg_normal_legacy")
    assert result.get("skip") is True
    assert result.get("reason") in {
        "frame_margin_asymmetry",
        "frame_margin_ratio_exceeded",
        "frame_aspect_mismatch",
        "frame_not_closed_or_unrecoverable",
    }


def test_frame_reconstructed_branch_uses_single_binary_path():
    img = _make_page(margin_x=90, margin_y=140, thickness=4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result, _ = detect_frame(img, page_id="p3", gray=gray, scan_branch="reconstructed")
    assert "bbox" in result, result


def test_frame_detection_is_deterministic():
    img = _make_page(margin_x=110, margin_y=150, thickness=5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result_a, _ = detect_frame(img, page_id="p4", gray=gray, scan_branch="cg_normal_legacy")
    result_b, _ = detect_frame(img, page_id="p4", gray=gray, scan_branch="cg_normal_legacy")
    assert result_a == result_b
