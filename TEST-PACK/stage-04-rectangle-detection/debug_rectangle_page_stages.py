"""
Single-page debug: binarize checkpoints + segment counts (matches workers/rectangle_detection.py).

Usage:
  py -3 tools/debug_rectangle_page_stages.py [image_path]

Writes PNGs and prints metrics to stdout. Default image: test_data/Screenshot_23-3-2026_82649_.jpeg
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Repo root = parent of tools/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from workers import rectangle_detection as rd  # noqa: E402


def _binarize_stages(roi_gray: np.ndarray, adaptive_thresh: bool, line_thickness: float) -> dict:
    """Mirror _binarize_robust; return intermediates + flags."""
    h, w = roi_gray.shape[:2]
    aspect = max(w / max(h, 1), h / max(w, 1))
    resolution_factor = max(h, w) / rd._A4_LONG_EDGE_PX
    full_page = resolution_factor >= rd._LOW_RES_MAX_SIDE_FRAC

    use_regional = (
        not rd._REGIONAL_BINARIZE_DISABLED
        and max(h, w) >= rd._REGIONAL_BINARIZE_MIN_SPAN
        and full_page
    )
    if use_regional:
        after_binarize = rd._regional_binarize_or_merge(
            roi_gray, adaptive_thresh, line_thickness, aspect
        )
        binarize_mode = "regional_or_merge"
    else:
        after_binarize = rd._binarize_patch_no_morph(
            roi_gray, adaptive_thresh, line_thickness, aspect
        )
        binarize_mode = "single_patch"

    binary = after_binarize.copy()
    dilation_applied = False
    if line_thickness < rd._SKEL_INSTABILITY_BAND_MAX:
        if resolution_factor >= 0.7:
            binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)
            dilation_applied = True
    after_thin_dilate = binary.copy()

    erosion_applied = False
    erosion_iters = 0
    if full_page and line_thickness > rd._THICK_STROKE_EROSION_MIN:
        eff_t = line_thickness + (1.5 if dilation_applied else 0.0)
        excess = eff_t - rd._TARGET_STROKE_BEFORE_SKEL_PX
        if excess > 0:
            iters = min(rd._MAX_PRE_SKEL_EROSION_ITERS, max(1, int(round(excess / 2.0))))
            k3 = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(binary, k3, iterations=iters)
            fg_before = int(np.count_nonzero(binary))
            fg_after = int(np.count_nonzero(eroded))
            min_keep = max(200, int(rd._MIN_FG_FRACTION_AFTER_EROSION * fg_before))
            if fg_before > 0 and fg_after >= min_keep:
                binary = eroded
                erosion_applied = True
                erosion_iters = iters

    return {
        "after_binarize": after_binarize,
        "after_thin_dilate": after_thin_dilate,
        "final_binary": binary,
        "meta": {
            "h": h,
            "w": w,
            "line_thickness_est": line_thickness,
            "resolution_factor": round(resolution_factor, 4),
            "full_page": full_page,
            "use_regional": use_regional,
            "binarize_mode": binarize_mode,
            "dilation_applied_2x2": dilation_applied,
            "erosion_applied": erosion_applied,
            "erosion_iters": erosion_iters,
            "fg_pixels_after_binarize": int(np.count_nonzero(after_binarize)),
            "fg_pixels_after_dilate": int(np.count_nonzero(after_thin_dilate)),
            "fg_pixels_final": int(np.count_nonzero(binary)),
        },
    }


def main() -> None:
    default_img = _ROOT / "test_data" / "Screenshot_23-3-2026_82649_.jpeg"
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_img
    if not img_path.is_file():
        print(f"Missing image: {img_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = _ROOT / "debug_outputs" / "rectangle_page_stages"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem.replace(" ", "_")[:80]

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"Failed to read: {img_path}", file=sys.stderr)
        sys.exit(1)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    roi_h, roi_w = gray.shape[:2]

    line_thickness = rd._estimate_line_thickness(gray)
    stages = _binarize_stages(gray, adaptive_thresh=True, line_thickness=line_thickness)

    cv2.imwrite(str(out_dir / f"{stem}_01_after_binarize.png"), stages["after_binarize"])
    cv2.imwrite(str(out_dir / f"{stem}_02_after_2x2_dilate.png"), stages["after_thin_dilate"])
    cv2.imwrite(str(out_dir / f"{stem}_03_final_before_skeleton.png"), stages["final_binary"])

    binary = stages["final_binary"]
    skeleton = rd._skeletonize_binary(binary)
    # skeleton bool -> uint8 for optional save
    sk_vis = (skeleton.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / f"{stem}_04_skeleton.png"), sk_vis)

    paths = rd._trace_skeleton_paths(skeleton)
    eff_t = line_thickness + 1.5 if stages["meta"]["dilation_applied_2x2"] else line_thickness
    approx_tol = (
        rd.APPROX_POLYGON_TOLERANCE_THIN if eff_t < 2.5 else rd.APPROX_POLYGON_TOLERANCE
    )
    segments_skeleton = rd._paths_to_hv_segments(paths, tolerance=approx_tol)
    median_spacing = rd._median_vertical_spacing_horizontal_segments(segments_skeleton)
    segments_pre_merge = list(segments_skeleton)
    morph_segs = rd._morphological_line_segments(
        binary, roi_h, roi_w, median_vertical_spacing=median_spacing
    )
    segments_pre_merge.extend(morph_segs)
    n_h_pre = sum(1 for s in segments_pre_merge if s.get("orientation") == "horizontal")
    n_v_pre = sum(1 for s in segments_pre_merge if s.get("orientation") == "vertical")

    segments_merged = rd._merge_coincident_segments(segments_pre_merge)
    n_h_post = sum(1 for s in segments_merged if s.get("orientation") == "horizontal")
    n_v_post = sum(1 for s in segments_merged if s.get("orientation") == "vertical")

    raw_rects = rd._identify_rectangles_from_segments(segments_merged, roi_h, roi_w)

    report = {
        "image": str(img_path),
        "outputs": str(out_dir),
        "binarize": stages["meta"],
        "segments": {
            "skeleton_only_total": len(segments_skeleton),
            "morph_only_total": len(morph_segs),
            "pre_merge_total": len(segments_pre_merge),
            "pre_merge_horizontal": n_h_pre,
            "pre_merge_vertical": n_v_pre,
            "post_merge_total": len(segments_merged),
            "post_merge_horizontal": n_h_post,
            "post_merge_vertical": n_v_post,
            "median_spacing_used": median_spacing,
        },
        "rectangles_from_line_mask": len(raw_rects),
        "paths_traced": len(paths),
    }
    print(json.dumps(report, indent=2))
    with open(out_dir / f"{stem}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
