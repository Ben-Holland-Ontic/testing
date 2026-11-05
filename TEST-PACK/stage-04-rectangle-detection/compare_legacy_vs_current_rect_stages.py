"""
Stage-by-stage comparison: WORKING OUTPUTS/legacyrect.PY vs workers/rectangle_detection.py
Same grayscale input; reports H/V segment counts and binary diff.

Usage:
  py -3 tools/compare_legacy_vs_current_rect_stages.py [path_to.jpg]
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LEGACY_PATH = _ROOT.parent / "WORKING OUTPUTS" / "legacyrect.PY"


def _load_legacy():
    # Uppercase .PY: spec_from_file_location may not attach a loader on Windows.
    loader = importlib.machinery.SourceFileLoader(
        "legacy_rect_module", str(LEGACY_PATH.resolve())
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _hv_counts(segments: list) -> tuple:
    h = sum(1 for s in segments if s.get("orientation") == "horizontal")
    v = sum(1 for s in segments if s.get("orientation") == "vertical")
    return h, v


def _rdp_epsilon_used(mod, approx_tol: float) -> float:
    """Match workers.rectangle_detection thin-band RDP scaling (legacy has no scale)."""
    thin = getattr(mod, "APPROX_POLYGON_TOLERANCE_THIN", None)
    scale = getattr(mod, "_PATH_SIMPLIFY_SCALE_THIN", None)
    if thin is not None and scale is not None and approx_tol <= thin:
        return float(approx_tol * scale)
    return float(approx_tol)


def _pipeline_full(mod, gray: np.ndarray):
    """Full pipeline through merged segments + line-mask rects (single compute path)."""
    roi_h, roi_w = gray.shape[:2]
    lt = mod._estimate_line_thickness(gray)
    binary, dilation_applied = mod._binarize_robust(gray, True, line_thickness=lt)
    skel = mod._skeletonize_binary(binary)
    paths = mod._trace_skeleton_paths(skel)
    eff_t = lt + 1.5 if dilation_applied else lt
    approx_tol = (
        mod.APPROX_POLYGON_TOLERANCE_THIN if eff_t < 2.5 else mod.APPROX_POLYGON_TOLERANCE
    )
    rdp_eps = _rdp_epsilon_used(mod, approx_tol)
    skel_segs = mod._paths_to_hv_segments(paths, tolerance=approx_tol)
    med = mod._median_vertical_spacing_horizontal_segments(skel_segs)
    _morph_kw = dict(
        binary=binary,
        roi_h=roi_h,
        roi_w=roi_w,
        median_vertical_spacing=med,
    )
    if "line_thickness" in inspect.signature(
        mod._morphological_line_segments
    ).parameters:
        _morph_kw["line_thickness"] = lt
    morph_segs = mod._morphological_line_segments(**_morph_kw)
    pre = list(skel_segs) + list(morph_segs)
    merged = mod._merge_coincident_segments(pre)
    rects = mod._identify_rectangles_from_segments(merged, roi_h, roi_w)
    line_mask = mod._segments_to_line_mask(merged, roi_h, roi_w)
    return {
        "roi_h": roi_h,
        "roi_w": roi_w,
        "line_thickness": lt,
        "dilation_applied": dilation_applied,
        "approx_tol_used": approx_tol,
        "rdp_epsilon_effective": rdp_eps,
        "median_spacing": med,
        "binary": binary,
        "fg_pixels_binary": int(np.count_nonzero(binary)),
        "paths": len(paths),
        "skel_segs": skel_segs,
        "morph_segs": morph_segs,
        "pre": pre,
        "merged": merged,
        "rects": rects,
        "line_mask": line_mask,
    }


def _stages_from_full(p: dict, label: str) -> dict:
    skel_segs, morph_segs, pre, merged = p["skel_segs"], p["morph_segs"], p["pre"], p["merged"]
    rects = p["rects"]
    sh, sv = _hv_counts(skel_segs)
    mh, mv = _hv_counts(morph_segs)
    ph, pv = _hv_counts(pre)
    mgh, mgv = _hv_counts(merged)
    return {
        "label": label,
        "estimate_line_thickness": p["line_thickness"],
        "dilation_applied": p["dilation_applied"],
        "approx_tol_used": p["approx_tol_used"],
        "rdp_epsilon_effective": p["rdp_epsilon_effective"],
        "median_spacing": p["median_spacing"],
        "fg_pixels_binary": p["fg_pixels_binary"],
        "paths": p["paths"],
        "skeleton_segments": {"total": len(skel_segs), "H": sh, "V": sv},
        "morph_segments": {"total": len(morph_segs), "H": mh, "V": mv},
        "pre_merge": {"total": len(pre), "H": ph, "V": pv},
        "post_merge": {"total": len(merged), "H": mgh, "V": mgv},
        "rectangles_line_mask": len(rects),
    }


def _bbox_iou(a: tuple, b: tuple) -> float:
    """a,b = (x,y,w,h) axis-aligned."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def _legacy_only_rects(legacy_rects: list, current_rects: list, iou_thr: float = 0.35):
    """Legacy boxes with no strong overlap with any current box (explains fewer current count)."""
    out = []
    for r in legacy_rects:
        best = max((_bbox_iou(r, c) for c in current_rects), default=0.0)
        if best < iou_thr:
            out.append({"bbox_xywh": list(r), "best_iou_with_current": best})
    return out


def _contours_ge_min_area(mod, merged: list, roi_h: int, roi_w: int) -> dict:
    """OpenCV contour counts on line mask vs rectangles accepted after shape filters."""
    m = mod._segments_to_line_mask(merged, roi_h, roi_w)
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_a = (roi_w * roi_h) * 0.0001
    n_big = sum(1 for c in cnts if cv2.contourArea(c) >= min_a)
    n_acc = len(mod._identify_rectangles_from_segments(merged, roi_h, roi_w))
    return {
        "contours_total": len(cnts),
        "contours_ge_min_area": n_big,
        "rectangles_after_shape_filters": n_acc,
    }


def _white_cc_count(mask_255_bg: np.ndarray) -> int:
    """Connected components of white (line mask background), 4-neighbors."""
    fg = (mask_255_bg >= 128).astype(np.uint8)
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=4)
    h, w = mask_255_bg.shape[:2]
    min_a = max(4, int((h * w) * 0.00005))
    return sum(
        1 for i in range(1, nlab) if stats[i, cv2.CC_STAT_AREA] >= min_a
    )


def main() -> None:
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "test_data" / "1759857039776.jpg"
    if not img_path.is_file():
        print(f"Missing: {img_path}", file=sys.stderr)
        sys.exit(1)
    if not LEGACY_PATH.is_file():
        print(f"Missing legacy module: {LEGACY_PATH}", file=sys.stderr)
        sys.exit(1)

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"Cannot read image: {img_path}", file=sys.stderr)
        sys.exit(1)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    leg = _load_legacy()
    from workers import rectangle_detection as cur

    pl = _pipeline_full(leg, gray)
    pc = _pipeline_full(cur, gray)
    bin_leg, bin_cur = pl["binary"], pc["binary"]
    xor = np.bitwise_xor(bin_leg > 0, bin_cur > 0)
    xor_pixels = int(np.count_nonzero(xor))

    mask_l = pl["line_mask"]
    mask_c = pc["line_mask"]
    line_xor = int(np.count_nonzero(mask_l != mask_c))
    black_l = int(np.count_nonzero(mask_l < 128))
    black_c = int(np.count_nonzero(mask_c < 128))

    legacy_rects = pl["rects"]
    current_rects = pc["rects"]
    legacy_only = _legacy_only_rects(legacy_rects, current_rects, iou_thr=0.35)
    current_only = _legacy_only_rects(current_rects, legacy_rects, iou_thr=0.35)

    out = {
        "image": str(img_path),
        "legacy_module": str(LEGACY_PATH),
        "line_thickness": {"legacy": pl["line_thickness"], "current": pc["line_thickness"]},
        "binary_xor_pixels": xor_pixels,
        "binary_identical": xor_pixels == 0,
        "line_mask": {
            "xor_pixels": line_xor,
            "black_pixels_legacy": black_l,
            "black_pixels_current": black_c,
            "black_delta_current_minus_legacy": black_c - black_l,
            "white_cc_approx_legacy": _white_cc_count(mask_l),
            "white_cc_approx_current": _white_cc_count(mask_c),
            "contours": {
                "legacy": _contours_ge_min_area(leg, pl["merged"], pl["roi_h"], pl["roi_w"]),
                "current": _contours_ge_min_area(cur, pc["merged"], pc["roi_h"], pc["roi_w"]),
            },
        },
        "rectangle_diff": {
            "legacy_only_no_current_match_iou035": legacy_only,
            "current_only_no_legacy_match_iou035": current_only,
        },
        "legacy": _stages_from_full(pl, "legacy"),
        "current": _stages_from_full(pc, "current"),
    }

    # Deltas (current - legacy) for horizontal-focused diagnosis
    l, c = out["legacy"], out["current"]
    out["delta_current_minus_legacy"] = {
        "skeleton_H": c["skeleton_segments"]["H"] - l["skeleton_segments"]["H"],
        "skeleton_V": c["skeleton_segments"]["V"] - l["skeleton_segments"]["V"],
        "morph_H": c["morph_segments"]["H"] - l["morph_segments"]["H"],
        "morph_V": c["morph_segments"]["V"] - l["morph_segments"]["V"],
        "post_merge_H": c["post_merge"]["H"] - l["post_merge"]["H"],
        "post_merge_V": c["post_merge"]["V"] - l["post_merge"]["V"],
        "rectangles": c["rectangles_line_mask"] - l["rectangles_line_mask"],
    }

    print(json.dumps(out, indent=2))

    # Human-readable hint (delta = current minus legacy; negative means current has fewer)
    d = out["delta_current_minus_legacy"]
    if xor_pixels > 0:
        print(
            f"\nNOTE: binaries differ in {xor_pixels} pixels; "
            "check _binarize_robust (regional patch, erosion, full_page).",
            file=sys.stderr,
        )
    if d["skeleton_H"] < 0 or d["skeleton_V"] < 0:
        print(
            f"\nNOTE: current has FEWER skeleton segments than legacy "
            f"(dH={d['skeleton_H']} dV={d['skeleton_V']}); "
            "check _paths_to_hv_segments (RDP epsilon, 1.5x axis test, reject-band drop).",
            file=sys.stderr,
        )
    elif d["skeleton_H"] > 0 or d["skeleton_V"] > 0:
        if xor_pixels == 0:
            print(
                f"\nNOTE: same binary/paths but current has MORE skeleton segments "
                f"(dH={d['skeleton_H']} dV={d['skeleton_V']}); "
                f"legacy RDP eps={l['rdp_epsilon_effective']:.4g} vs current={c['rdp_epsilon_effective']:.4g} "
                "and/or orthogonalize branch in current _paths_to_hv_segments.",
                file=sys.stderr,
            )
    if xor_pixels == 0 and d["skeleton_H"] == 0 and d["skeleton_V"] == 0:
        if d["morph_H"] != 0 or d["morph_V"] != 0:
            print(
                f"\nNOTE: skeleton segs match; morph dH={d['morph_H']} dV={d['morph_V']} "
                "-> _morphological_line_segments.",
                file=sys.stderr,
            )
        elif d["post_merge_H"] != 0 or d["post_merge_V"] != 0:
            print(
                f"\nNOTE: morph match; merge dH={d['post_merge_H']} dV={d['post_merge_V']} "
                "-> _merge_coincident_segments.",
                file=sys.stderr,
            )
    if d["rectangles"] != 0:
        print(
            f"\nNOTE: _identify_rectangles_from_segments count delta={d['rectangles']} "
            "(line-mask rects only; not contour fallback).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
