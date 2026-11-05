import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure repository root is on sys.path when running as standalone script.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import rectangle detection internals for stage-by-stage diagnostics.
from workers import rectangle_detection as rd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _percentiles(values: np.ndarray, points: List[int]) -> Dict[str, float]:
    if values.size == 0:
        return {f"p{p}": 0.0 for p in points}
    return {f"p{p}": float(np.percentile(values, p)) for p in points}


def _histogram(values: np.ndarray, min_v: float = 0.0, max_v: float = 12.0, step: float = 0.5) -> Dict[str, object]:
    if values.size == 0:
        return {"bin_edges": [], "counts": []}
    bins = np.arange(min_v, max_v + step, step)
    hist, edges = np.histogram(values, bins=bins)
    return {
        "bin_edges": [float(v) for v in edges.tolist()],
        "counts": [int(v) for v in hist.tolist()],
    }


def _modal_from_histogram(hist: Dict[str, object]) -> float:
    edges = hist.get("bin_edges", [])
    counts = hist.get("counts", [])
    if not edges or not counts:
        return 0.0
    idx = int(np.argmax(np.array(counts)))
    if idx + 1 >= len(edges):
        return float(edges[idx])
    return float((edges[idx] + edges[idx + 1]) * 0.5)


def _draw_skeleton_overlay(skeleton: np.ndarray) -> np.ndarray:
    vis = np.ones((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8) * 255
    vis[skeleton] = (0, 0, 0)
    return vis


def _segment_axis_stats(segments: List[dict]) -> Dict[str, object]:
    """Reproducible H/V counts and length tails — use to detect missing verticals vs over-merged horizontals."""
    h_segs = [s for s in segments if s.get("orientation") == "horizontal"]
    v_segs = [s for s in segments if s.get("orientation") == "vertical"]
    h_lens = np.array([float(s.get("length", 0)) for s in h_segs], dtype=np.float64)
    v_lens = np.array([float(s.get("length", 0)) for s in v_segs], dtype=np.float64)

    def _tail(arr: np.ndarray) -> Dict[str, float]:
        if arr.size == 0:
            return {"max": 0.0, "p95": 0.0, "mean": 0.0}
        return {
            "max": float(np.max(arr)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(np.mean(arr)),
        }

    return {
        "n_horizontal": len(h_segs),
        "n_vertical": len(v_segs),
        "horizontal_length": _tail(h_lens),
        "vertical_length": _tail(v_lens),
    }


def _draw_segment_overlay(image: np.ndarray, segments: List[dict]) -> np.ndarray:
    vis = image.copy()
    h, w = image.shape[:2]
    for seg in segments:
        x1, y1 = seg["x1"], seg["y1"]
        x2, y2 = seg["x2"], seg["y2"]
        pt1 = (max(0, min(x1, w - 1)), max(0, min(y1, h - 1)))
        pt2 = (max(0, min(x2, w - 1)), max(0, min(y2, h - 1)))
        color = (0, 255, 0) if seg.get("orientation") == "horizontal" else (255, 0, 0)
        cv2.line(vis, pt1, pt2, color, thickness=1)
    return vis


def _pairwise_overlap_distribution(rectangles: List[dict]) -> Dict[str, object]:
    ratios: List[float] = []
    n = len(rectangles)
    for i in range(n):
        b1 = rectangles[i]["bbox"]
        a1 = max(1, b1[2] * b1[3])
        for j in range(i + 1, n):
            b2 = rectangles[j]["bbox"]
            a2 = max(1, b2[2] * b2[3])
            inter = rd._bbox_intersection_area(b1, b2)
            ratio = float(inter / min(a1, a2))
            ratios.append(ratio)

    arr = np.array(ratios, dtype=np.float32) if ratios else np.array([], dtype=np.float32)
    pct = _percentiles(arr, [5, 25, 50, 75, 95])
    high_overlap_pairs = int(np.sum(arr >= 0.8)) if arr.size else 0
    partial_overlap_pairs = int(np.sum((arr >= 0.05) & (arr < 0.8))) if arr.size else 0

    return {
        "pair_count": int(arr.size),
        "overlap_ratio_percentiles": pct,
        "high_overlap_pairs": high_overlap_pairs,
        "partial_overlap_pairs": partial_overlap_pairs,
        "flags": {
            "possible_double_edge_cells": high_overlap_pairs > 0,
            "possible_split_cells": partial_overlap_pairs > 0,
        },
    }


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    union = max(1e-6, aw * ah + bw * bh - inter)
    return float(inter / union)


def _load_gt(image_path: Path, gt_dir: Optional[Path]) -> Dict[str, object]:
    """
    Weak GT format (per-image JSON):
      expected_rect_count_min: int
      expected_rect_count_max: int
      optional gt_bboxes: [[x, y, w, h], ...]
      manual_pass: bool
    """
    if gt_dir is None:
        return {}
    gt_path = gt_dir / f"{image_path.stem}.json"
    if not gt_path.exists():
        return {}
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _evaluate_vs_gt(
    predicted_rectangles: List[dict],
    weak_gt: Dict[str, object],
    fp_fn_iou_threshold: float = 0.5,
    duplicate_iou_threshold: float = 0.85,
) -> Dict[str, object]:
    pred_boxes = [[float(v) for v in r["bbox"]] for r in predicted_rectangles]
    gt_boxes = [[float(v) for v in box] for box in weak_gt.get("gt_bboxes", [])] if weak_gt.get("gt_bboxes") else []

    # Duplicate flagging among predictions.
    duplicate_pairs = 0
    for i in range(len(pred_boxes)):
        for j in range(i + 1, len(pred_boxes)):
            if _bbox_iou(pred_boxes[i], pred_boxes[j]) > duplicate_iou_threshold:
                duplicate_pairs += 1

    # FP/FN against optional GT boxes.
    fp = 0
    fn = 0
    tp = 0
    if gt_boxes:
        matched_gt = set()
        for pb in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for gi, gb in enumerate(gt_boxes):
                iou = _bbox_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_iou > fp_fn_iou_threshold and best_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        fn = len(gt_boxes) - len(matched_gt)

    rect_count = len(pred_boxes)
    min_expected = weak_gt.get("expected_rect_count_min")
    max_expected = weak_gt.get("expected_rect_count_max")
    in_range = True
    if isinstance(min_expected, int):
        in_range = in_range and rect_count >= min_expected
    if isinstance(max_expected, int):
        in_range = in_range and rect_count <= max_expected

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    duplicate_ratio = float(duplicate_pairs / max(1, len(pred_boxes)))

    return {
        "count_range_check": {
            "expected_rect_count_min": min_expected,
            "expected_rect_count_max": max_expected,
            "predicted_count": rect_count,
            "in_expected_range": bool(in_range),
        },
        "bbox_match_metrics": {
            "iou_threshold_fp_fn": fp_fn_iou_threshold,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": precision,
            "recall": recall,
        },
        "duplicate_metrics": {
            "iou_threshold_duplicate": duplicate_iou_threshold,
            "duplicate_pairs": int(duplicate_pairs),
            "duplicate_ratio_per_pred": duplicate_ratio,
        },
        "manual_pass_label": weak_gt.get("manual_pass"),
    }


def _to_canonical_report(report: Dict[str, object], image_path: Path) -> Dict[str, object]:
    """
    Strip volatile fields and absolute paths for deterministic strict diff.
    """
    # Build deterministic artifact file names (relative only).
    stem = image_path.stem
    canonical = {
        "image_name": image_path.name,
        "image_shape": report["image_shape"],
        "adaptive_thresh": report["adaptive_thresh"],
        "estimated_dominant_line_thickness_px": report["estimated_dominant_line_thickness_px"],
        "modal_stroke_width_px": report["modal_stroke_width_px"],
        "post_binarization_distance_percentiles_px": report["post_binarization_distance_percentiles_px"],
        "post_binarization_stroke_width_percentiles_px": report["post_binarization_stroke_width_percentiles_px"],
        "post_binarization_stroke_width_histogram": report["post_binarization_stroke_width_histogram"],
        "dilation_applied": report["dilation_applied"],
        "effective_thickness_for_tolerance_px": report["effective_thickness_for_tolerance_px"],
        "approx_polygon_tolerance_used": report["approx_polygon_tolerance_used"],
        "normalization": report["normalization"],
        "segment_counts": report["segment_counts"],
        "rectangle_counts": report["rectangle_counts"],
        "bbox_overlap_distribution": report["bbox_overlap_distribution"],
        "evaluation": report.get("evaluation", {}),
        "artifacts": {
            "binary": f"{stem}_diagnostic_binary.png",
            "skeleton": f"{stem}_diagnostic_skeleton.png",
            "segment_overlay": f"{stem}_diagnostic_segments.png",
            "rect_overlay": f"{stem}_diagnostic_rect_overlay.png",
        },
    }
    return canonical


def diagnose_image(
    path: str,
    adaptive_thresh: bool = True,
    gt_dir: Optional[str] = None,
    canonical_json_out: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run standalone rectangle-detection diagnostics on one image path.
    Writes JSON + intermediate images next to the input image.
    Returns the report dict.
    """
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Core stage metrics (full-ROI estimate vs effective thickness from binarize, incl. regional median).
    line_thickness_global = rd._estimate_line_thickness(gray)
    _binarize_out = rd._binarize_robust(gray, adaptive_thresh, line_thickness=None)
    if isinstance(_binarize_out, tuple) and len(_binarize_out) == 3:
        binary, dilation_applied, lt_from_binarize = _binarize_out
    elif isinstance(_binarize_out, tuple) and len(_binarize_out) == 2:
        binary, dilation_applied = _binarize_out
        lt_from_binarize = line_thickness_global
    else:
        raise ValueError(f"Unexpected _binarize_robust return shape: {type(_binarize_out)}")
    normalization_info = {}

    fg = (binary > 0).astype("uint8")
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    non_zero = dist[dist > 0]
    stroke_width_vals = non_zero * 2.0
    dist_pct = _percentiles(non_zero, [5, 25, 50, 75, 95])
    stroke_width_pct = {k: float(v * 2.0) for k, v in dist_pct.items()}
    stroke_hist = _histogram(stroke_width_vals, min_v=0.0, max_v=12.0, step=0.5)
    modal_thickness = _modal_from_histogram(stroke_hist)

    effective_thickness = float(lt_from_binarize + 1.5 if dilation_applied else lt_from_binarize)
    approx_tol = rd.APPROX_POLYGON_TOLERANCE_THIN if effective_thickness < 2.5 else rd.APPROX_POLYGON_TOLERANCE

    skeleton = rd._skeletonize_binary(binary)
    paths = rd._trace_skeleton_paths(skeleton)
    segments_skeleton = rd._paths_to_hv_segments(paths, tolerance=approx_tol)
    median_spacing = rd._median_vertical_spacing_horizontal_segments(segments_skeleton)
    segments_morph = rd._morphological_line_segments(binary, h, w, median_vertical_spacing=median_spacing)
    segments_merged = rd._merge_coincident_segments(list(segments_skeleton) + list(segments_morph))

    # Rectangle counts: skeleton path only, contour-only baseline, final detect_rectangles output.
    raw_rects = rd._identify_rectangles_from_segments(segments_merged, h, w)
    contour_rects = rd._detect_rectangles_contour_based(image, 0, 0, None, adaptive_thresh, img_w=w, img_h=h)
    final_rect_data, rect_overlay = rd.detect_rectangles(
        image=image,
        frame_rect=None,
        page_id=image_path.stem,
        adaptive_thresh=adaptive_thresh,
        debug=False,
        basename=None,
        gray=gray,
    )
    final_rectangles = final_rect_data.get("rectangles", [])
    overlap_stats = _pairwise_overlap_distribution(final_rectangles)
    weak_gt = _load_gt(image_path, Path(gt_dir) if gt_dir else None)
    evaluation = _evaluate_vs_gt(final_rectangles, weak_gt)

    # Save intermediate images beside input.
    out_prefix = image_path.with_suffix("")
    binary_path = out_prefix.with_name(f"{out_prefix.name}_diagnostic_binary.png")
    skeleton_path = out_prefix.with_name(f"{out_prefix.name}_diagnostic_skeleton.png")
    segments_path = out_prefix.with_name(f"{out_prefix.name}_diagnostic_segments.png")
    rect_overlay_path = out_prefix.with_name(f"{out_prefix.name}_diagnostic_rect_overlay.png")

    cv2.imwrite(str(binary_path), binary)
    cv2.imwrite(str(skeleton_path), _draw_skeleton_overlay(skeleton))
    cv2.imwrite(str(segments_path), _draw_segment_overlay(image, segments_merged))
    cv2.imwrite(str(rect_overlay_path), rect_overlay)

    report = {
        "image_path": str(image_path),
        "image_shape": [int(h), int(w)],
        "adaptive_thresh": bool(adaptive_thresh),
        "estimated_dominant_line_thickness_px": float(line_thickness_global),
        "median_patch_line_thickness_from_binarize_px": float(lt_from_binarize),
        "modal_stroke_width_px": float(modal_thickness),
        "post_binarization_distance_percentiles_px": dist_pct,
        "post_binarization_stroke_width_percentiles_px": stroke_width_pct,
        "post_binarization_stroke_width_histogram": stroke_hist,
        "dilation_applied": bool(dilation_applied),
        "effective_thickness_for_tolerance_px": effective_thickness,
        "approx_polygon_tolerance_used": float(approx_tol),
        "normalization": normalization_info,
        "segment_counts": {
            "skeleton": int(len(segments_skeleton)),
            "morphological": int(len(segments_morph)),
            "after_merge": int(len(segments_merged)),
        },
        "segment_axis_stats": {
            "after_skeleton_hv": _segment_axis_stats(segments_skeleton),
            "after_morph_hv": _segment_axis_stats(segments_morph),
            "after_merge_hv": _segment_axis_stats(segments_merged),
        },
        "rectangle_counts": {
            "skeleton_path_only": int(len(raw_rects)),
            "contour_only": int(len(contour_rects)),
            "final_detect_rectangles_output": int(len(final_rectangles)),
        },
        "bbox_overlap_distribution": overlap_stats,
        "evaluation": evaluation,
        "ground_truth": weak_gt,
        "artifacts": {
            "binary": str(binary_path),
            "skeleton": str(skeleton_path),
            "segment_overlay": str(segments_path),
            "rect_overlay": str(rect_overlay_path),
        },
    }

    report_path = out_prefix.with_name(f"{out_prefix.name}_diagnostic_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    canonical_report = _to_canonical_report(report, image_path)
    canonical_path = out_prefix.with_name(f"{out_prefix.name}_diagnostic_report.canonical.json")
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(canonical_report, f, indent=2, sort_keys=True)

    if canonical_json_out:
        extra = Path(canonical_json_out)
        extra.parent.mkdir(parents=True, exist_ok=True)
        with open(extra, "w", encoding="utf-8") as f:
            json.dump(canonical_report, f, indent=2, sort_keys=True)

    report["report_path"] = str(report_path)
    report["canonical_report_path"] = str(canonical_path)
    return report


def _collect_images(target: Path) -> List[Path]:
    def _is_fixture_image(p: Path) -> bool:
        name = p.stem.lower()
        # Prevent recursive self-processing of generated artifacts.
        if "_diagnostic_" in name:
            return False
        return p.suffix.lower() in IMAGE_EXTS

    if target.is_file():
        return [target] if _is_fixture_image(target) else []
    if target.is_dir():
        files: List[Path] = []
        for ext in IMAGE_EXTS:
            files.extend(target.rglob(f"*{ext}"))
        files = [p for p in files if _is_fixture_image(p)]
        return sorted(set(files))
    return []


def _main() -> int:
    parser = argparse.ArgumentParser(description="Standalone rectangle-detection diagnostics.")
    parser.add_argument("path", help="Input image file or directory of images")
    parser.add_argument("--no-adaptive-thresh", action="store_true", help="Disable adaptive thresholding")
    parser.add_argument(
        "--require-reports",
        action="store_true",
        help="Exit non-zero if any image fails to produce a diagnostic report (CI gate).",
    )
    parser.add_argument(
        "--gt-dir",
        default=None,
        help="Directory containing per-image weak GT JSON files named <image_stem>.json",
    )
    parser.add_argument(
        "--require-canonical",
        action="store_true",
        help="Exit non-zero if any canonical report is missing.",
    )
    parser.add_argument(
        "--baseline-canonical-dir",
        default=None,
        help="Optional directory of canonical baseline JSONs for strict literal diff gate.",
    )
    parser.add_argument(
        "--canonical-json-out",
        default=None,
        help=(
            "Optional extra path to write the same canonical JSON as "
            "*_diagnostic_report.canonical.json (for fixed output filenames in CI)."
        ),
    )
    args = parser.parse_args()

    target = Path(args.path)
    images = _collect_images(target)
    if not images:
        print(f"No supported images found at: {target}")
        return 2
    if args.canonical_json_out and len(images) != 1:
        print("--canonical-json-out is only supported when the input path resolves to exactly one image.")
        return 2

    adaptive_thresh = not args.no_adaptive_thresh
    failures: List[Tuple[Path, str]] = []
    generated_reports: List[Path] = []

    for img_path in images:
        try:
            report = diagnose_image(
                str(img_path),
                adaptive_thresh=adaptive_thresh,
                gt_dir=args.gt_dir,
                canonical_json_out=args.canonical_json_out,
            )
            generated_reports.append(Path(report["report_path"]))
            print(f"[OK] {img_path} -> {report['canonical_report_path']}")
        except Exception as e:
            failures.append((img_path, str(e)))
            print(f"[FAIL] {img_path}: {e}")

    if failures:
        print("\nFailed images:")
        for img_path, msg in failures:
            print(f"  - {img_path}: {msg}")

    if args.require_reports:
        missing = [img for img in images if not img.with_suffix("").with_name(f"{img.with_suffix('').name}_diagnostic_report.json").exists()]
        if missing:
            print("\nMissing reports:")
            for m in missing:
                print(f"  - {m}")
            return 1

    if args.require_canonical:
        missing_canonical = [
            img for img in images
            if not img.with_suffix("").with_name(f"{img.with_suffix('').name}_diagnostic_report.canonical.json").exists()
        ]
        if missing_canonical:
            print("\nMissing canonical reports:")
            for m in missing_canonical:
                print(f"  - {m}")
            return 1

    if args.baseline_canonical_dir:
        baseline_dir = Path(args.baseline_canonical_dir)
        strict_diff_failures: List[str] = []
        for img in images:
            current = img.with_suffix("").with_name(f"{img.with_suffix('').name}_diagnostic_report.canonical.json")
            baseline = baseline_dir / current.name
            if not current.exists():
                strict_diff_failures.append(f"{img.name}: current canonical report missing")
                continue
            if not baseline.exists():
                strict_diff_failures.append(f"{img.name}: baseline canonical report missing at {baseline}")
                continue
            with open(current, "r", encoding="utf-8") as f:
                current_text = f.read().strip()
            with open(baseline, "r", encoding="utf-8") as f:
                baseline_text = f.read().strip()
            if current_text != baseline_text:
                strict_diff_failures.append(f"{img.name}: canonical JSON differs from baseline")
        if strict_diff_failures:
            print("\nStrict canonical diff failures:")
            for item in strict_diff_failures:
                print(f"  - {item}")
            return 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())

