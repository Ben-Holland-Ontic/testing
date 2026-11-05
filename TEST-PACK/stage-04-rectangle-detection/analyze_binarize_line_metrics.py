#!/usr/bin/env python3
"""
Measure vertical/horizontal line structure on the *same* binary as rectangle_detection
(_binarize_robust after normalization). Use this to explain tradeoffs like
"more verticals in one region, fewer in another" when tuning thresholds.

Metrics (intuitive):
  - vertical_fragmentation_index: mean (# of disjoint foreground runs per column);
      higher => more broken vertical strokes / speckle columns.
  - vertical_continuity_mass: sum over columns of (longest run length); rewards
      long unbroken verticals.
  - horizontal_*: same along rows.
  - projection_peak_count_*: crude count of strong peaks in column/row sums
      (proxy for dominant grid lines vs noise).
  - per_quadrant: same stats on TL/TR/BL/BR so regional binarization tradeoffs show up.

Examples:
  py -3 tools/analyze_binarize_line_metrics.py test_data/Screenshot_18-3-2026_101153_.jpeg
  py -3 tools/analyze_binarize_line_metrics.py page.png --json-out metrics.json
  py -3 tools/analyze_binarize_line_metrics.py page.png --compare-binary old_diagnostic_binary.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workers import rectangle_detection as rd


def _fg_mask(binary: np.ndarray) -> np.ndarray:
    return (binary > 0).astype(np.uint8)


def _runs_1d(mask_col: np.ndarray) -> Tuple[int, int]:
    """Longest run of 1s and number of runs in a 1D 0/1 array."""
    if mask_col.size == 0:
        return 0, 0
    padded = np.pad(mask_col.astype(np.int8), (1, 1), mode="constant")
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if starts.size == 0:
        return 0, 0
    lengths = ends - starts
    return int(lengths.max()), int(lengths.size)


def _axis_metrics(fg: np.ndarray, axis: int) -> Dict[str, Any]:
    """
    axis=0 => scan columns (vertical strokes: tall runs along y in each column).
    axis=1 => scan rows (horizontal strokes).
    """
    h, w = fg.shape
    if axis == 0:
        n = w
        max_runs: List[int] = []
        n_runs: List[int] = []
        for x in range(w):
            col = fg[:, x]
            mx, nr = _runs_1d(col)
            max_runs.append(mx)
            n_runs.append(nr)
        proj = fg.sum(axis=0).astype(np.float64)
    else:
        n = h
        max_runs = []
        n_runs = []
        for y in range(h):
            row = fg[y, :]
            mx, nr = _runs_1d(row)
            max_runs.append(mx)
            n_runs.append(nr)
        proj = fg.sum(axis=1).astype(np.float64)

    max_runs_a = np.array(max_runs, dtype=np.float64)
    n_runs_a = np.array(n_runs, dtype=np.float64)

    peak_thr = 0.08 * float(np.max(proj)) if np.max(proj) > 0 else 0.0
    peaks = 0
    if proj.size >= 3:
        for i in range(1, len(proj) - 1):
            if proj[i] >= proj[i - 1] and proj[i] >= proj[i + 1] and proj[i] > peak_thr:
                peaks += 1

    long_cols = int(np.sum(max_runs_a >= 8))

    return {
        "continuity_mass": float(np.sum(max_runs_a)),
        "fragmentation_index": float(np.mean(n_runs_a)) if n else 0.0,
        "max_run_p50": float(np.percentile(max_runs_a, 50)) if n else 0.0,
        "max_run_p90": float(np.percentile(max_runs_a, 90)) if n else 0.0,
        "mean_max_run": float(np.mean(max_runs_a)) if n else 0.0,
        "projection_peak_count": int(peaks),
        "lanes_with_long_run_ge_8px": long_cols,
    }


def _quadrant_regions(h: int, w: int) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    th, tw = h // 2, w // 2
    return [
        ("TL", (0, th, 0, tw)),
        ("TR", (0, th, tw, w)),
        ("BL", (th, h, 0, tw)),
        ("BR", (th, h, tw, w)),
    ]


def analyze_binary(fg: np.ndarray, label: str = "binary") -> Dict[str, Any]:
    h, w = fg.shape[:2]
    vert = _axis_metrics(fg, axis=0)
    horiz = _axis_metrics(fg, axis=1)
    out: Dict[str, Any] = {
        "label": label,
        "shape": [int(h), int(w)],
        "foreground_pixel_fraction": float(np.mean(fg > 0)),
        "vertical": vert,
        "horizontal": horiz,
        "per_quadrant": {},
    }
    for name, (y1, y2, x1, x2) in _quadrant_regions(h, w):
        patch = fg[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        v = _axis_metrics(patch, axis=0)
        hmet = _axis_metrics(patch, axis=1)
        out["per_quadrant"][name] = {
            "bbox": [y1, y2, x1, x2],
            "vertical": v,
            "horizontal": hmet,
            "foreground_pixel_fraction": float(np.mean(patch > 0)),
        }
    return out


def load_compare_binary(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read binary/compare image: {path}")
    return _fg_mask(img)


def main() -> int:
    parser = argparse.ArgumentParser(description="Line-structure metrics on _binarize_robust output.")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--no-adaptive-thresh", action="store_true", help="Match diagnose_rectangle_image --no-adaptive-thresh")
    parser.add_argument("--compare-binary", type=str, default=None, help="Optional prior diagnostic binary PNG to diff metrics")
    parser.add_argument("--json-out", type=str, default=None, help="Write full report JSON here")
    args = parser.parse_args()

    path = Path(args.image)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"Failed to load: {path}", file=sys.stderr)
        return 2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    adaptive = not args.no_adaptive_thresh
    lt = rd._estimate_line_thickness(gray)
    binary, dilation_applied, lt_binarize = rd._binarize_robust(gray, adaptive)
    norm_info = {}
    fg = _fg_mask(binary)

    report: Dict[str, Any] = {
        "image_path": str(path.resolve()),
        "adaptive_thresh": adaptive,
        "estimated_line_thickness_px": float(lt),
        "binarize_effective_line_thickness_px": float(lt_binarize),
        "dilation_applied": bool(dilation_applied),
        "normalization": norm_info,
        "current": analyze_binary(fg, "current"),
    }

    if args.compare_binary:
        cpath = Path(args.compare_binary)
        fg_b = load_compare_binary(cpath)
        if fg_b.shape != fg.shape:
            print(
                f"Warning: compare binary shape {fg_b.shape} != current {fg.shape}; skipping compare.",
                file=sys.stderr,
            )
        else:
            report["baseline"] = analyze_binary(fg_b, str(cpath))
            report["delta"] = _delta_metrics(report["current"], report["baseline"])

    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    # Human-readable summary
    cur = report["current"]
    print(f"Image: {path.name}  shape={cur['shape']}  fg_frac={cur['foreground_pixel_fraction']:.4f}")
    print(f"  vertical:  continuity_mass={cur['vertical']['continuity_mass']:.0f}  "
          f"frag_index={cur['vertical']['fragmentation_index']:.3f}  "
          f"peaks={cur['vertical']['projection_peak_count']}  "
          f"long_lanes_ge8={cur['vertical']['lanes_with_long_run_ge_8px']}")
    print(f"  horizontal: continuity_mass={cur['horizontal']['continuity_mass']:.0f}  "
          f"frag_index={cur['horizontal']['fragmentation_index']:.3f}  "
          f"peaks={cur['horizontal']['projection_peak_count']}")
    print("  per_quadrant (vertical continuity / vertical frag / horiz continuity / horiz frag):")
    for qn in ("TL", "TR", "BL", "BR"):
        q = cur["per_quadrant"].get(qn)
        if not q:
            continue
        v, h = q["vertical"], q["horizontal"]
        print(
            f"    {qn}: v_mass={v['continuity_mass']:.0f} v_frag={v['fragmentation_index']:.3f} | "
            f"h_mass={h['continuity_mass']:.0f} h_frag={h['fragmentation_index']:.3f} fg={q['foreground_pixel_fraction']:.4f}"
        )
    rb = norm_info.get("regional_binarize") or {}
    if rb.get("enabled"):
        print("  regional tiles low_thresh_selected:", [t.get("low_thresh_selected") for t in rb.get("tiles", [])])
        print("  regional tiles histogram_p5:", [round(t.get("histogram_p5", 0), 2) for t in rb.get("tiles", [])])

    if "delta" in report:
        d = report["delta"]
        print("\nDelta vs baseline (current - baseline):")
        print(f"  vertical continuity_mass: {d['vertical']['continuity_mass']:+.0f}")
        print(f"  vertical fragmentation_index: {d['vertical']['fragmentation_index']:+.4f}")
        print(f"  horizontal continuity_mass: {d['horizontal']['continuity_mass']:+.0f}")
        print(f"  horizontal fragmentation_index: {d['horizontal']['fragmentation_index']:+.4f}")
        for qn in ("TL", "TR", "BL", "BR"):
            if qn in d.get("per_quadrant", {}):
                dq = d["per_quadrant"][qn]
                print(
                    f"  {qn} v_mass {dq['vertical']['continuity_mass']:+.0f} v_frag {dq['vertical']['fragmentation_index']:+.4f} | "
                    f"h_mass {dq['horizontal']['continuity_mass']:+.0f} h_frag {dq['horizontal']['fragmentation_index']:+.4f}"
                )

    if args.json_out:
        print(f"\nWrote JSON: {args.json_out}")
    return 0


def _delta_metrics(cur: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "vertical": {
            "continuity_mass": cur["vertical"]["continuity_mass"] - base["vertical"]["continuity_mass"],
            "fragmentation_index": cur["vertical"]["fragmentation_index"] - base["vertical"]["fragmentation_index"],
            "projection_peak_count": cur["vertical"]["projection_peak_count"] - base["vertical"]["projection_peak_count"],
        },
        "horizontal": {
            "continuity_mass": cur["horizontal"]["continuity_mass"] - base["horizontal"]["continuity_mass"],
            "fragmentation_index": cur["horizontal"]["fragmentation_index"] - base["horizontal"]["fragmentation_index"],
            "projection_peak_count": cur["horizontal"]["projection_peak_count"] - base["horizontal"]["projection_peak_count"],
        },
        "per_quadrant": {},
    }
    for qn in cur.get("per_quadrant", {}):
        if qn not in base.get("per_quadrant", {}):
            continue
        cq, bq = cur["per_quadrant"][qn], base["per_quadrant"][qn]
        out["per_quadrant"][qn] = {
            "vertical": {
                "continuity_mass": cq["vertical"]["continuity_mass"] - bq["vertical"]["continuity_mass"],
                "fragmentation_index": cq["vertical"]["fragmentation_index"] - bq["vertical"]["fragmentation_index"],
            },
            "horizontal": {
                "continuity_mass": cq["horizontal"]["continuity_mass"] - bq["horizontal"]["continuity_mass"],
                "fragmentation_index": cq["horizontal"]["fragmentation_index"] - bq["horizontal"]["fragmentation_index"],
            },
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
