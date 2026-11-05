"""
Render synthetic R-line fixtures into visual debug PNGs.

Usage:
  py -3 tools/visualize_synthetic_rline_pages.py
  py -3 tools/visualize_synthetic_rline_pages.py --max-pages 10 --show-text 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import networkx as nx
import numpy as np


DEFAULT_CFG = {
    "X_OVERLAP_INDENT_TOLERANCE_MULT": 2.0,
    "X_OVERLAP_INDENT_TOLERANCE_FLOOR_PX": 80,
    "X_OVERLAP_PROXIMITY_RELAX_MAX": 1.5,
    "X_OVERLAP_RHYTHM_BONUS_MAX": 2.0,
    "X_OVERLAP_RHYTHM_CV_CAP": 1.0,
    "X_OVERLAP_RHYTHM_BONUS_ENABLED": True,
    "CLUSTER_SAME_BAND_HORIZ_GAP_MULT": 0.30,
    "CLUSTER_VERTICAL_ADJ_HORIZ_GAP_MULT": 0.40,
    "CLUSTER_DISTANCE_ADAPTIVE_ENABLED": True,
    "CLUSTER_DISTANCE_WHITESPACE_WEIGHT": 0.65,
    "CLUSTER_DISTANCE_MIN_SCALE": 0.35,
    "CLUSTER_WHITESPACE_CAP_PRESSURE_BOOST": 1.0,
    "X_OVERLAP_WHITESPACE_PENALTY_ENABLED": True,
    "X_OVERLAP_WHITESPACE_MAX_REDUCTION": 0.75,
}


SCENARIO_COLORS = {
    "paragraph_left_taper": (70, 130, 180),
    "paragraph_center_taper": (90, 180, 90),
    "numbered_list_with_gaps": (0, 165, 255),
    "header_body_gap": (180, 105, 255),
    "two_lists_side_by_side": (220, 120, 40),
}


COMPONENT_PALETTE = [
    (50, 180, 255),
    (255, 100, 100),
    (120, 255, 120),
    (255, 220, 80),
    (220, 120, 255),
    (255, 170, 80),
    (80, 220, 220),
]


def _bbox(line: dict[str, Any]) -> list[float]:
    b = line.get("bbox") or [0, 0, 0, 0]
    if len(b) != 4:
        return [0, 0, 0, 0]
    return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]


def _line_height(lines: list[dict[str, Any]]) -> float:
    hs = []
    for l in lines:
        b = _bbox(l)
        hs.append(max(1.0, b[3] - b[1]))
    return float(np.median(hs)) if hs else 1.0


def _rhythm_profile(lines: list[dict[str, Any]]) -> tuple[float, float]:
    order = sorted(range(len(lines)), key=lambda i: (_bbox(lines[i])[1], _bbox(lines[i])[0]))
    gaps = []
    for k in range(len(order) - 1):
        a = _bbox(lines[order[k]])
        b = _bbox(lines[order[k + 1]])
        vg = b[1] - a[3]
        if vg >= 0:
            gaps.append(float(vg))
    if not gaps:
        return 0.0, 1.0
    if len(gaps) >= 3:
        rhythm_gap = float(np.percentile(gaps, 25))
        sorted_gaps = sorted(gaps)
        trimmed = sorted_gaps[:-1]
        denom = max(float(np.mean(trimmed)), 1.0)
        rhythm_cv = float(np.std(trimmed)) / denom
        return rhythm_gap, rhythm_cv
    if len(gaps) == 2:
        return float(min(gaps)), 0.4
    # len == 1
    median_h = _line_height(lines)
    rhythm_cv = 0.3 if gaps[0] < 3.0 * median_h else 1.0
    return float(gaps[0]), rhythm_cv


def _components(lines: list[dict[str, Any]], threshold: float, use_rhythm: bool, cfg: dict[str, Any]) -> list[set[int]]:
    g = nx.Graph()
    for i in range(len(lines)):
        g.add_node(i)
    width = max(1.0, max((_bbox(l)[2] for l in lines), default=1.0))
    rhythm_gap, rhythm_cv = _rhythm_profile(lines)
    widths = [max(1.0, _bbox(l)[2] - _bbox(l)[0]) for l in lines]
    median_width = float(np.median(widths)) if widths else 1.0
    same_band_h_samples = []
    order = sorted(range(len(lines)), key=lambda i: (_bbox(lines[i])[1], _bbox(lines[i])[0]))
    for k in range(len(order) - 1):
        a = _bbox(lines[order[k]])
        b = _bbox(lines[order[k + 1]])
        if a[2] < b[0]:
            hh = b[0] - a[2]
        elif b[2] < a[0]:
            hh = a[0] - b[2]
        else:
            hh = 0.0
        if a[3] < b[1]:
            vv = b[1] - a[3]
        elif b[3] < a[1]:
            vv = a[1] - b[3]
        else:
            vv = 0.0
        lh = max(1.0, max(a[3] - a[1], b[3] - b[1]))
        if hh > 0 and vv <= lh:
            same_band_h_samples.append(float(hh))
    same_band_mult = float(cfg.get("CLUSTER_SAME_BAND_HORIZ_GAP_MULT", 0.30))
    vert_adj_mult = float(cfg.get("CLUSTER_VERTICAL_ADJ_HORIZ_GAP_MULT", 0.40))
    same_band_cap = max(22.0, same_band_mult * max(1.0, median_width))
    vert_adj_cap = max(same_band_cap, max(30.0, vert_adj_mult * max(1.0, median_width)))
    for i in range(len(lines)):
        li = _bbox(lines[i])
        for j in range(i + 1, len(lines)):
            lj = _bbox(lines[j])
            if li[2] < lj[0]:
                h = lj[0] - li[2]
            elif lj[2] < li[0]:
                h = li[0] - lj[2]
            else:
                h = 0.0
            if li[3] < lj[1]:
                v = lj[1] - li[3]
            elif lj[3] < li[1]:
                v = li[1] - lj[3]
            else:
                v = 0.0
            x_overlap = min(li[2], lj[2]) - max(li[0], lj[0])
            line_height = max(li[3] - li[1], lj[3] - lj[1], 1.0)
            wi = max(1.0, li[2] - li[0])
            wj = max(1.0, lj[2] - lj[0])
            dist = max(h, v)

            whitespace_pressure = 0.0
            cap_pressure = 0.0
            if h > 0 and v <= line_height and h > same_band_cap:
                excess_same = (h - same_band_cap) / max(same_band_cap, 1.0)
                cap_pressure = max(cap_pressure, min(1.0, 0.35 + excess_same))
            if h > 0 and v > 0 and h > vert_adj_cap:
                excess_vert = (h - vert_adj_cap) / max(vert_adj_cap, 1.0)
                cap_pressure = max(cap_pressure, min(1.0, 0.45 + excess_vert))

            effective_threshold = float(threshold)
            if bool(cfg.get("CLUSTER_DISTANCE_ADAPTIVE_ENABLED", True)) and h > 0:
                w_norm = h / max(1.0, min(wi, wj))
                same_band_like = v <= (1.25 * line_height)
                vert_factor = 1.0 if same_band_like else max(0.35, 1.0 - min(v / max(3.0 * line_height, 1.0), 0.65))
                whitespace_pressure = min(1.0, max(0.0, w_norm)) * vert_factor
                if cap_pressure > 0.0:
                    boost = max(0.0, float(cfg.get("CLUSTER_WHITESPACE_CAP_PRESSURE_BOOST", 1.0)))
                    whitespace_pressure = min(1.0, max(whitespace_pressure, min(1.0, cap_pressure * boost)))
                weight = max(0.0, float(cfg.get("CLUSTER_DISTANCE_WHITESPACE_WEIGHT", 0.65)))
                min_scale = min(1.0, max(0.10, float(cfg.get("CLUSTER_DISTANCE_MIN_SCALE", 0.35))))
                scale = max(min_scale, 1.0 - (weight * whitespace_pressure))
                effective_threshold = float(threshold) * scale

            if dist > effective_threshold:
                continue
            indent_tol = max(
                float(cfg["X_OVERLAP_INDENT_TOLERANCE_FLOOR_PX"]),
                float(cfg["X_OVERLAP_INDENT_TOLERANCE_MULT"]) * line_height,
            )
            v_ratio = float(v) / float(max(line_height, 1.0))
            proximity_relax = max(0.0, 1.5 - v_ratio)
            proximity_relax = min(proximity_relax, float(cfg["X_OVERLAP_PROXIMITY_RELAX_MAX"]))
            adaptive_indent = indent_tol * (1.0 + proximity_relax)
            if (
                use_rhythm
                and bool(cfg.get("X_OVERLAP_RHYTHM_BONUS_ENABLED", True))
                and rhythm_gap > 0.0
                and v >= 0.0
            ):
                gap_deviation = abs(v - rhythm_gap) / max(rhythm_gap, 1.0)
                cv_cap = float(cfg.get("X_OVERLAP_RHYTHM_CV_CAP", 1.0))
                cv_factor = max(0.0, 1.0 - min(rhythm_cv, cv_cap))
                dev_factor = max(0.0, 1.0 - min(gap_deviation, 1.0))
                rhythm_bonus = float(cfg.get("X_OVERLAP_RHYTHM_BONUS_MAX", 2.0)) * cv_factor * dev_factor
                adaptive_indent *= (1.0 + rhythm_bonus)
            if (
                bool(cfg.get("X_OVERLAP_WHITESPACE_PENALTY_ENABLED", True))
                and h > 0
                and x_overlap < 0
                and whitespace_pressure > 0.0
            ):
                effective_pressure = whitespace_pressure
                max_reduction = min(0.95, max(0.0, float(cfg.get("X_OVERLAP_WHITESPACE_MAX_REDUCTION", 0.75))))
                penalty_scale = max(0.1, 1.0 - (max_reduction * effective_pressure))
                adaptive_indent *= penalty_scale
            if v > 0 and x_overlap < -adaptive_indent:
                continue
            if v == 0 and h > 0 and x_overlap < -indent_tol:
                continue
            g.add_edge(i, j)
    return list(nx.connected_components(g))


def _draw_block_components(
    canvas: np.ndarray,
    block_lines: list[dict[str, Any]],
    components: list[set[int]],
    show_text: bool,
) -> None:
    comp_lookup: dict[int, int] = {}
    for ci, comp in enumerate(components):
        for idx in comp:
            comp_lookup[idx] = ci
    for idx, line in enumerate(block_lines):
        b = _bbox(line)
        cidx = comp_lookup.get(idx, 0) % len(COMPONENT_PALETTE)
        color = COMPONENT_PALETTE[cidx]
        p1 = (int(b[0]), int(b[1]))
        p2 = (int(b[2]), int(b[3]))
        cv2.rectangle(canvas, p1, p2, color, 2)
        if show_text:
            label = f"C{comp_lookup.get(idx, 0)}:{line.get('line_id', f'L{idx}')}"
            cv2.putText(canvas, label, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    # Draw explicit component hull boxes so split components are unambiguous.
    for ci, comp in enumerate(components):
        cidx = ci % len(COMPONENT_PALETTE)
        color = COMPONENT_PALETTE[cidx]
        cbbs = [_bbox(block_lines[idx]) for idx in comp if 0 <= idx < len(block_lines)]
        if not cbbs:
            continue
        x0 = int(min(b[0] for b in cbbs))
        y0 = int(min(b[1] for b in cbbs))
        x1 = int(max(b[2] for b in cbbs))
        y1 = int(max(b[3] for b in cbbs))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3)
        cv2.putText(canvas, f"C{ci}", (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _annotate_block_counts(
    canvas: np.ndarray,
    block: dict[str, Any],
    line_by_id: dict[str, dict[str, Any]],
    actual_components: int,
    expected_components: int,
) -> bool:
    line_ids = block.get("line_ids", [])
    bbs = [_bbox(line_by_id[lid]) for lid in line_ids if lid in line_by_id]
    if not bbs:
        return True
    x0 = int(min(b[0] for b in bbs))
    y0 = int(min(b[1] for b in bbs))
    x1 = int(max(b[2] for b in bbs))
    y1 = int(max(b[3] for b in bbs))
    ok = (expected_components <= 0) or (actual_components == expected_components)
    color = (60, 170, 60) if ok else (50, 50, 220)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 1)
    text = f"{block.get('block_id','B')} exp:{expected_components} act:{actual_components}"
    cv2.putText(canvas, text, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    return ok


def _draw_scenarios(canvas: np.ndarray, page: dict[str, Any], line_by_id: dict[str, dict[str, Any]]) -> None:
    for block in page.get("blocks", []):
        scenario = block.get("scenario", "")
        color = SCENARIO_COLORS.get(scenario, (200, 200, 200))
        line_ids = block.get("line_ids", [])
        bbs = [_bbox(line_by_id[lid]) for lid in line_ids if lid in line_by_id]
        if not bbs:
            continue
        x0 = int(min(b[0] for b in bbs))
        y0 = int(min(b[1] for b in bbs))
        x1 = int(max(b[2] for b in bbs))
        y1 = int(max(b[3] for b in bbs))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3)
        cv2.putText(canvas, scenario, (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def _render_page(page: dict[str, Any], out_dir: Path, show_text: bool, scale: float, show_scenarios: bool) -> tuple[int, int]:
    width = int(page["width"])
    height = int(page["height"])
    raw = np.full((height, width, 3), 255, dtype=np.uint8)
    baseline = raw.copy()
    rhythm = raw.copy()

    line_by_id = {l["line_id"]: l for l in page.get("lines", [])}
    if show_scenarios:
        _draw_scenarios(raw, page, line_by_id)
        _draw_scenarios(baseline, page, line_by_id)
        _draw_scenarios(rhythm, page, line_by_id)

    total_baseline = 0
    total_rhythm = 0
    mismatch_baseline = 0
    mismatch_rhythm = 0
    for block in page.get("blocks", []):
        block_lines = [line_by_id[lid] for lid in block.get("line_ids", []) if lid in line_by_id]
        if not block_lines:
            continue
        b_components = _components(block_lines, threshold=80.0, use_rhythm=False, cfg=DEFAULT_CFG)
        r_components = _components(block_lines, threshold=80.0, use_rhythm=True, cfg=DEFAULT_CFG)
        total_baseline += len(b_components)
        total_rhythm += len(r_components)
        _draw_block_components(baseline, block_lines, b_components, show_text)
        _draw_block_components(rhythm, block_lines, r_components, show_text)
        expected_b = int(block.get("expected_components_current", 0))
        expected_r = int(block.get("expected_components_rhythm", 0))
        ok_b = _annotate_block_counts(baseline, block, line_by_id, len(b_components), expected_b)
        ok_r = _annotate_block_counts(rhythm, block, line_by_id, len(r_components), expected_r)
        if not ok_b:
            mismatch_baseline += 1
        if not ok_r:
            mismatch_rhythm += 1

    header_h = 76
    board = np.full((height + header_h, width * 3, 3), 245, dtype=np.uint8)
    board[header_h:, 0:width] = raw
    board[header_h:, width : width * 2] = baseline
    board[header_h:, width * 2 : width * 3] = rhythm

    cv2.putText(board, f"{page['page_id']} | {page['page_size']}", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(board, "Scenarios", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(board, f"Baseline components: {total_baseline}", (width + 20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(board, f"Rhythm components: {total_rhythm}", (width * 2 + 20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(board, f"Baseline mismatches: {mismatch_baseline}", (width + 430, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (55, 55, 55), 2, cv2.LINE_AA)
    cv2.putText(board, f"Rhythm mismatches: {mismatch_rhythm}", (width * 2 + 430, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (55, 55, 55), 2, cv2.LINE_AA)

    if scale and scale > 0 and scale != 1.0:
        board = cv2.resize(
            board,
            dsize=(int(board.shape[1] * scale), int(board.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    out_file = out_dir / f"{page['page_id']}.png"
    ok = cv2.imwrite(str(out_file), board)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_file}")
    return total_baseline, total_rhythm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize synthetic R-line fixtures.")
    parser.add_argument(
        "--in-json",
        type=Path,
        default=Path("tests/fixtures/synthetic_rline_pages.json"),
        help="Input synthetic fixture JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/fixtures/synthetic_rline_pages_viz"),
        help="Output folder for PNGs.",
    )
    parser.add_argument("--max-pages", type=int, default=0, help="Render only the first N pages (0=all).")
    parser.add_argument("--show-text", type=int, default=0, help="Draw line IDs on boxes (0/1).")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.35,
        help="Output image scale factor for large A3 pages (1.0 keeps full size).",
    )
    parser.add_argument(
        "--show-scenarios",
        type=int,
        default=0,
        help="Draw scenario envelope boxes (0/1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = args.in_json
    if not in_path.is_absolute():
        in_path = Path(__file__).resolve().parent.parent / in_path
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    pages = data.get("pages", [])
    if args.max_pages > 0:
        pages = pages[: args.max_pages]

    baseline_sum = 0
    rhythm_sum = 0
    for page in pages:
        b, r = _render_page(page, out_dir, bool(args.show_text), float(args.scale), bool(args.show_scenarios))
        baseline_sum += b
        rhythm_sum += r

    summary = {
        "input_json": str(in_path),
        "output_dir": str(out_dir),
        "pages_rendered": len(pages),
        "total_baseline_components": baseline_sum,
        "total_rhythm_components": rhythm_sum,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
