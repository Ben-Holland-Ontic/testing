"""
Generate synthetic R-line page layouts for clustering validation.

Usage:
  py -3 tools/generate_synthetic_rline_pages.py
  py -3 tools/generate_synthetic_rline_pages.py --pages 36 --seed 11 --out tests/fixtures/synthetic_rline_pages.json
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PageSpec:
    name: str
    width: int
    height: int


PAGE_SPECS = [
    PageSpec("A4_PORTRAIT", 2480, 3508),
    PageSpec("A4_LANDSCAPE", 3508, 2480),
    PageSpec("A3_PORTRAIT", 3508, 4961),
    PageSpec("A3_LANDSCAPE", 4961, 3508),
]


def _make_line(line_id: str, text: str, x0: int, y0: int, x1: int, y1: int, group_id: str) -> dict[str, Any]:
    return {
        "line_id": line_id,
        "text": text,
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "group_id": group_id,
    }


def _jitter(v: int, rng: random.Random, spread: int) -> int:
    if spread <= 0:
        return v
    return int(v + rng.randint(-spread, spread))


def _add_block(
    blocks: list[dict[str, Any]],
    page_lines: list[dict[str, Any]],
    *,
    block_id: str,
    scenario: str,
    expected_current: int,
    expected_rhythm: int,
    lines: list[dict[str, Any]],
) -> None:
    blocks.append(
        {
            "block_id": block_id,
            "scenario": scenario,
            "expected_components_current": expected_current,
            "expected_components_rhythm": expected_rhythm,
            "line_ids": [ln["line_id"] for ln in lines],
        }
    )
    page_lines.extend(lines)


def _derive_block_truth(lines: list[dict[str, Any]]) -> int:
    groups = {str(ln.get("group_id") or "") for ln in lines if str(ln.get("group_id") or "")}
    return max(1, len(groups))


def _apply_ocr_noise(page: PageSpec, rng: random.Random, lines: list[dict[str, Any]], block_id: str) -> list[dict[str, Any]]:
    """Inject OCR-like geometric artifacts while preserving semantic group IDs."""
    out: list[dict[str, Any]] = []
    skew = rng.uniform(-0.012, 0.012)  # subtle scan skew
    for idx, line in enumerate(lines):
        b = list(line.get("bbox") or [0, 0, 0, 0])
        if len(b) < 4:
            continue
        x0, y0, x1, y1 = b
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)

        # Affine-like drift, independent edge jitter, and clipping.
        y_shift = int((x0 - page.width * 0.5) * skew)
        x0 = max(0, min(page.width - 2, x0 + rng.randint(-8, 10)))
        x1 = max(x0 + 1, min(page.width - 1, x1 + rng.randint(-10, 12)))
        y0 = max(0, min(page.height - 2, y0 + y_shift + rng.randint(-5, 5)))
        y1 = max(y0 + 1, min(page.height - 1, y1 + y_shift + rng.randint(-4, 6)))

        # Randomly trim edges to mimic detector clipping.
        if rng.random() < 0.18:
            x0 = min(x1 - 1, x0 + rng.randint(2, min(16, max(2, w // 8))))
        if rng.random() < 0.18:
            x1 = max(x0 + 1, x1 - rng.randint(2, min(18, max(2, w // 7))))

        noisy_line = dict(line)
        noisy_line["bbox"] = [x0, y0, x1, y1]

        # Split a line into two nearby fragments (same semantic group) to simulate OCR fragmentation.
        if rng.random() < 0.12 and (x1 - x0) > 180:
            cut = rng.randint(x0 + int((x1 - x0) * 0.35), x0 + int((x1 - x0) * 0.65))
            gap = rng.randint(4, 16)
            left = dict(noisy_line)
            right = dict(noisy_line)
            left["line_id"] = f"{line['line_id']}_N{idx}a"
            right["line_id"] = f"{line['line_id']}_N{idx}b"
            left["bbox"] = [x0, y0, cut, y1]
            right["bbox"] = [min(page.width - 1, cut + gap), y0 + rng.randint(-2, 2), x1, y1 + rng.randint(-1, 2)]
            out.extend([left, right])
        else:
            out.append(noisy_line)

        # Add sparse false positive short boxes (ignored by truth by dedicated group id).
        if rng.random() < 0.05:
            fx0 = max(0, min(page.width - 20, x0 + rng.randint(-30, 30)))
            fy0 = max(0, min(page.height - 10, y0 + rng.randint(-12, 12)))
            fw = rng.randint(14, 42)
            fh = rng.randint(8, 16)
            out.append(
                _make_line(
                    f"{block_id}_NOISE_{idx}",
                    "noise",
                    fx0,
                    fy0,
                    min(page.width - 1, fx0 + fw),
                    min(page.height - 1, fy0 + fh),
                    f"{block_id}__noise",
                )
            )
    return out


def paragraph_left_taper(page: PageSpec, rng: random.Random, block_id: str, y_start: int) -> tuple[list[dict[str, Any]], int]:
    x0 = int(page.width * rng.uniform(0.10, 0.20))
    max_w = int(page.width * rng.uniform(0.40, 0.62))
    h = rng.randint(34, 48)
    gap = rng.randint(9, 19)
    n = rng.randint(5, 8)
    lines: list[dict[str, Any]] = []
    y = y_start
    for i in range(n):
        shrink = int((i / max(1, n - 1)) * max_w * rng.uniform(0.25, 0.58))
        w = max_w - shrink
        line_x0 = _jitter(x0, rng, spread=10)
        x1 = line_x0 + max(120, w)
        # Occasional hanging-indent line to create harder x-overlap cases.
        if i > 0 and rng.random() < 0.18:
            line_x0 += rng.randint(12, 28)
            x1 += rng.randint(8, 22)
        lines.append(_make_line(f"{block_id}_L{i+1}", f"Paragraph line {i+1}", line_x0, y, x1, y + h, block_id))
        y += h + _jitter(gap, rng, spread=6)
    return lines, y + rng.randint(25, 80)


def paragraph_center_taper(page: PageSpec, rng: random.Random, block_id: str, y_start: int) -> tuple[list[dict[str, Any]], int]:
    cx = int(page.width * rng.uniform(0.45, 0.55))
    max_w = int(page.width * rng.uniform(0.38, 0.55))
    h = rng.randint(34, 48)
    gap = rng.randint(8, 16)
    n = rng.randint(5, 8)
    lines: list[dict[str, Any]] = []
    y = y_start
    for i in range(n):
        wave = (i % 2) * rng.uniform(0.03, 0.08)
        shrink = int((i / max(1, n - 1)) * max_w * (0.20 + wave))
        w = max(130, max_w - shrink)
        # Simulate real centered blocks with imperfect center alignment.
        center_shift = rng.randint(-18, 18)
        x0 = cx - w // 2 + center_shift
        x1 = cx + w // 2 + center_shift
        if i == n - 1 and rng.random() < 0.6:
            # Common short final centered line.
            x0 += rng.randint(30, 90)
            x1 -= rng.randint(30, 90)
        lines.append(_make_line(f"{block_id}_L{i+1}", f"Centered paragraph {i+1}", x0, y, x1, y + h, block_id))
        y += h + _jitter(gap, rng, spread=5)
    return lines, y + rng.randint(25, 75)


def numbered_list_with_gaps(page: PageSpec, rng: random.Random, block_id: str, y_start: int) -> tuple[list[dict[str, Any]], int]:
    bullet_x = int(page.width * rng.uniform(0.08, 0.16))
    text_x = bullet_x + rng.randint(45, 85)
    max_w = int(page.width * rng.uniform(0.34, 0.50))
    h = rng.randint(30, 42)
    n = rng.randint(6, 10)
    lines: list[dict[str, Any]] = []
    y = y_start
    for i in range(1, n + 1):
        item_gap = rng.randint(10, 34)
        cont_gap = rng.randint(6, 14)
        lead_w = max_w - rng.randint(0, max(20, max_w // 6))
        cont_w = max(120, lead_w - rng.randint(50, 220))
        lead_x = _jitter(bullet_x, rng, spread=8)
        cont_x = _jitter(text_x, rng, spread=10)
        lines.append(_make_line(f"{block_id}_I{i}", f"{i}. Item {i}", lead_x, y, lead_x + lead_w, y + h, block_id))
        y += h + cont_gap
        lines.append(_make_line(f"{block_id}_C{i}", f"Continuation {i}", cont_x, y, cont_x + cont_w, y + h, block_id))
        # occasional second continuation line with different indent
        if rng.random() < 0.35:
            y += h + rng.randint(2, 8)
            cont2_x = cont_x + rng.randint(8, 26)
            cont2_w = max(100, cont_w - rng.randint(40, 140))
            lines.append(_make_line(f"{block_id}_C{i}b", f"Continuation {i}b", cont2_x, y, cont2_x + cont2_w, y + h, block_id))
        y += h + _jitter(item_gap, rng, spread=10)
    return lines, y + rng.randint(20, 60)


def two_lists_side_by_side(page: PageSpec, rng: random.Random, block_id: str, y_start: int) -> tuple[list[dict[str, Any]], int]:
    left_x = int(page.width * rng.uniform(0.06, 0.12))
    right_x = int(page.width * rng.uniform(0.55, 0.66))
    h = rng.randint(30, 42)
    n = rng.randint(5, 8)
    y = y_start
    lines: list[dict[str, Any]] = []
    for i in range(n):
        gap = rng.randint(6, 18)
        # Deliberately stagger y to produce imperfect interleaving patterns.
        ly = y + rng.randint(-7, 7)
        ry = y + rng.randint(-7, 7)
        lw = rng.randint(int(page.width * 0.18), int(page.width * 0.28))
        rw = rng.randint(int(page.width * 0.18), int(page.width * 0.28))
        lx = _jitter(left_x, rng, spread=14)
        rx = _jitter(right_x, rng, spread=14)
        lines.append(_make_line(f"{block_id}_L{i+1}", f"L{i+1}", lx, ly, lx + lw, ly + h, f"{block_id}_LEFT"))
        lines.append(_make_line(f"{block_id}_R{i+1}", f"R{i+1}", rx, ry, rx + rw, ry + h, f"{block_id}_RIGHT"))
        y += h + gap
    return lines, y + rng.randint(20, 60)


def header_body_gap(page: PageSpec, rng: random.Random, block_id: str, y_start: int) -> tuple[list[dict[str, Any]], int]:
    x0 = int(page.width * rng.uniform(0.10, 0.18))
    h = rng.randint(32, 46)
    body_gap = rng.randint(8, 14)
    header_gap = rng.randint(55, 95)
    header_w = int(page.width * rng.uniform(0.25, 0.36))
    body_w = int(page.width * rng.uniform(0.35, 0.52))
    lines: list[dict[str, Any]] = []
    y = y_start
    lines.append(_make_line(f"{block_id}_H", "Header", _jitter(x0, rng, 6), y, x0 + header_w, y + h, block_id))
    y += h + header_gap
    n = rng.randint(4, 7)
    for i in range(n):
        shrink = int((i / max(1, n - 1)) * body_w * rng.uniform(0.10, 0.34))
        w = max(150, body_w - shrink)
        line_x = _jitter(x0, rng, spread=10)
        lines.append(_make_line(f"{block_id}_B{i+1}", f"Body {i+1}", line_x, y, line_x + w, y + h, block_id))
        y += h + _jitter(body_gap, rng, spread=6)
    return lines, y + rng.randint(15, 55)


def ragged_paragraph_realistic(page: PageSpec, rng: random.Random, block_id: str, y_start: int) -> tuple[list[dict[str, Any]], int]:
    base_x = int(page.width * rng.uniform(0.09, 0.22))
    base_w = int(page.width * rng.uniform(0.34, 0.58))
    h = rng.randint(30, 44)
    n = rng.randint(6, 12)
    y = y_start
    lines: list[dict[str, Any]] = []
    for i in range(n):
        # Mix of justified, ragged-right, and occasional hanging indent lines.
        x_shift = rng.randint(-14, 18)
        x0 = base_x + x_shift
        if rng.random() < 0.22 and i > 0:
            x0 += rng.randint(10, 32)
        width_drop = int((i / max(1, n - 1)) * base_w * rng.uniform(0.05, 0.45))
        random_drop = rng.randint(0, int(base_w * 0.16))
        x1 = x0 + max(120, base_w - width_drop - random_drop)
        lines.append(_make_line(f"{block_id}_P{i+1}", f"Ragged paragraph {i+1}", x0, y, x1, y + h, block_id))
        # Real docs have variable leading due to OCR/crop effects.
        y += h + rng.randint(4, 22)
    return lines, y + rng.randint(18, 70)


def mixed_bullets_indents(page: PageSpec, rng: random.Random, block_id: str, y_start: int) -> tuple[list[dict[str, Any]], int]:
    marker_x = int(page.width * rng.uniform(0.07, 0.15))
    body_x = marker_x + rng.randint(36, 92)
    body_w = int(page.width * rng.uniform(0.30, 0.46))
    h = rng.randint(28, 40)
    y = y_start
    lines: list[dict[str, Any]] = []
    marker_set = ["1.", "2.", "3.", "-", "•", "a)"]
    n = rng.randint(6, 10)
    for i in range(n):
        mk = marker_set[i % len(marker_set)]
        mx = _jitter(marker_x, rng, spread=10)
        bw = body_w - rng.randint(0, int(body_w * 0.20))
        lines.append(_make_line(f"{block_id}_M{i+1}", f"{mk} item", mx, y, mx + bw, y + h, block_id))
        y += h + rng.randint(3, 11)
        if rng.random() < 0.8:
            cx = _jitter(body_x, rng, spread=16)
            cw = max(100, bw - rng.randint(30, 170))
            lines.append(_make_line(f"{block_id}_B{i+1}", f"detail {i+1}", cx, y, cx + cw, y + h, block_id))
            y += h + rng.randint(8, 22)
        else:
            y += rng.randint(10, 26)
    return lines, y + rng.randint(20, 80)


def twin_columns_with_callouts(page: PageSpec, rng: random.Random, block_id: str, y_start: int) -> tuple[list[dict[str, Any]], int]:
    """Two dense columns plus crossing callout snippets: hard merge-separation stress."""
    lx = int(page.width * rng.uniform(0.06, 0.12))
    rx = int(page.width * rng.uniform(0.54, 0.66))
    base_h = rng.randint(28, 38)
    n = rng.randint(7, 11)
    y = y_start
    lines: list[dict[str, Any]] = []
    for i in range(n):
        lh = base_h + rng.randint(-4, 6)
        rh = base_h + rng.randint(-4, 6)
        lw = rng.randint(int(page.width * 0.15), int(page.width * 0.26))
        rw = rng.randint(int(page.width * 0.15), int(page.width * 0.26))
        ly = y + rng.randint(-6, 9)
        ry = y + rng.randint(-8, 8)
        lines.append(_make_line(f"{block_id}_L{i+1}", f"L{i+1}", _jitter(lx, rng, 12), ly, _jitter(lx, rng, 12) + lw, ly + lh, f"{block_id}_LEFT"))
        lines.append(_make_line(f"{block_id}_R{i+1}", f"R{i+1}", _jitter(rx, rng, 12), ry, _jitter(rx, rng, 12) + rw, ry + rh, f"{block_id}_RIGHT"))
        # occasional in-between callout anchored to one side (keeps truth as side-specific)
        if rng.random() < 0.35:
            cx = rng.randint(lx + lw + 18, rx - 40)
            cw = rng.randint(70, 160)
            cy = y + rng.randint(-4, 10)
            gid = f"{block_id}_LEFT" if rng.random() < 0.5 else f"{block_id}_RIGHT"
            lines.append(_make_line(f"{block_id}_C{i+1}", "callout", cx, cy, min(page.width - 1, cx + cw), cy + rng.randint(18, 30), gid))
        y += base_h + rng.randint(6, 18)
    return lines, y + rng.randint(20, 70)


def tri_column_staggered_lists(page: PageSpec, rng: random.Random, block_id: str, y_start: int) -> tuple[list[dict[str, Any]], int]:
    cols = [int(page.width * p) for p in (0.07, 0.38, 0.69)]
    gids = [f"{block_id}_C1", f"{block_id}_C2", f"{block_id}_C3"]
    h = rng.randint(26, 36)
    y = y_start
    lines: list[dict[str, Any]] = []
    n = rng.randint(5, 9)
    for i in range(n):
        for c_idx, cx in enumerate(cols):
            if rng.random() < 0.12:
                continue
            yy = y + rng.randint(-7, 7) + c_idx * rng.randint(-2, 3)
            w = rng.randint(int(page.width * 0.10), int(page.width * 0.18))
            lines.append(_make_line(f"{block_id}_C{c_idx+1}_{i+1}", f"{c_idx+1}.{i+1}", _jitter(cx, rng, 10), yy, _jitter(cx, rng, 10) + w, yy + h, gids[c_idx]))
        y += h + rng.randint(7, 20)
    return lines, y + rng.randint(24, 80)


def aligned_enumeration_with_large_gaps(page: PageSpec, rng: random.Random, block_id: str, y_start: int) -> tuple[list[dict[str, Any]], int]:
    """Single aligned enumeration where some items have large vertical gaps."""
    num_x = int(page.width * rng.uniform(0.08, 0.14))
    text_x = num_x + rng.randint(42, 78)
    h = rng.randint(28, 38)
    n = rng.randint(8, 14)
    y = y_start
    lines: list[dict[str, Any]] = []
    for i in range(1, n + 1):
        # Keep strong vertical alignment with small x jitter only.
        nx = _jitter(num_x, rng, spread=4)
        tx = _jitter(text_x, rng, spread=6)
        tw = rng.randint(int(page.width * 0.24), int(page.width * 0.40))
        lines.append(_make_line(f"{block_id}_E{i}", f"{i}. enumerated item {i}", nx, y, tx + tw, y + h, block_id))

        # Optional continuation line with same left text alignment.
        if rng.random() < 0.42:
            cy = y + h + rng.randint(4, 12)
            cw = max(120, tw - rng.randint(30, 180))
            lines.append(_make_line(f"{block_id}_EC{i}", f"continuation {i}", tx, cy, tx + cw, cy + h, block_id))
            y = cy + h
        else:
            y += h

        # Irregular vertical spacing: mostly normal gaps, with occasional large separator gaps.
        if rng.random() < 0.28:
            y += rng.randint(40, 95)   # large gap between aligned enumerations
        else:
            y += rng.randint(8, 20)    # normal list gap
    return lines, y + rng.randint(20, 60)


SCENARIOS = [
    ("paragraph_left_taper", paragraph_left_taper, 2, 1),
    ("paragraph_center_taper", paragraph_center_taper, 2, 1),
    ("numbered_list_with_gaps", numbered_list_with_gaps, 2, 1),
    ("two_lists_side_by_side", two_lists_side_by_side, 2, 2),
    ("header_body_gap", header_body_gap, 2, 1),
    ("ragged_paragraph_realistic", ragged_paragraph_realistic, 2, 1),
    ("mixed_bullets_indents", mixed_bullets_indents, 2, 1),
    ("twin_columns_with_callouts", twin_columns_with_callouts, 2, 2),
    ("tri_column_staggered_lists", tri_column_staggered_lists, 3, 3),
    ("aligned_enumeration_with_large_gaps", aligned_enumeration_with_large_gaps, 1, 1),
]


def build_dataset(num_pages: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    pages: list[dict[str, Any]] = []
    for p in range(num_pages):
        page_spec = rng.choice(PAGE_SPECS)
        lines: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []
        y = rng.randint(int(page_spec.height * 0.05), int(page_spec.height * 0.12))
        n_blocks = rng.randint(2, 4)
        for b in range(n_blocks):
            scenario_name, fn, expected_current, expected_rhythm = rng.choice(SCENARIOS)
            block_id = f"P{p+1}_B{b+1}"
            block_lines, next_y = fn(page_spec, rng, block_id, y)
            block_lines = _apply_ocr_noise(page_spec, rng, block_lines, block_id)
            truth_regions = _derive_block_truth([ln for ln in block_lines if "__noise" not in str(ln.get("group_id") or "")])
            _add_block(
                blocks,
                lines,
                block_id=block_id,
                scenario=scenario_name,
                expected_current=truth_regions,
                expected_rhythm=truth_regions,
                lines=block_lines,
            )
            blocks[-1]["expected_components_declared"] = {
                "legacy_current": int(expected_current),
                "legacy_rhythm": int(expected_rhythm),
            }
            blocks[-1]["ground_truth_regions"] = int(truth_regions)
            y = next_y
            if y > int(page_spec.height * 0.82):
                break
        pages.append(
            {
                "page_id": f"SYNTH_{p+1:03d}",
                "page_size": page_spec.name,
                "width": page_spec.width,
                "height": page_spec.height,
                "seed": seed,
                "blocks": blocks,
                "lines": lines,
            }
        )
    return {
        "schema_version": "1.1",
        "description": "Synthetic line boxes with OCR-style noise and derived per-block ground truth regions.",
        "seed": seed,
        "num_pages": num_pages,
        "pages": pages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic R-line page fixtures.")
    parser.add_argument("--pages", type=int, default=24, help="Number of pages to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/synthetic_rline_pages.json"),
        help="Output JSON path (relative to bk-main root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_dataset(num_pages=max(1, args.pages), seed=args.seed)
    out_path = args.out
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent.parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {data['num_pages']} synthetic pages to {out_path}")


if __name__ == "__main__":
    main()
