"""Build a CANONICAL VIEW of a page's cluster regions for an LLM.

DESIGN INTENT. Earlier probes handed the model OUR conclusions -- is_part_of_grid,
grid_row_idx, grid_col_idx, grid_rows x grid_cols. That is the pipeline's inference, not
evidence, and it either biased the answer or drowned it: with all 34 signals per region the
model collapsed into digit noise at ~9.4k tokens.

So this presents RAW SPATIAL TRUTH only:
  - poly id
  - the text the cell contains
  - where it is on the page

plus a small number of HOLISTIC signals about the set as a whole -- never per-cell row or
column indices, because inferring rows and columns is the question being asked.

Three renderings are offered because presentation has been the dominant variable in every
test so far, and which one works is an empirical question:

  "list"    one line per region, absolute pixel coordinates
  "norm"    same, normalised to a 0-1000 page grid (scale-free)
  "map"     an ASCII spatial map -- the layout drawn, not described

The ASCII map exists because a language model has seen far more text laid out in columns
than it has seen coordinate lists, and column alignment is visible in a monospace grid
without any arithmetic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _cells(page: Dict[str, Any], x_max: float = 1e9, y_max: float = 1e9) -> List[Dict]:
    out = []
    for c in page.get("cluster_lines", []):
        bb = c.get("bbox")
        if not bb or not c.get("text_raw"):
            continue
        if bb[0] > x_max or bb[1] > y_max:
            continue
        out.append(c)
    out.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    return out


def render_list(cells: Sequence[Dict]) -> str:
    rows = ["poly_id, x0, y0, x1, y1, text"]
    for c in cells:
        b = [int(v) for v in c["bbox"]]
        rows.append(f'{c["poly_id"]}, {b[0]}, {b[1]}, {b[2]}, {b[3]}, "{c["text_raw"]}"')
    return "\n".join(rows)


def render_norm(cells: Sequence[Dict]) -> str:
    """Normalised to 0-1000 so the model never reasons about page scale."""
    xs = [v for c in cells for v in (c["bbox"][0], c["bbox"][2])]
    ys = [v for c in cells for v in (c["bbox"][1], c["bbox"][3])]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    sx = 1000.0 / max(1.0, x1 - x0)
    sy = 1000.0 / max(1.0, y1 - y0)
    rows = ["poly_id, left, top, right, bottom, text   (page normalised to 0-1000)"]
    for c in cells:
        b = c["bbox"]
        rows.append(f'{c["poly_id"]}, {int((b[0]-x0)*sx)}, {int((b[1]-y0)*sy)}, '
                    f'{int((b[2]-x0)*sx)}, {int((b[3]-y0)*sy)}, "{c["text_raw"]}"')
    return "\n".join(rows)


def render_map(cells: Sequence[Dict], width: int = 96, height: int = 34) -> str:
    """Draw the layout in a monospace grid: alignment becomes visible, not computed."""
    if not cells:
        return ""
    xs = [v for c in cells for v in (c["bbox"][0], c["bbox"][2])]
    ys = [v for c in cells for v in (c["bbox"][1], c["bbox"][3])]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    sx = (width - 1) / max(1.0, x1 - x0)
    sy = (height - 1) / max(1.0, y1 - y0)
    canvas = [[" "] * width for _ in range(height)]
    for c in cells:
        b = c["bbox"]
        col = int((b[0] - x0) * sx)
        row = int(((b[1] + b[3]) / 2 - y0) * sy)
        row = max(0, min(height - 1, row))
        txt = str(c["text_raw"])
        for i, ch in enumerate(txt):
            p = col + i
            if 0 <= p < width and canvas[row][p] == " ":
                canvas[row][p] = ch
    return "\n".join("".join(r).rstrip() for r in canvas)


def render_doc(cells: Sequence[Dict], feats: Dict[str, Dict] | None = None,
               width: int = 100, height: int = 40) -> str:
    """Redraw the DOCUMENT: cell boundaries AND the text inside them.

    render_map draws text alone, so a ruled table and a loose column of labels look
    identical. Here the detected polygon of every cluster region is drawn as a box, with
    its text placed inside. That is the segment geometry the extraction already has --
    the model then sees a reconstructed document rather than a description of one.

    Box borders are drawn first and text second, so text never hides structure it sits in.
    """
    if not cells:
        return ""
    xs = [v for c in cells for v in (c["bbox"][0], c["bbox"][2])]
    ys = [v for c in cells for v in (c["bbox"][1], c["bbox"][3])]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    sx = (width - 1) / max(1.0, x1 - x0)
    sy = (height - 1) / max(1.0, y1 - y0)
    canvas = [[" "] * width for _ in range(height)]

    def put(r, c, ch, over=False):
        if 0 <= r < height and 0 <= c < width and (over or canvas[r][c] == " "):
            canvas[r][c] = ch

    for c in cells:
        b = c["bbox"]
        l, t = int((b[0] - x0) * sx), int((b[1] - y0) * sy)
        r, bo = int((b[2] - x0) * sx), int((b[3] - y0) * sy)
        r = max(r, l + 1)
        bo = max(bo, t + 1)
        for cc in range(l, min(r + 1, width)):          # horizontal edges
            put(t, cc, "-", True)
            put(bo, cc, "-", True)
        for rr in range(t, min(bo + 1, height)):        # vertical edges
            put(rr, l, "|", True)
            put(rr, r, "|", True)
        for rr, cc in ((t, l), (t, r), (bo, l), (bo, r)):
            put(rr, cc, "+", True)

    for c in cells:                                      # text inside the boxes
        b = c["bbox"]
        l = int((b[0] - x0) * sx) + 1
        row = int(((b[1] + b[3]) / 2 - y0) * sy)
        row = max(0, min(height - 1, row))
        for i, ch in enumerate(str(c["text_raw"])):
            put(row, l + i, ch)

    return "\n".join("".join(r).rstrip() for r in canvas)


def render_page(page: Dict[str, Any], width: int = 120, height: int = 60,
                min_seg: float = 0.0, with_rules: bool = True) -> str:
    """Redraw the WHOLE PAGE: detected ruling lines from the image, plus the OCR text.

    This is the closest thing to handing the model the drawing itself. `line_segments` are
    the raw rules the image pass found -- the actual printed lines of the sheet, not our
    inference about cells. Rules are drawn first, text over the top, so the model sees a
    document rather than a description of one.

    min_seg drops short segments (hatching, arrowheads, dimension ticks) as a fraction of
    page width; a real table rule is long. With ~20k segments per page, filtering is the
    difference between a readable drawing and a solid block of ink.
    """
    pg = page.get("page") or {}
    W = float(pg.get("width") or 1.0)
    H = float(pg.get("height") or 1.0)
    sx = (width - 1) / max(1.0, W)
    sy = (height - 1) / max(1.0, H)
    canvas = [[" "] * width for _ in range(height)]

    def put(r, c, ch, over=False):
        if 0 <= r < height and 0 <= c < width and (over or canvas[r][c] == " "):
            canvas[r][c] = ch

    if with_rules:
        cut = min_seg * W
        for s in page.get("line_segments", []) or []:
            b = s.get("bbox")
            if not b:
                continue
            x0, y0, x1, y1 = b
            horiz = str(s.get("orientation", "")).startswith("h")
            if (x1 - x0 if horiz else y1 - y0) < cut:
                continue
            if horiz:
                r = int(((y0 + y1) / 2) * sy)
                for c in range(int(x0 * sx), int(x1 * sx) + 1):
                    put(r, c, "-", True)
            else:
                c = int(((x0 + x1) / 2) * sx)
                for r in range(int(y0 * sy), int(y1 * sy) + 1):
                    put(r, c, "|", True)

    for c in _cells(page):
        b = c["bbox"]
        col = int(b[0] * sx)
        row = int(((b[1] + b[3]) / 2) * sy)
        for i, ch in enumerate(str(c["text_raw"])):
            put(row, col + i, ch, True)

    return "\n".join("".join(r).rstrip() for r in canvas)


def holistic(cells: Sequence[Dict], feats: Dict[str, Dict]) -> str:
    """Signals about the SET, never per-cell row/column indices."""
    n = len(cells)
    if not n:
        return ""
    ruled = sum(1 for c in cells if (feats.get(c["poly_id"]) or {}).get("has_ruling_lines"))
    j4 = sum(1 for c in cells if (feats.get(c["poly_id"]) or {}).get("has_4line_junction"))
    ws = [c["bbox"][2] - c["bbox"][0] for c in cells]
    hs = [c["bbox"][3] - c["bbox"][1] for c in cells]
    lines = sum(int((feats.get(c["poly_id"]) or {}).get("n_lines_in_rect") or 0) for c in cells)

    def cv(v):
        m = sum(v) / len(v)
        return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5 / m if m else 0.0

    return (f"regions: {n}\n"
            f"fully ruled boxes: {ruled}/{n}\n"
            f"boxes with 4-line junctions: {j4}/{n}\n"
            f"cell width variation (cv): {cv(ws):.2f}\n"
            f"cell height variation (cv): {cv(hs):.2f}\n"
            f"total text lines across regions: {lines}")
