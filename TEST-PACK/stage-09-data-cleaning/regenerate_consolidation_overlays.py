#!/usr/bin/env python3
"""Run extraction + data cleaning only, then write consolidated-table overlays and tables JSON.

Uses the same plumbing as full pipeline through ``consolidate_tables`` (no relationship resolution).
Image list defaults to filenames in ``output/unified_diag/post_nlines_raw_fix_v1_summary.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from input.process_input import DocumentData, _extract_image_page  # noqa: E402
from pipeline import (  # noqa: E402
    _draw_table_consolidation_overlay_to_path,
    process_page_data_cleaning,
    process_page_extraction,
)

DEFAULT_SUMMARY = ROOT / "output" / "unified_diag" / "post_nlines_raw_fix_v1_summary.json"
DEFAULT_DATA_DIRS = [
    Path(os.environ.get("BK_DESKTOP_DATA", "")).expanduser() if os.environ.get("BK_DESKTOP_DATA") else None,
    Path(r"C:/Users/benja/OneDrive/Desktop/data"),
    ROOT / "input" / "data",
]


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _sanitize_dir_name(name: str) -> str:
    s = re.sub(r"[<>:\"/\\|?*]", "_", name)
    s = s.replace(" ", "_").strip("._")
    return s or "image"


def _find_image(filename: str, data_dirs: List[Path]) -> Optional[Path]:
    for d in data_dirs:
        if not d or not d.is_dir():
            continue
        p = d / filename
        if p.is_file():
            return p
    return None


def _image_names_from_summary(path: Path) -> List[str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [str(row["image"]) for row in data if isinstance(row, dict) and row.get("image")]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        action="append",
        default=[],
        help="Folder containing source images (repeatable). Default: BK_DESKTOP_DATA, OneDrive/data, input/data",
    )
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="JSON list with 'image' keys")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "unified_diag" / "post_nlines_raw_fix_v1_viz")
    ap.add_argument("--image", action="append", default=[], help="Single image filename or path (repeatable)")
    args = ap.parse_args()

    data_dirs: List[Path] = []
    for raw in args.data_dir:
        data_dirs.append(Path(raw).expanduser())
    for d in DEFAULT_DATA_DIRS:
        if d and d not in data_dirs:
            data_dirs.append(d)

    names: List[str] = []
    if args.image:
        for s in args.image:
            p = Path(s)
            names.append(p.name if p.is_file() else s)
    elif args.summary.is_file():
        names = _image_names_from_summary(args.summary)
    else:
        logging.error("No --image and summary missing: %s", args.summary)
        sys.exit(1)

    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    missing: List[str] = []
    for filename in names:
        img_path = Path(filename)
        if not img_path.is_file():
            found = _find_image(filename, data_dirs)
            img_path = found if found else img_path
        if not img_path.is_file():
            missing.append(filename)
            logging.warning("Skip (not found): %s", filename)
            continue

        sub = out_root / _sanitize_dir_name(img_path.stem)
        sub.mkdir(parents=True, exist_ok=True)

        doc = DocumentData(
            document_id=_sanitize_dir_name(img_path.name),
            batch_id="VIZ_ONLY",
            source_path=str(img_path),
            file_size=img_path.stat().st_size,
        )
        page = _extract_image_page(doc, str(img_path))[0]
        pr = process_page_extraction(page)
        process_page_data_cleaning(page, pr)

        cons = pr.consolidation or {}
        tables_path = sub / "consolidation_tables.json"
        with tables_path.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(cons.get("tables", [])), f, indent=2)

        img = np.asarray(page.image).copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        overlay_path = sub / "overlay_consolidated_tables.jpg"
        _draw_table_consolidation_overlay_to_path(img, cons if isinstance(cons, dict) else None, str(overlay_path))

        logging.info("Wrote %s (%s tables)", overlay_path, len(cons.get("tables", []) or []))

    if missing:
        logging.error("Missing %d image(s). Tried dirs: %s", len(missing), [str(d) for d in data_dirs if d])


if __name__ == "__main__":
    main()
