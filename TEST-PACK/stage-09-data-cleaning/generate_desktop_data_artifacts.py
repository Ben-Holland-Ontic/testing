#!/usr/bin/env python3
"""Generate consolidated-cell visualizations + extraction_results.json per image.

Single pass per image:
  extraction -> cleaning -> relationship resolution

Outputs under --out/<image_stem>/:
  - overlay_consolidated_cells.png
  - overlay_classification_cells.png
  - consolidation_tables.json
  - extraction_results.json
  - assembled_records.json (table payloads for quick review)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from input.process_input import DocumentData, _extract_image_page  # noqa: E402
from pipeline import (  # noqa: E402
    DocumentExtractionResult,
    _draw_classification_overlay_to_path,
    _draw_table_consolidation_overlay_to_path,
    aggregate_document_results,
    augment_aggregated_with_validation,
    process_page_data_cleaning,
    process_page_extraction,
    process_page_relationship_resolution,
)


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


def _sanitize(name: str) -> str:
    s = re.sub(r"[<>:\"/\\|?*]", "_", name)
    s = s.replace(" ", "_").strip("._")
    return s or "image"


def _image_paths(data_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(data_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        out.append(p)
    return out


def _write_extraction_results(out_dir: Path, page, page_result) -> None:
    doc_result = DocumentExtractionResult(
        document_id=page.document_id,
        batch_id=page.batch_id,
    )
    doc_result.page_results[page.page_id] = page_result
    doc_result.aggregated_results = aggregate_document_results(doc_result)
    payload = dict(doc_result.aggregated_results)
    augment_aggregated_with_validation(payload, page.document_id, {}, doc_result=doc_result)
    with (out_dir / "extraction_results.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)


def run_one(image_path: Path, out_root: Path) -> Dict[str, Any]:
    sub = out_root / _sanitize(image_path.stem)
    sub.mkdir(parents=True, exist_ok=True)

    doc = DocumentData(
        document_id=_sanitize(image_path.name),
        batch_id="DESKTOP_DATA_SINGLEPASS",
        source_path=str(image_path),
        file_size=image_path.stat().st_size,
    )
    page = _extract_image_page(doc, str(image_path))[0]

    page_result = process_page_extraction(page)
    page_result = process_page_data_cleaning(page, page_result)
    page_result = process_page_relationship_resolution(page, page_result)

    consolidation = page_result.consolidation or {}
    classification = page_result.classification or {}

    img = np.asarray(page.image).copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    _draw_table_consolidation_overlay_to_path(
        img.copy(),
        consolidation if isinstance(consolidation, dict) else None,
        str(sub / "overlay_consolidated_cells.png"),
    )
    _draw_classification_overlay_to_path(
        img.copy(),
        page_result.cleaning_inputs,
        classification,
        consolidation,
        str(sub / "overlay_classification_cells.png"),
    )

    with (sub / "consolidation_tables.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(consolidation.get("tables", [])), f, indent=2, ensure_ascii=False)

    assembled_records: Dict[str, Any] = {}
    for query_name, result in (page_result.relationships or {}).items():
        rows = result if isinstance(result, list) else [result]
        for row in rows:
            if not row:
                continue
            meta = row.get("meta", {}) if isinstance(row, dict) else {}
            if isinstance(meta, dict) and "assembled_records" in meta:
                assembled_records[query_name] = meta.get("assembled_records") or []
    with (sub / "assembled_records.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(assembled_records), f, indent=2, ensure_ascii=False)

    _write_extraction_results(sub, page, page_result)

    ec_types = [str(t.get("ec_type") or "STANDARD") for t in (consolidation.get("tables") or [])]
    return {
        "image": str(image_path),
        "n_tables": len(consolidation.get("tables") or []),
        "ec_types": ec_types,
        "out_dir": str(sub),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="Input image directory")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "unified_diag" / "desktop_data_singlepass",
        help="Output root directory",
    )
    args = ap.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    out_root = args.out.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    images = _image_paths(data_dir)
    if not images:
        raise SystemExit(f"No PNG/JPG images found in: {data_dir}")

    items: List[Dict[str, Any]] = []
    for p in images:
        print(f"[RUN] {p.name}")
        items.append(run_one(p, out_root))

    summary = {"images": len(images), "items": items}
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[DONE] summary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()

