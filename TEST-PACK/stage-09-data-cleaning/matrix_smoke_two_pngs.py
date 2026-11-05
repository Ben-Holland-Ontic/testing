"""Smoke test for matrix examples: extraction -> consolidation -> relationship readout."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

logging.basicConfig(level=logging.WARNING)

from input.process_input import DocumentData, _extract_image_page
from pipeline import (
    process_page_data_cleaning,
    process_page_extraction,
    process_page_relationship_resolution,
)

DEFAULT_PATHS = [
    r"C:\Users\benja\OneDrive\Desktop\data\Screenshot 2026-04-08 113432.png",
    r"C:\Users\benja\OneDrive\Desktop\data\Screenshot 2026-04-08 113409.png",
]


def _extract_table_results(relationships: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for query_name, rel_or_list in (relationships or {}).items():
        rows = rel_or_list if isinstance(rel_or_list, list) else [rel_or_list]
        for rel in rows:
            if rel is None:
                continue
            fmt = getattr(rel, "format", None)
            if fmt is None and isinstance(rel, dict):
                fmt = rel.get("format")
            if str(fmt or "").upper() != "TABLE":
                continue
            meta = getattr(rel, "meta", None)
            if meta is None and isinstance(rel, dict):
                meta = rel.get("meta")
            meta = meta or {}
            out.append(
                {
                    "query_name": str(query_name),
                    "field": str(getattr(rel, "field", None) or (rel.get("field") if isinstance(rel, dict) else "") or query_name),
                    "block_id": str(meta.get("block_id") or ""),
                    "orientation_case": meta.get("orientation_case"),
                    "chosen_orientation": meta.get("chosen_orientation"),
                    "assembled_records_n": len(meta.get("assembled_records") or []),
                    "assembled_sample": list((meta.get("assembled_records") or [])[:3]),
                }
            )
    return out


def main() -> None:
    paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PATHS
    batch_id = "MATRIX_SMOKE_BATCH"
    out_dir = Path(ROOT) / "output" / "unified_diag" / "matrix_smoke_two_pngs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        print("\n" + "=" * 72)
        print("FILE:", path)
        if not os.path.isfile(path):
            print("MISSING FILE")
            continue
        doc_id = "DOC_" + hashlib.md5(path.encode("utf-8")).hexdigest()[:12]
        doc = DocumentData(
            document_id=doc_id,
            batch_id=batch_id,
            source_path=path,
            file_size=os.path.getsize(path),
        )
        pages = _extract_image_page(doc, path)
        page = pages[0]
        pr = process_page_extraction(page)
        st = getattr(pr, "status", None)
        print("extraction_status:", st)
        if st == "SKIPPED" or getattr(pr, "errors", None):
            print("errors:", getattr(pr, "errors", None))
            continue
        cleaned = process_page_data_cleaning(page, pr)
        resolved = process_page_relationship_resolution(page, cleaned)
        cons = cleaned.consolidation or {}
        tables = cons.get("tables", []) or []
        table_results = _extract_table_results(getattr(resolved, "relationships", {}) or {})

        n_table_clusters = sum(
            1
            for c in (cleaned.classification or {}).get("clusters", {}).values()
            if isinstance(c, dict) and c.get("label") == "TABLE"
        )
        print("TABLE-labeled clusters:", n_table_clusters)
        print("consolidated tables:", len(tables))
        ec2 = [t for t in tables if str(t.get("ec_type", "")).upper() == "EC2"]
        print("EC2 tables:", len(ec2))
        for t in tables:
            ec = t.get("ec_type", "STANDARD")
            rp = t.get("reading_path", "")
            gs = t.get("grid_shape", [])
            nc = len(t.get("cells", []) or [])
            print(
                f"  {t.get('table_id')} ec_type={ec} reading_path={rp} "
                f"grid={gs} n_cells={nc}"
            )
            mr = t.get("matrix_records")
            if isinstance(mr, dict) and mr:
                hg = mr.get("header_groups") or []
                print(
                    f"    matrix_orientation={mr.get('orientation')} "
                    f"header_groups={len(hg)} "
                    f"synth_t={len(mr.get('synth_t_cells') or [])}"
                )
        print("table query results:", len(table_results))
        for tr in table_results[:12]:
            print(
                "  "
                f"{tr['query_name']} block={tr['block_id']} "
                f"assembled={tr['assembled_records_n']} "
                f"orient={tr.get('chosen_orientation') or tr.get('orientation_case')}"
            )

        stem = Path(path).stem.replace(" ", "_")
        payload = {
            "image": path,
            "tables": [
                {
                    "table_id": t.get("table_id"),
                    "ec_type": t.get("ec_type"),
                    "reading_path": t.get("reading_path"),
                    "grid_shape": t.get("grid_shape"),
                    "n_cells": len(t.get("cells") or []),
                    "matrix_orientation": (t.get("matrix_records") or {}).get("orientation")
                    if isinstance(t.get("matrix_records"), dict)
                    else None,
                    "header_groups_n": len(((t.get("matrix_records") or {}).get("header_groups") or []))
                    if isinstance(t.get("matrix_records"), dict)
                    else 0,
                }
                for t in tables
            ],
            "table_results": table_results,
        }
        out_path = out_dir / f"{stem}_review.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("wrote:", out_path)


if __name__ == "__main__":
    main()
