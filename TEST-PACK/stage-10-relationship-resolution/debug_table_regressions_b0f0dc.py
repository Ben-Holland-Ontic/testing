#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from input.process_input import DocumentData, _extract_image_page  # noqa: E402
from pipeline import (  # noqa: E402
    process_page_data_cleaning,
    process_page_extraction,
    process_page_relationship_resolution,
)

LOG_PATH = ROOT / "debug-b0f0dc.log"
SESSION_ID = "b0f0dc"


def _log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    rec = {
        "sessionId": SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _flatten_results(relationships: Dict[str, Any]) -> List[Any]:
    out: List[Any] = []
    for _q, v in (relationships or {}).items():
        if isinstance(v, list):
            out.extend(v)
        elif v is not None:
            out.append(v)
    return out


def _extract_table_query_snapshot(rr: Any) -> List[Dict[str, Any]]:
    rel = getattr(rr, "relationships", {}) or {}
    rows: List[Dict[str, Any]] = []
    for query, v in rel.items():
        vals = v if isinstance(v, list) else [v]
        for r in vals:
            if r is None:
                continue
            fmt = (getattr(r, "format", "") or "").upper()
            if fmt != "TABLE":
                continue
            meta = getattr(r, "meta", None) or {}
            if not isinstance(meta, dict):
                continue
            rows.append(
                {
                    "query": str(query),
                    "field": str(getattr(r, "field", "") or query),
                    "block_id": str(meta.get("block_id") or ""),
                    "orientation_case": meta.get("orientation_case"),
                    "chosen_orientation": meta.get("chosen_orientation"),
                    "values_n": len(getattr(r, "values", None) or []),
                    "values_sample": list((getattr(r, "values", None) or [])[:8]),
                    "assembled_records_n": len(meta.get("assembled_records") or []),
                    "assembled_sample": list((meta.get("assembled_records") or [])[:4]),
                    "definition": str(getattr(r, "definition", "") or ""),
                }
            )
    return rows


def _table_catalog(rr: Any) -> List[Dict[str, Any]]:
    cons = getattr(rr, "consolidation", {}) or {}
    out: List[Dict[str, Any]] = []
    for t in (cons.get("tables") or []):
        out.append(
            {
                "table_id": str(t.get("table_id") or ""),
                "ec_type": str(t.get("ec_type") or ""),
                "grid_shape": list(t.get("grid_shape") or []),
                "n_cells": len(t.get("cells") or []),
                "reading_path": str(t.get("reading_path") or ""),
            }
        )
    return out


def _capture_t1_non_empty_cells(rr: Any) -> Dict[str, Any]:
    cons = getattr(rr, "consolidation", {}) or {}
    tables = cons.get("tables") or []
    t1 = None
    for t in tables:
        if str(t.get("table_id")) == "T1":
            t1 = t
            break
    if not t1:
        return {"missing_t1": True}
    cells = list(t1.get("cells") or [])
    non_empty = []
    for c in cells:
        txt = str(c.get("text") or "").strip()
        if not txt:
            continue
        non_empty.append(
            {
                "text": txt,
                "row_idx": c.get("row_idx"),
                "col_idx": c.get("col_idx"),
                "cell_id": c.get("cell_id"),
                "poly_id": c.get("poly_id"),
            }
        )
    non_empty.sort(key=lambda x: (x.get("row_idx", 999), x.get("col_idx", 999)))
    return {"missing_t1": False, "non_empty_cells": non_empty}


def _run_image(image_path: Path, run_id: str) -> Any:
    doc = DocumentData(
        document_id="DOC_" + hashlib.md5(str(image_path).encode()).hexdigest()[:12],
        batch_id="DEBUG_B0F0DC",
        source_path=str(image_path),
        file_size=image_path.stat().st_size,
    )
    page = _extract_image_page(doc, str(image_path))[0]
    pr = process_page_extraction(page)
    cl = process_page_data_cleaning(page, pr)
    rr = process_page_relationship_resolution(page, cl)
    _log(
        run_id,
        "H4",
        "tools/debug_table_regressions_b0f0dc.py:_run_image",
        "table_catalog",
        {"image": str(image_path), "tables": _table_catalog(rr)},
    )
    _log(
        run_id,
        "H5",
        "tools/debug_table_regressions_b0f0dc.py:_run_image",
        "table_query_snapshot",
        {"image": str(image_path), "queries": _extract_table_query_snapshot(rr)},
    )
    return rr


def main() -> None:
    run_id = f"run_{int(time.time())}"
    images = [
        Path(r"C:/Users/benja/OneDrive/Desktop/data/Capture (1).PNG"),
        Path(r"C:/Users/benja/OneDrive/Desktop/data/01_original.png"),
        Path(r"C:/Users/benja/OneDrive/Desktop/data/Ball Valve dwg 1.PNG"),
    ]
    _log(
        run_id,
        "H0",
        "tools/debug_table_regressions_b0f0dc.py:main",
        "debug_run_start",
        {"images": [str(p) for p in images]},
    )
    for img in images:
        rr = _run_image(img, run_id)
        if "Capture (1).PNG" in str(img):
            _log(
                run_id,
                "H1_H2_H3",
                "tools/debug_table_regressions_b0f0dc.py:main",
                "capture_t1_non_empty_grid_cells",
                {"image": str(img), **_capture_t1_non_empty_cells(rr)},
            )
    _log(
        run_id,
        "H0",
        "tools/debug_table_regressions_b0f0dc.py:main",
        "debug_run_end",
        {"images_n": len(images)},
    )


if __name__ == "__main__":
    main()
