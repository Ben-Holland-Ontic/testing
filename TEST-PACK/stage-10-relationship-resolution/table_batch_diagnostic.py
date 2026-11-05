#!/usr/bin/env python3
"""Batch table diagnostics for consolidation/reading regression tracking.

Runs extraction -> cleaning -> relationship resolution on a batch of images and
records:
  - consolidated tables (ec_type, reading_path, grid shape)
  - matrix payload presence/counts
  - debug counters from existing logs:
      * H_fastpath_guard_miss (duplicate/bounds)
      * axis_fallback_activated / axis_fallback_skipped_unresolved_anchor_lookup

Outputs under ``output/table_batch_diag/<batch_name>/``:
  - summary.json
  - per_image.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import hashlib
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from input.process_input import DocumentData, _extract_image_page  # noqa: E402
from pipeline import (  # noqa: E402
    process_page_extraction,
    process_page_data_cleaning,
    process_page_relationship_resolution,
)


VALID_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _iter_images(root_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(root_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            out.append(p)
    return out


def _read_new_log_events(path: Path, start_offset: int) -> Tuple[List[Dict[str, Any]], int]:
    if not path.exists():
        return [], start_offset
    with path.open("r", encoding="utf-8") as fh:
        fh.seek(start_offset)
        chunk = fh.read()
        end = fh.tell()
    events: List[Dict[str, Any]] = []
    for ln in chunk.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            events.append(json.loads(ln))
        except Exception:
            continue
    return events, end


def _extract_payload_metrics(rr: Any) -> Dict[str, Any]:
    rel = getattr(rr, "relationships", {}) or {}
    payloads = []
    for query, v in rel.items():
        vals = v if isinstance(v, list) else [v]
        for r in vals:
            meta = getattr(r, "meta", None) or (r.get("meta", {}) if isinstance(r, dict) else {})
            ar = meta.get("assembled_records") if isinstance(meta, dict) else None
            if isinstance(ar, list) and ar and isinstance(ar[0], dict):
                mr = ar[0].get("matrix_records")
                if isinstance(mr, dict):
                    payloads.append(
                        {
                            "query": query,
                            "orientation": mr.get("orientation"),
                            "orientation_source": mr.get("orientation_source"),
                            "authority_source": mr.get("authority_source"),
                            "main_headers_n": len(mr.get("main_headers") or []),
                            "sub_headers_by_main_n": len(mr.get("sub_headers_by_main") or []),
                            "row_records_n": len(mr.get("row_records") or []),
                            "header_groups_n": len(mr.get("header_groups") or []),
                            "row_axis_fallback_fired": bool(mr.get("row_axis_fallback_fired")),
                        }
                    )
    return {
        "payload_count": len(payloads),
        "payloads": payloads,
    }


def _table_bbox(table: Dict[str, Any]) -> Tuple[float, float, float, float]:
    bb = table.get("bbox") or (0, 0, 0, 0)
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        return (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
    return (0.0, 0.0, 0.0, 0.0)


def _bbox_contains(bbox: Tuple[float, float, float, float], x: float, y: float) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _extract_query_reads_by_table(rr: Any) -> Dict[str, List[Dict[str, Any]]]:
    by_table: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rel = getattr(rr, "relationships", {}) or {}
    for query, v in rel.items():
        vals = v if isinstance(v, list) else [v]
        for r in vals:
            if r is None:
                continue
            field = getattr(r, "field", None)
            definition = getattr(r, "definition", None)
            values = getattr(r, "values", None)
            score = getattr(r, "score", None)
            meta = getattr(r, "meta", None) or {}
            if not isinstance(meta, dict):
                meta = {}
            block_id = str(meta.get("block_id") or "")
            if not block_id:
                continue
            by_table[block_id].append(
                {
                    "query": str(query),
                    "field": str(field) if field is not None else str(query),
                    "definition": str(definition or ""),
                    "values": [str(x) for x in (values or [])],
                    "score": float(score) if score is not None else 0.0,
                    "strategy": str(meta.get("strategy") or ""),
                    "orientation_case": meta.get("orientation_case"),
                    "chosen_orientation": meta.get("chosen_orientation"),
                    "assembled_records_n": len(meta.get("assembled_records") or []),
                }
            )
    return by_table


def _extract_survival_reasons(cl: Any, table_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cons = (getattr(cl, "consolidation", None) or {})
    tables = cons.get("tables") or []
    table_by_id = {str(t.get("table_id") or ""): t for t in tables}
    anchors_dict = getattr(cl, "anchors", None) or {}
    anchors_all = []
    if isinstance(anchors_dict, dict):
        for _, arr in anchors_dict.items():
            if isinstance(arr, list):
                anchors_all.extend(arr)

    out: Dict[str, Dict[str, Any]] = {}
    for tm in table_metrics:
        tid = str(tm.get("table_id") or "")
        t = table_by_id.get(tid) or {}
        t_bbox = _table_bbox(t)
        cells = t.get("cells") or []
        texts = [str((c.get("text") or "")).strip() for c in cells]
        texts_non_empty = [x for x in texts if x]
        max_len = max((len(x) for x in texts_non_empty), default=0)
        avg_len = (sum(len(x) for x in texts_non_empty) / max(1, len(texts_non_empty)))
        dense_cells = sum(1 for x in texts_non_empty if len(x) >= 40)
        multi_token_cells = sum(1 for x in texts_non_empty if len(x.split()) >= 6)
        anchor_in_table = 0
        for a in anchors_all:
            bb = getattr(a, "bbox", None)
            if not bb or len(bb) != 4:
                continue
            cx = (float(bb[0]) + float(bb[2])) * 0.5
            cy = (float(bb[1]) + float(bb[3])) * 0.5
            if _bbox_contains(t_bbox, cx, cy):
                anchor_in_table += 1

        flags = {
            "single_column_stack": bool((tm.get("grid_shape") or [0, 0])[1] == 1 and (tm.get("grid_shape") or [0, 0])[0] >= 3),
            "high_content_density": bool(dense_cells >= 1 or avg_len >= 18.0),
            "multi_anchor_inside_table": bool(anchor_in_table >= 2),
            "many_multi_token_cells": bool(multi_token_cells >= 2),
        }
        out[tid] = {
            "table_bbox": list(t_bbox),
            "anchor_count_inside_table": int(anchor_in_table),
            "cell_text_stats": {
                "n_cells": int(len(cells)),
                "n_non_empty": int(len(texts_non_empty)),
                "max_len": int(max_len),
                "avg_len": float(round(avg_len, 3)),
                "dense_cells_ge_40chars": int(dense_cells),
                "multi_token_cells_ge_6tokens": int(multi_token_cells),
            },
            "suspect_flags": flags,
        }
    return out


def _extract_table_metrics(rr: Any) -> List[Dict[str, Any]]:
    cons = (getattr(rr, "consolidation", None) or {})
    tables = cons.get("tables") or []
    out = []
    for t in tables:
        mr = t.get("matrix_records") if isinstance(t.get("matrix_records"), dict) else {}
        out.append(
            {
                "table_id": t.get("table_id"),
                "ec_type": t.get("ec_type"),
                "reading_path": t.get("reading_path"),
                "matrix_orientation": t.get("matrix_orientation"),
                "grid_shape": t.get("grid_shape"),
                "n_cells": len(t.get("cells") or []),
                "main_headers_n": len(mr.get("main_headers") or []),
                "sub_headers_n": len(mr.get("sub_headers") or []),
                "header_groups_n": len(mr.get("header_groups") or []),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default=r"C:\Users\benja\OneDrive\Desktop\data",
        help="Directory containing test images.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional max images.")
    ap.add_argument("--batch-name", default="default_batch")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Missing data dir: {data_dir}")

    images = _iter_images(data_dir)
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {data_dir}")

    # Use existing debug logs; do not add instrumentation to core modules.
    os.environ["PIPELINE_DEBUG_LOGS"] = "1"
    debug_fast = ROOT / "debug-b1a05e.log"
    debug_axis = ROOT / "debug-363307.log"
    debug_matrix = Path(tempfile.gettempdir()) / "OculiPipeline" / "logs" / "debug-b0f0dc.ndjson"
    for p in (debug_fast, debug_axis):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    fast_off = 0
    axis_off = 0
    matrix_off = 0
    per_image: List[Dict[str, Any]] = []
    totals = {
        "fast_guard_failures": Counter(),
        "axis_fallback": Counter(),
        "ec_type": Counter(),
    }

    for img in images:
        doc = DocumentData(
            document_id="DOC_" + hashlib.md5(str(img).encode()).hexdigest()[:12],
            batch_id="TABLE_BATCH_DIAG",
            source_path=str(img),
            file_size=img.stat().st_size,
        )
        page = _extract_image_page(doc, str(img))[0]
        pr = process_page_extraction(page)
        cl = process_page_data_cleaning(page, pr)
        rr = process_page_relationship_resolution(page, cl)

        fast_events, fast_off = _read_new_log_events(debug_fast, fast_off)
        axis_events, axis_off = _read_new_log_events(debug_axis, axis_off)
        matrix_events, matrix_off = _read_new_log_events(debug_matrix, matrix_off)

        fast_counts = Counter()
        for ev in fast_events:
            if ev.get("hypothesisId") == "H_fastpath_guard_miss":
                reason = ((ev.get("data") or {}).get("failure") or "unknown")
                fast_counts[str(reason)] += 1
                totals["fast_guard_failures"][str(reason)] += 1

        axis_counts = Counter()
        for ev in axis_events:
            if ev.get("location") == "assemble_table_records":
                msg = str(ev.get("message") or "")
                if msg in (
                    "axis_fallback_activated",
                    "axis_fallback_skipped_unresolved_anchor_lookup",
                ):
                    axis_counts[msg] += 1
                    totals["axis_fallback"][msg] += 1

        matrix_evidence = {
            "use_col_header_votes": [],
            "t_signal_undercount": [],
            "hierarchy_strict": [],
        }
        for ev in matrix_events:
            hid = str(ev.get("hypothesisId") or "")
            data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
            if hid == "H_USE_COL_HEADER_VOTE":
                matrix_evidence["use_col_header_votes"].append(data)
            elif hid == "H_T_SIGNAL_UNDERCOUNT":
                matrix_evidence["t_signal_undercount"].append(data)
            elif hid == "H_HIERARCHY_FORMULA_STRICT":
                matrix_evidence["hierarchy_strict"].append(data)

        table_metrics = _extract_table_metrics(rr)
        by_table_query_reads = _extract_query_reads_by_table(rr)
        survival_reasons = _extract_survival_reasons(cl, table_metrics)
        for t in table_metrics:
            totals["ec_type"][str(t.get("ec_type") or "UNKNOWN")] += 1
            tid = str(t.get("table_id") or "")
            t["query_reads"] = by_table_query_reads.get(tid, [])
            t["survival_reasons"] = survival_reasons.get(tid, {})
        payload_metrics = _extract_payload_metrics(rr)

        per_image.append(
            {
                "image": str(img),
                "tables": table_metrics,
                "resolver_matrix_payload": payload_metrics,
                "matrix_evidence": matrix_evidence,
                "debug_fastpath_guard": dict(fast_counts),
                "debug_axis_fallback": dict(axis_counts),
            }
        )

    out_dir = ROOT / "output" / "table_batch_diag" / args.batch_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_image.json").write_text(json.dumps(per_image, indent=2), encoding="utf-8")

    summary = {
        "data_dir": str(data_dir),
        "n_images": len(images),
        "images": [str(p) for p in images],
        "totals": {
            "fast_guard_failures": dict(totals["fast_guard_failures"]),
            "axis_fallback": dict(totals["axis_fallback"]),
            "ec_type": dict(totals["ec_type"]),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("OUT:", out_dir)
    print("images:", len(images))
    print("fast_guard_failures:", dict(totals["fast_guard_failures"]))
    print("axis_fallback:", dict(totals["axis_fallback"]))
    print("ec_type:", dict(totals["ec_type"]))


if __name__ == "__main__":
    main()

