"""
Comparative table assembly test.
Runs the data-cleaning + relationship-resolution pipeline on a cached extraction
using BOTH the current code and the 1.5.0 release, then prints a side-by-side
comparison of orientation, anchor grid, and assembled_records for every TABLE block.

Usage:
    python compare_table_assembly.py
"""
import gzip
import importlib
import json
import os
import sys
import copy
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

SESSION_CACHE = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "DocumentExtractionConsole" / "sessions" / "amor" / "extraction_cache.json.gz"
)

CURRENT_PATH  = Path(__file__).resolve().parent / "bk-main"
OLD_PATH      = Path(__file__).resolve().parent / "dist" / "release" / "unzip-1.5.0" / "bk-main"


def _load_cache() -> List[Dict[str, Any]]:
    with gzip.open(str(SESSION_CACHE), "rt", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("documents", [])


def _setup_syspath(base: Path) -> None:
    sub_dirs = ["", "input", "workers", "cleaners", "bootstrap"]
    for sd in reversed(sub_dirs):
        p = str(base / sd) if sd else str(base)
        if p not in sys.path:
            sys.path.insert(0, p)


def _teardown_syspath(base: Path) -> None:
    sub_dirs = ["", "input", "workers", "cleaners", "bootstrap"]
    for sd in sub_dirs:
        p = str(base / sd) if sd else str(base)
        try:
            sys.path.remove(p)
        except ValueError:
            pass


def _unload_modules() -> None:
    """Remove cached modules so next import gets fresh copies from current sys.path."""
    to_remove = [k for k in list(sys.modules.keys())
                 if any(k.startswith(p) for p in (
                     "data_cleaningv2", "relationship_resolver_v2_enhanced",
                     "pipeline", "process_input", "anchor_collector",
                     "cleaners.", "workers.", "bootstrap.", "input.",
                 ))]
    for k in to_remove:
        del sys.modules[k]


def run_pipeline_for_version(label: str, base: Path, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run data-cleaning + resolve for every page in docs under `base`.
    Returns a dict keyed by (doc_id, page_id) with table assembly info.
    """
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  Path: {base}")
    print(f"{'='*60}")

    _unload_modules()
    _setup_syspath(base)

    results: Dict[str, Any] = {}
    try:
        from process_input import PageData
        from pipeline import (
            PageExtractionResult,
            process_page_data_cleaning,
            process_page_relationship_resolution,
        )
        import numpy as np

        for doc in docs:
            doc_id = doc.get("document_id", "unknown")
            for p in doc.get("pages", []):
                page_id = p.get("page_id", "unknown")
                page_num = int(p.get("page_number") or 1)
                w = int(p.get("width") or 1)
                h = int(p.get("height") or 1)

                synthetic_page = PageData(
                    page_id=page_id,
                    document_id=doc_id,
                    batch_id=doc.get("batch_id", ""),
                    page_number=page_num,
                    image=np.zeros((h, w, 3), dtype=np.uint8),
                    width=w,
                    height=h,
                    storage_path="",
                    patches=0,
                    status="READY_FOR_EXTRACTION",
                )
                page_result = PageExtractionResult(
                    page_id=page_id,
                    document_id=doc_id,
                    batch_id=doc.get("batch_id", ""),
                    rect_data=p.get("rect_data"),
                    cluster_results=p.get("cluster_results"),
                    mask_info=p.get("mask_info"),
                    region_results=p.get("region_results"),
                    status=p.get("status", "EXTRACTED"),
                    errors=p.get("errors") or [],
                    extraction_timing=p.get("extraction_timing"),
                )
                if page_result.status.upper() != "EXTRACTED":
                    continue

                # ---- Data cleaning ----
                cleaned = process_page_data_cleaning(
                    synthetic_page,
                    page_result,
                    active_query_field_names=None,
                )
                if cleaned.status not in {"CLEANED", "COMPLETE"}:
                    print(f"  [{label}] {page_id}: cleaning status={cleaned.status}")
                    continue

                # ---- Snapshot: anchors by query (TABLE source_type only) ----
                table_anchors: Dict[str, Any] = {}
                if cleaned.anchors:
                    by_query = cleaned.anchors.get("by_query", {})
                    for qn, ads in by_query.items():
                        ta = [a for a in ads if a.get("source_type") == "table_cell"]
                        if ta:
                            table_anchors[qn] = [
                                {
                                    "source_id": a.get("source_id"),
                                    "block_id": a.get("block_id"),
                                    "score": round(float(a.get("score", 0)), 4),
                                    "fuzz_score": round(float(a.get("fuzz_score", 0)), 4),
                                    "cosine_score": round(float(a.get("cosine_score", 0)), 4),
                                    "anchor_text": a.get("anchor_text"),
                                }
                                for a in ta
                            ]

                # ---- Relationship resolution ----
                resolved = process_page_relationship_resolution(synthetic_page, cleaned)

                # ---- Collect assembled_records for TABLE results ----
                assembled: Dict[str, Any] = {}
                if resolved.relationships:
                    for qn, rlist in resolved.relationships.items():
                        items = rlist if isinstance(rlist, list) else ([rlist] if rlist else [])
                        for r in items:
                            if r is None:
                                continue
                            fmt = (getattr(r, "format", "") or "").upper()
                            if fmt != "TABLE":
                                continue
                            meta = getattr(r, "meta", {}) or {}
                            block_id = meta.get("block_id", "?")
                            orientation_case = meta.get("orientation_case")
                            ar_raw = meta.get("assembled_records", [])
                            ar_clean = []
                            for rec in (ar_raw or []):
                                if isinstance(rec, dict):
                                    ar_clean.append({k: v for k, v in rec.items() if not k.startswith("__")})
                            assembled[qn] = {
                                "block_id": block_id,
                                "orientation_case": orientation_case,
                                "n_records": len(ar_clean),
                                "records_preview": ar_clean[:4],
                            }

                key = f"{doc_id}::{page_id}"
                results[key] = {
                    "table_anchors": table_anchors,
                    "assembled": assembled,
                }
                n_table_anchors = sum(len(v) for v in table_anchors.values())
                print(f"  [{label}] {page_id}: {n_table_anchors} TABLE anchors, "
                      f"{len(assembled)} TABLE results assembled")

    except Exception:
        print(f"  [{label}] FAILED:")
        traceback.print_exc()
    finally:
        _teardown_syspath(base)
        _unload_modules()

    return results


def compare_results(current: Dict, old: Dict) -> None:
    print(f"\n{'='*60}")
    print("  COMPARISON (current vs 1.5.0)")
    print(f"{'='*60}")

    all_keys = sorted(set(current) | set(old))
    for key in all_keys:
        print(f"\n--- Page: {key} ---")
        cur = current.get(key, {})
        prv = old.get(key, {})

        # Compare table anchors per query
        all_qnames = sorted(set(cur.get("table_anchors", {})) | set(prv.get("table_anchors", {})))
        if all_qnames:
            print("  TABLE ANCHORS:")
        for qn in all_qnames:
            ca = cur.get("table_anchors", {}).get(qn, [])
            pa = prv.get("table_anchors", {}).get(qn, [])
            same = ca == pa
            marker = "  " if same else "!!"
            cur_scores = [(a["anchor_text"], a["score"], a["fuzz_score"]) for a in ca]
            prv_scores = [(a["anchor_text"], a["score"], a["fuzz_score"]) for a in pa]
            print(f"  {marker} Query '{qn}':")
            print(f"       current : {cur_scores}")
            print(f"       1.5.0   : {prv_scores}")

        # Compare assembled records per query
        all_aqn = sorted(set(cur.get("assembled", {})) | set(prv.get("assembled", {})))
        if all_aqn:
            print("  ASSEMBLED RECORDS:")
        for qn in all_aqn:
            ca = cur.get("assembled", {}).get(qn, {})
            pa = prv.get("assembled", {}).get(qn, {})
            same = ca == pa
            marker = "  " if same else "!!"
            print(f"  {marker} Query '{qn}':")
            print(f"       current : orientation={ca.get('orientation_case')}, "
                  f"n_records={ca.get('n_records')}")
            print(f"                 preview={ca.get('records_preview', [])[:2]}")
            print(f"       1.5.0   : orientation={pa.get('orientation_case')}, "
                  f"n_records={pa.get('n_records')}")
            print(f"                 preview={pa.get('records_preview', [])[:2]}")


def main() -> None:
    if not SESSION_CACHE.exists():
        print(f"ERROR: extraction cache not found at:\n  {SESSION_CACHE}")
        sys.exit(1)

    print(f"Loading extraction cache: {SESSION_CACHE}")
    docs = _load_cache()
    print(f"  {len(docs)} document(s), "
          f"{sum(len(d.get('pages',[])) for d in docs)} page(s) total")

    current_results = run_pipeline_for_version("CURRENT (all queries)", CURRENT_PATH, docs)
    old_results     = run_pipeline_for_version("1.5.0 (all queries)",   OLD_PATH,     docs)

    compare_results(current_results, old_results)

    out = {
        "current": current_results,
        "old_1_5_0": old_results,
    }
    out_path = Path(__file__).parent / "compare_table_assembly_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
