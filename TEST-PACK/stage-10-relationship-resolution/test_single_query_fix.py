"""
Test the single-query filtering fix for TABLE blocks.
Runs data-cleaning with only "Revision" query active to reproduce the user's scenario.
"""
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

SESSION_CACHE = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "DocumentExtractionConsole" / "sessions" / "amor" / "extraction_cache.json.gz"
)

CURRENT_PATH = Path(__file__).resolve().parent / "bk-main"

sys.path.insert(0, str(CURRENT_PATH))
sys.path.insert(0, str(CURRENT_PATH / "input"))
sys.path.insert(0, str(CURRENT_PATH / "workers"))
sys.path.insert(0, str(CURRENT_PATH / "cleaners"))
sys.path.insert(0, str(CURRENT_PATH / "bootstrap"))

from process_input import PageData
from pipeline import (
    PageExtractionResult,
    process_page_data_cleaning,
    process_page_relationship_resolution,
)
import numpy as np


def main():
    with gzip.open(str(SESSION_CACHE), "rt", encoding="utf-8") as f:
        payload = json.load(f)
    docs = payload.get("documents", [])
    
    print(f"Testing with ONLY 'Revision' query active (user's scenario)")
    print(f"{'='*60}\n")
    
    for doc in docs:
        doc_id = doc.get("document_id", "unknown")
        for p in doc.get("pages", []):
            page_id = p.get("page_id", "unknown")
            if "c25ec847fc8efdf7" not in page_id:  # Focus on the problematic page
                continue
                
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
            
            # ---- Data cleaning with ONLY "Revision" active ----
            cleaned = process_page_data_cleaning(
                synthetic_page,
                page_result,
                active_query_field_names={"Revision"},  # User's scenario
            )
            
            # Check anchors
            if cleaned.anchors:
                by_query = cleaned.anchors.get("by_query", {})
                print(f"Page: {page_id}")
                print(f"  Anchors collected for queries: {list(by_query.keys())}")
                for qn, ads in by_query.items():
                    table_anchors = [a for a in ads if a.get("source_type") == "table_cell"]
                    if table_anchors:
                        print(f"    {qn}: {len(table_anchors)} TABLE anchors")
                        for a in table_anchors[:3]:
                            print(f"      - {a.get('anchor_text')} (score={a.get('score'):.3f})")
            
            # ---- Relationship resolution ----
            resolved = process_page_relationship_resolution(synthetic_page, cleaned)
            
            # Check assembled records
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
                        orientation_case = meta.get("orientation_case")
                        ar_raw = meta.get("assembled_records", [])
                        ar_clean = []
                        for rec in (ar_raw or []):
                            if isinstance(rec, dict):
                                ar_clean.append({k: v for k, v in rec.items() if not k.startswith("__")})
                        
                        print(f"\n  TABLE result for '{qn}':")
                        print(f"    Orientation: {orientation_case}")
                        print(f"    Records: {len(ar_clean)}")
                        if ar_clean:
                            print(f"    First record: {ar_clean[0]}")
                            print(f"    Keys in record: {list(ar_clean[0].keys())}")
                            
                            # Check if we're getting the bug (dict keys as values)
                            first_rec = ar_clean[0]
                            if any(":" in str(v) and ";" in str(v) for v in first_rec.values()):
                                print(f"    ⚠️  BUG DETECTED: Values contain dict-like strings!")
                            else:
                                print(f"    ✓  FIX WORKING: Values are clean")


if __name__ == "__main__":
    main()
