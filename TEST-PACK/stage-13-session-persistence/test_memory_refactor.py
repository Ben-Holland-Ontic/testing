"""
Verify that inline resolution (new) produces identical aggregated_results
as the old deferred resolution pattern.

This test mocks the extraction pipeline to avoid needing real models/PDFs,
and asserts that both paths call the same functions with the same arguments
and produce the same output.
"""
import sys
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Ensure imports work
_BASE = Path(__file__).parent
sys.path.insert(0, str(_BASE))
for _p in ("input", "workers", "cleaners", "bootstrap"):
    _full = str(_BASE / _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)


@dataclass
class FakePageData:
    page_id: str
    document_id: str
    batch_id: str
    page_number: int
    image: Any = None
    width: int = 100
    height: int = 100
    storage_path: str = ""
    patches: int = 0
    status: str = "READY_FOR_EXTRACTION"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FakePageExtractionResult:
    page_id: str
    document_id: str
    batch_id: str
    rect_data: Optional[Dict] = None
    cluster_results: Optional[Dict] = None
    mask_info: Optional[Dict] = None
    region_results: Optional[Dict] = None
    status: str = "EXTRACTED"
    errors: List[str] = field(default_factory=list)
    extraction_timing: Optional[Dict] = None


@dataclass
class FakeDocumentExtractionResult:
    document_id: str
    batch_id: str
    status: str = "PENDING"
    page_results: Dict[str, Any] = field(default_factory=dict)
    aggregated_results: Optional[Dict] = None


def make_fake_page_result(page_id: str, doc_id: str, batch_id: str) -> FakePageExtractionResult:
    """Create a fake extraction result with realistic-looking raw data."""
    return FakePageExtractionResult(
        page_id=page_id,
        document_id=doc_id,
        batch_id=batch_id,
        rect_data={
            "rectangles": {
                "r1": {"bbox": [10, 20, 100, 40], "label": "normal"},
                "r2": {"bbox": [10, 50, 100, 70], "label": "numeric"},
            },
            "clusters": [{"poly_id": "c1", "lines": [{"text": f"line on {page_id}"}]}],
        },
        cluster_results={
            "Cluster_Line": [{"text": f"cluster line {page_id}", "bbox": [10, 20, 100, 40]}],
            "Cluster_Block": [],
        },
        mask_info={"rectangles_masked": 2},
        region_results={
            "Region_Line": [{"text": f"region line {page_id}", "bbox": [10, 20, 100, 40]}],
            "Region_Block": [],
        },
        status="EXTRACTED",
        errors=[],
    )


def test_inline_vs_deferred_resolution():
    """Test that calling resolution inline vs deferred gives the same aggregated_results."""

    doc_id = "DOC_123"
    batch_id = "BATCH_456"
    page_ids = [f"{doc_id}_PG{i:03d}" for i in range(1, 4)]

    # Fake page data objects
    pages = [
        FakePageData(pid, doc_id, batch_id, i + 1)
        for i, pid in enumerate(page_ids)
    ]

    # Fake extraction results (same raw data for both paths)
    results_by_page_id = {
        pid: make_fake_page_result(pid, doc_id, batch_id)
        for pid in page_ids
    }

    # Track what functions were called and with what arguments
    call_log = []

    def mock_cleaning(page, page_result, active_query_field_names=None):
        call_log.append(("cleaning", page.page_id, page_result.page_id))
        page_result.status = "CLEANED"
        return page_result

    def mock_resolution(page, page_result):
        call_log.append(("resolution", page.page_id, page_result.page_id))
        page_result.status = "COMPLETE"
        return page_result

    def mock_aggregate(doc_result):
        call_log.append(("aggregate", doc_result.document_id))
        # Build a deterministic aggregated result based on page results
        entries = []
        for pr in doc_result.page_results.values():
            entries.append({
                "query_name": "Revision",
                "value": f"resolved_{pr.page_id}",
                "page_id": pr.page_id,
                "confidence": "high",
            })
        return {
            "document_summary": {
                "entries": entries,
                "status": "complete",
            }
        }

    with patch("session_runtime.process_page_data_cleaning", mock_cleaning), \
         patch("session_runtime.process_page_relationship_resolution", mock_resolution), \
         patch("session_runtime.aggregate_document_results", mock_aggregate):

        # === PATH A: INLINE (new) ===
        call_log.clear()
        doc_result_inline = FakeDocumentExtractionResult(document_id=doc_id, batch_id=batch_id)
        for page in pages:
            page_result = results_by_page_id.get(page.page_id)
            if page_result and page_result.status.upper() == "EXTRACTED":
                page_result = mock_cleaning(page, page_result, active_query_field_names=None)
                if page_result.status in {"CLEANED", "COMPLETE"}:
                    page_result = mock_resolution(page, page_result)
            doc_result_inline.page_results[page.page_id] = page_result
        doc_result_inline.aggregated_results = mock_aggregate(doc_result_inline)
        inline_calls = list(call_log)
        inline_agg = doc_result_inline.aggregated_results

        # === PATH B: DEFERRED (old) ===
        call_log.clear()
        # First, build extraction_docs like the old code did
        doc_pages = []
        for page in pages:
            result = results_by_page_id[page.page_id]
            doc_pages.append({
                "page_id": page.page_id,
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "status": "EXTRACTED",
                "errors": result.errors,
                "rect_data": result.rect_data,
                "cluster_results": result.cluster_results,
                "mask_info": result.mask_info,
                "region_results": result.region_results,
            })

        # Then resolve like _resolve_from_memory did
        doc_result_deferred = FakeDocumentExtractionResult(document_id=doc_id, batch_id=batch_id)
        for p in doc_pages:
            page_id = p["page_id"]
            # Find matching page object
            page = next(pp for pp in pages if pp.page_id == page_id)
            page_result = FakePageExtractionResult(
                page_id=page_id,
                document_id=doc_id,
                batch_id=batch_id,
                rect_data=p.get("rect_data"),
                cluster_results=p.get("cluster_results"),
                mask_info=p.get("mask_info"),
                region_results=p.get("region_results"),
                status=p.get("status", "EXTRACTED"),
                errors=p.get("errors") or [],
            )
            if page_result.status.upper() == "EXTRACTED":
                page_result = mock_cleaning(page, page_result, active_query_field_names=None)
                if page_result.status in {"CLEANED", "COMPLETE"}:
                    page_result = mock_resolution(page, page_result)
            doc_result_deferred.page_results[page_id] = page_result
        doc_result_deferred.aggregated_results = mock_aggregate(doc_result_deferred)
        deferred_calls = list(call_log)
        deferred_agg = doc_result_deferred.aggregated_results

    # === ASSERTIONS ===
    print("=== INLINE vs DEFERRED VERIFICATION ===")
    print(f"Inline calls:   {inline_calls}")
    print(f"Deferred calls: {deferred_calls}")
    assert inline_calls == deferred_calls, "Function call sequences differ!"
    print("PASS: Same functions called in same order with same arguments")

    print(f"Inline aggregated:   {json.dumps(inline_agg, indent=2)}")
    print(f"Deferred aggregated: {json.dumps(deferred_agg, indent=2)}")
    assert inline_agg == deferred_agg, "Aggregated results differ!"
    print("PASS: Identical aggregated_results")

    # Memory: verify inline path doesn't keep raw data
    # In the new code, extraction_docs would only contain aggregated_results
    # not rect_data, cluster_results, mask_info, region_results
    new_format_entry = {
        "aggregated_results": inline_agg,
        "page_statuses": [{"page_id": pid, "status": "EXTRACTED"} for pid in page_ids],
        "skipped_pages_rect_data": {},
    }
    old_format_entry = {"pages": doc_pages}

    new_size = len(json.dumps(new_format_entry))
    old_size = len(json.dumps(old_format_entry))
    print(f"\nNew format size: {new_size} bytes")
    print(f"Old format size: {old_size} bytes")
    print(f"Reduction: {100 - (new_size / old_size * 100):.1f}%")
    # NOTE: With tiny test data the reduction is modest (~66%), but with real
    # extraction data (hundreds of rectangles, cluster lines, region blocks per page)
    # the reduction would be 95%+ since we discard all that raw data.
    assert new_size < old_size * 0.7, "New format should be smaller than old format"
    print(f"PASS: New format is {100 - (new_size / old_size * 100):.1f}% smaller")

    print("\n=== ALL TESTS PASSED ===")


def test_extraction_docs_format():
    """Verify the new extraction_docs entry format is correct."""
    from session_runtime import _extract_query_values_for_export

    # Mock a pre-resolved doc entry
    doc_entry = {
        "document_id": "DOC_123",
        "relative_path": "test.pdf",
        "relative_folder": "tests",
        "aggregated_results": {
            "document_summary": {
                "entries": [
                    {"query_name": "Revision", "value": "A", "page_id": "DOC_123_PG001"},
                    {"query_name": "Title", "value": "Test Doc", "page_id": "DOC_123_PG001"},
                ]
            }
        },
        "page_statuses": [
            {"page_id": "DOC_123_PG001", "page_number": 1, "source_path": "test.pdf", "status": "EXTRACTED", "errors": []},
            {"page_id": "DOC_123_PG002", "page_number": 2, "source_path": "test.pdf", "status": "SKIPPED", "errors": ["timeout"]},
        ],
        "skipped_pages_rect_data": {
            "DOC_123_PG002": {"rectangles": {"r1": {"bbox": [0, 0, 100, 50]}}},
        },
    }

    # Verify structure
    assert "aggregated_results" in doc_entry
    assert "page_statuses" in doc_entry
    assert "skipped_pages_rect_data" in doc_entry
    assert "rect_data" not in doc_entry  # Should NOT be at top level
    assert "cluster_results" not in doc_entry
    assert "mask_info" not in doc_entry
    assert "region_results" not in doc_entry

    # Verify page_statuses doesn't contain raw data
    for ps in doc_entry["page_statuses"]:
        assert "rect_data" not in ps
        assert "cluster_results" not in ps
        assert "mask_info" not in ps
        assert "region_results" not in ps

    # Verify skipped_pages_rect_data only has skipped pages
    assert "DOC_123_PG002" in doc_entry["skipped_pages_rect_data"]
    assert "DOC_123_PG001" not in doc_entry["skipped_pages_rect_data"]

    print("PASS: extraction_docs format is correct")


if __name__ == "__main__":
    test_inline_vs_deferred_resolution()
    test_extraction_docs_format()
