"""Verify the unmatched-text residual section in the completed document.

DIAGNOSTIC. Builds page results with the real shapes (cleaning inputs + relationships +
classification) and runs the real aggregate_document_results, checking the property that
matters: NO extracted line can vanish silently.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY unmatched_section_probe.py
"""
from __future__ import annotations

import os
import sys
import types
import warnings

warnings.filterwarnings("ignore")

BK = r"C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main"
for p in (BK, os.path.join(BK, "workers"), os.path.join(BK, "input"),
          os.path.join(BK, "cleaners")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PADDLEOCR_USE_GPU", "0")

FAIL = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAIL.append(name)


def main() -> int:
    import pipeline

    # --- real-shaped stand-ins -------------------------------------------------------
    def region_line(i, text, block_id, y):
        o = types.SimpleNamespace()
        o.id, o.text_raw, o.text_norm = f"RL{i}", text, text
        o.bbox, o.block_id = [10.0, float(y), 200.0, float(y + 12)], block_id
        return o

    def cluster_line(i, text, poly_id, y):
        o = types.SimpleNamespace()
        o.id, o.text_raw, o.text_norm = f"CL{i}", text, text
        o.bbox, o.poly_id = [300.0, float(y), 480.0, float(y + 12)], poly_id
        return o

    ci = types.SimpleNamespace(
        region_lines=[
            region_line(1, "USED BY QUERY", "B1", 100),
            region_line(2, "dropped from a classified block", "B1", 120),
            region_line(3, "dropped from an ABSTAIN block", "B2", 140),
            region_line(4, "dropped, block never classified", "B9", 160),
            region_line(5, "   ", "B1", 180),                  # whitespace -> excluded
        ],
        cluster_lines=[
            cluster_line(1, "dense table cell nobody queried", "P1", 200),
            cluster_line(2, "ALSO USED", "P1", 220),
        ],
        region_blocks=[], )

    page = types.SimpleNamespace(
        page_id="p1", status="COMPLETE", errors=None, cleaning_inputs=ci,
        classification={"blocks": {"B1": {"label": "KEY_VALUE"}, "B2": {"label": "ABSTAIN"}},
                        "clusters": {"P1": {"label": "TABLE"}}},
        relationships={
            "PartNumber": {"format": "string", "values": ["X"], "score": 0.9,
                           "meta": {"line_id": "RL1"}},
            "Table": {"format": "table", "values": ["Y"], "score": 0.8,
                      "meta": {"line_ids": ["CL2"]}},
        },
    )
    # A page that resolved nothing at all — the misclassified-page case.
    page2 = types.SimpleNamespace(
        page_id="p2", status="COMPLETE", errors=None,
        cleaning_inputs=types.SimpleNamespace(
            region_lines=[region_line(10, "whole page produced no relationships", "B1", 50)],
            cluster_lines=[], region_blocks=[]),
        classification={"blocks": {}, "clusters": {}},
        relationships=None,
    )

    doc = types.SimpleNamespace(
        document_id="DOC1", batch_id="B", original_document_id="doc.pdf",
        page_results={"p1": page, "p2": page2})

    agg = pipeline.aggregate_document_results(doc)

    # --- assertions -----------------------------------------------------------------
    check("section exists", "unmatched_text" in agg)
    sec = agg.get("unmatched_text", {})
    keys = list(agg.keys())
    check("section is LAST in the artefact (sits at the bottom)",
          keys and keys[-1] == "unmatched_text", f"order={keys}")

    texts = [l["text"] for l in sec.get("lines", [])]
    ids = [l["line_id"] for l in sec.get("lines", [])]

    check("used lines are EXCLUDED", "RL1" not in ids and "CL2" not in ids, f"ids={ids}")
    check("unused region line included", "RL2" in ids)
    check("unused cluster line included (dense cell)", "CL1" in ids)
    check("whitespace-only line excluded", "RL5" not in ids)
    check("page that resolved nothing is covered", "RL10" in ids)

    by_id = {l["line_id"]: l for l in sec.get("lines", [])}
    check("reason: classified region, no query match",
          by_id.get("RL2", {}).get("reason") == "no_query_matched",
          by_id.get("RL2", {}).get("reason"))
    check("reason: ABSTAIN region",
          by_id.get("RL3", {}).get("reason") == "region_abstained",
          by_id.get("RL3", {}).get("reason"))
    check("reason: region never classified",
          by_id.get("RL4", {}).get("reason") == "region_unclassified",
          by_id.get("RL4", {}).get("reason"))
    check("reason: page produced no relationships",
          by_id.get("RL10", {}).get("reason") == "no_relationships_on_page",
          by_id.get("RL10", {}).get("reason"))

    check("every line carries a bbox",
          all(l.get("bbox") and len(l["bbox"]) == 4 for l in sec.get("lines", [])),
          "bbox present on all")
    check("every line carries page_id",
          all(l.get("page_id") for l in sec.get("lines", [])))
    check("counts consistent",
          sec.get("total_lines") == len(sec.get("lines", [])) == 5,
          f"total={sec.get('total_lines')}")
    check("by_reason tallies", sum(sec.get("by_reason", {}).values()) == sec.get("total_lines"),
          str(sec.get("by_reason")))

    # THE property this feature exists for.
    all_extracted = {"RL1", "RL2", "RL3", "RL4", "CL1", "CL2", "RL10"}   # non-blank only
    emitted = set(ids)
    used = pipeline._collect_used_line_ids_from_relationships(page.relationships)
    accounted = emitted | used
    check("NOTHING EXTRACTED IS SILENTLY LOST (every line is either used or emitted)",
          all_extracted <= accounted, f"missing={sorted(all_extracted - accounted)}")

    print()
    print(f"by_reason: {sec.get('by_reason')}")
    print(f"pages_with_unmatched: {sec.get('pages_with_unmatched')}")
    for l in sec.get("lines", []):
        print(f"  p={l['page_id']} {l['line_id']:<5} {l['reason']:<26} "
              f"bbox={l['bbox']} {l['text'][:40]!r}")

    print()
    print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} CHECK(S) FAILED: {FAIL}")
    return len(FAIL)


if __name__ == "__main__":
    raise SystemExit(main())
