"""Diagnostic: show assembled_records row structure for TABLE results in latest batch output."""
import json, os

BATCH = "BATCH_87163082"
OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs", BATCH)

for doc_dir in sorted(os.listdir(OUTPUTS)):
    er_path = os.path.join(OUTPUTS, doc_dir, "extraction_results.json")
    if not os.path.isfile(er_path):
        continue
    d = json.load(open(er_path))
    for page_id, pdata in d.get("pages", {}).items():
        tables = [r for r in pdata.get("results", []) if r.get("data_format") == "TABLE"]
        if not tables:
            continue
        print("=" * 70)
        print("DOC:", doc_dir)
        for t in tables:
            qnames = t.get("query_names") or [t.get("query_name")]
            orient = t.get("orientation")
            val = t.get("value", [])
            print(f"  query_names={qnames}  orientation={orient}  rows={len(val)}")
            for row in val[:6]:
                rec = row.get("record", {})
                print(f"    row {row.get('row_index')}: {json.dumps(rec)[:300]}")
            if len(val) > 6:
                print(f"    ... +{len(val)-6} more rows")
