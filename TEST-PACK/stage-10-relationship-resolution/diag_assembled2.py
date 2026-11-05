"""Show assembled_records for 01_original and 1520 in latest batch."""
import json, os

BATCH = "BATCH_87163082"
OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs", BATCH)
TARGET_DOCS = ("01_original", "1520217031974")

for doc_dir in sorted(os.listdir(OUTPUTS)):
    if not any(t in doc_dir for t in TARGET_DOCS):
        continue
    er_path = os.path.join(OUTPUTS, doc_dir, "extraction_results.json")
    if not os.path.isfile(er_path):
        continue
    d = json.load(open(er_path))
    print("=" * 70)
    print("DOC:", doc_dir)
    for page_id, pdata in d.get("pages", {}).items():
        tables = [r for r in pdata.get("results", []) if r.get("data_format") == "TABLE"]
        if not tables:
            print("  (no TABLE results)")
            continue
        for t in tables:
            qnames = t.get("query_names") or [t.get("query_name")]
            orient = t.get("orientation")
            val = t.get("value", [])
            print(f"  query_names={qnames}")
            print(f"  orientation={orient}  rows={len(val)}")
            for row in val[:5]:
                rec = row.get("record", {})
                # Show all keys and values
                print(f"    row {row.get('row_index')}:")
                for k, v in rec.items():
                    print(f"      {k!r}: {str(v)[:60]!r}")
            if len(val) > 5:
                print(f"    ... +{len(val)-5} more rows")
            print()
