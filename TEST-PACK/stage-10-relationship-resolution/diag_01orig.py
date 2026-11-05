"""Show full assembled_records for 01_original TABLE results."""
import json, os, sys

BATCH = "BATCH_87163082"
OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs", BATCH)
DOC = "01_original"

found = False
for doc_dir in sorted(os.listdir(OUTPUTS)):
    if DOC not in doc_dir:
        continue
    found = True
    er_path = os.path.join(OUTPUTS, doc_dir, "extraction_results.json")
    if not os.path.isfile(er_path):
        print(f"No extraction_results.json in {doc_dir}")
        continue
    d = json.load(open(er_path))
    print("=" * 70)
    print("DOC:", doc_dir)
    for page_id, pdata in d.get("pages", {}).items():
        all_results = pdata.get("results", [])
        tables = [r for r in all_results if r.get("data_format") == "TABLE"]
        print(f"  Total results: {len(all_results)}, TABLE results: {len(tables)}")
        for t in tables:
            qnames = t.get("query_names") or [t.get("query_name")]
            orient = t.get("orientation")
            val = t.get("value", [])
            print(f"  query_names={qnames}")
            print(f"  orientation={orient}  rows={len(val)}")
            for row in val:
                rec = row.get("record", {})
                print(f"    row {row.get('row_index')}:")
                for k, v in rec.items():
                    marker = " <<<" if k in ("Revision", "Sheet Number") else ""
                    print(f"      {k!r}: {str(v)[:80]!r}{marker}")
            print()
        # Also show non-table Revision/Sheet results
        for r in all_results:
            qn = r.get("query_name", "")
            if qn in ("Revision", "Sheet Number") and r.get("data_format") != "TABLE":
                print(f"  [{r.get('data_format')}] {qn}: {str(r.get('value',''))[:100]!r}")

if not found:
    print(f"No doc matching '{DOC}' in {OUTPUTS}")
    for d in sorted(os.listdir(OUTPUTS)):
        print(f"  {d}")
