"""Diagnostic: table cell structure + resolved query values, focused on Revision/Sheet regression."""
import json, os
from collections import Counter

SESSION_DIR = os.path.join(os.environ.get("LOCALAPPDATA", ""),
    "DocumentExtractionConsole", "sessions", "amor")
PREV_BATCHES = os.path.join(os.path.dirname(__file__), "outputs")
BATCH_ID = "BATCH_02514834"

TARGET_FIELDS = {"Revision", "Sheet", "Drawing Number", "Item Number", "Part Number"}


def cell_keys(cells):
    """Show all keys present in the first cell."""
    if cells:
        return list(cells[0].keys())
    return []


def show_grid(cells, label=""):
    """Print TABLE cells sorted by row_idx/col_idx (internal) or row/col (export)."""
    if not cells:
        return
    # Detect which key names are used
    sample = cells[0]
    rkey = "row_idx" if "row_idx" in sample else ("row" if "row" in sample else None)
    ckey = "col_idx" if "col_idx" in sample else ("col" if "col" in sample else None)
    print(f"  Cell keys: {list(sample.keys())[:10]}")
    print(f"  Using row_key={rkey!r}, col_key={ckey!r}")
    def sort_key(c):
        r = c.get(rkey, 0) if rkey else 0
        cc = c.get(ckey, 0) if ckey else 0
        return (r or 0, cc or 0)
    for c in sorted(cells, key=sort_key)[:35]:
        r = c.get(rkey) if rkey else "?"
        col = c.get(ckey) if ckey else "?"
        print(f"    [{r},{col}] {repr(c.get('text',''))[:55]}")
    if len(cells) > 35:
        print(f"    ... +{len(cells)-35}")


# ── 1. Resolved query values ───────────────────────────────────────────────
print("="*65)
print("RESOLVED QUERY VALUES (current session)")
print("="*65)
rp = os.path.join(SESSION_DIR, "resolved_results.json")
if os.path.isfile(rp):
    with open(rp) as f:
        rd = json.load(f)
    for row in rd.get("export_rows", []):
        if BATCH_ID not in row.get("document_id", ""):
            continue
        did = row["document_id"]
        qv = row.get("query_values", {})
        print(f"\nDoc: {did}")
        if isinstance(qv, dict):
            for k, v in qv.items():
                val_str = str(v.get("value", "") if isinstance(v, dict) else v)
                flag = " <<<<" if k in TARGET_FIELDS else ""
                print(f"  {k}: {val_str[:120]}{flag}")
else:
    print(f"NOT FOUND: {rp}")

# ── 2. TABLE cell grids in export_digitised (current session) ─────────────
print()
print("="*65)
print("TABLE CELLS in export_digitised (current session)")
print("="*65)
dd = os.path.join(SESSION_DIR, "digitised")
for fn in sorted(os.listdir(dd)):
    if BATCH_ID not in fn or "export_digitised" not in fn:
        continue
    with open(os.path.join(dd, fn)) as f:
        d = json.load(f)
    for pid, pdata in d.get("pages", {}).items():
        results = pdata.get("results", [])
        tables = [r for r in results if r.get("data_format") == "TABLE"]
        print(f"\n{fn}  ({len(tables)} tables)")
        for i, r in enumerate(tables):
            val = r.get("value", {})
            if not isinstance(val, dict):
                continue
            cells = val.get("cells", [])
            print(f"  Table {i}: grid={val.get('grid_shape')} cells={len(cells)}")
            show_grid(cells)

# ── 3. TABLE cell grids in digitised_document.json (consolidation output) ─
print()
print("="*65)
print("TABLE CELLS in digitised_document.json (consolidation, current)")
print("="*65)
for fn in sorted(os.listdir(dd)):
    if BATCH_ID not in fn or "digitised_document" not in fn:
        continue
    with open(os.path.join(dd, fn)) as f:
        d = json.load(f)
    for pid, pdata in d.get("pages", {}).items():
        results = pdata.get("results", [])
        tables = [r for r in results if r.get("data_format") == "TABLE"]
        print(f"\n{fn}  ({len(tables)} tables)")
        for i, r in enumerate(tables):
            val = r.get("value", {})
            if not isinstance(val, dict):
                continue
            cells = val.get("cells", [])
            print(f"  Table {i}: grid={val.get('grid_shape')} cells={len(cells)}")
            show_grid(cells)

# ── 4. Previous batch comparison (BATCH_10510050) ─────────────────────────
print()
print("="*65)
PREV_BATCH_ID = "BATCH_10510050"
bp = os.path.join(PREV_BATCHES, PREV_BATCH_ID)
print(f"PREVIOUS BATCH {PREV_BATCH_ID} - resolved values")
print("="*65)
if os.path.isdir(bp):
    for doc_dir in sorted(os.listdir(bp)):
        digi_fp = os.path.join(bp, doc_dir, "digitised_document.json")
        if not os.path.isfile(digi_fp):
            continue
        with open(digi_fp) as f:
            d = json.load(f)
        pages = d.get("pages", {})
        for pid, pdata in (pages.items() if isinstance(pages, dict) else []):
            results = pdata.get("results", [])
            tables = [r for r in results if r.get("data_format") == "TABLE"]
            print(f"\n{doc_dir}  ({len(tables)} tables, {len(results)} total)")
            for i, r in enumerate(tables):
                val = r.get("value", {})
                cells = val.get("cells", []) if isinstance(val, dict) else []
                print(f"  Table {i}: grid={val.get('grid_shape') if isinstance(val,dict) else '?'} cells={len(cells)}")
                show_grid(cells)
else:
    print(f"NOT FOUND: {bp}")
    for bd in os.listdir(PREV_BATCHES):
        print(f"  available: {bd}")
