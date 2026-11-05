"""
Trace Drawing Number through the full pipeline for 01_original.png.
Run with: py -3 trace_drawing_number.py
"""
import sys, os, json, gzip, tempfile, pathlib, subprocess, uuid, shutil

RUNTIME_PYTHON = r"C:\Users\benja\AppData\Local\Temp\oculi-mock-150\runtime\python\python.exe"
SESSION_RUNTIME = str(pathlib.Path(__file__).parent / "bk-main" / "session_runtime.py")
INPUT_FILE = r"C:\Users\benja\OneDrive\Desktop\data\01_original.png"

QUERY_PACKAGE = {
    "schema_version": 1,
    "freetext_queries": [],
    "structured_fields": [
        {"field_name": "Drawing Number"},
        {"field_name": "Drawing Description"},
        {"field_name": "Revision"},
        {"field_name": "CAGE Code"},
    ],
    "task_ids": [],
}

def run(cmd, env):
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)

def main():
    with tempfile.TemporaryDirectory(prefix="oculi_trace_") as tmpdir:
        run_id = "trace_" + str(uuid.uuid4())[:8]
        session_dir = pathlib.Path(tmpdir) / "sessions" / run_id
        session_dir.mkdir(parents=True)

        qp_path = session_dir / "query_package.json"
        qp_path.write_text(json.dumps(QUERY_PACKAGE), encoding="utf-8")

        input_dir = pathlib.Path(tmpdir) / "input"
        input_dir.mkdir()
        shutil.copy2(INPUT_FILE, input_dir / pathlib.Path(INPUT_FILE).name)

        env = os.environ.copy()
        env["DEC_APPDATA_ROOT"] = tmpdir

        print("\n=== STEP 1: EXTRACT ===")
        r = run([RUNTIME_PYTHON, SESSION_RUNTIME, "extract",
                 "--run-id", run_id, "--input-path", str(input_dir)], env)
        print(f"Exit: {r.returncode}")
        if r.stderr.strip():
            print("[STDERR tail]\n" + r.stderr.strip()[-2000:])

        print("\n=== STEP 1b: RESOLVE FROM DISK ===")
        r = run([RUNTIME_PYTHON, SESSION_RUNTIME, "resolve_from_disk",
                 "--run-id", run_id, "--query-package", str(qp_path)], env)
        print(f"Exit: {r.returncode}")
        if r.stderr.strip():
            print("[STDERR tail]\n" + r.stderr.strip()[-1000:])

        print("\n=== STEP 2: EXTRACTION CACHE ===")
        cache_path = session_dir / "extraction_cache.json.gz"
        if cache_path.exists():
            with gzip.open(str(cache_path), "rt", encoding="utf-8") as f:
                cache = json.load(f)
            for doc in cache.get("documents", []):
                for page in doc.get("pages", []):
                    print(f"  Page: {page.get('page_id')} status={page.get('status')}")
                    cluster_results = page.get("cluster_results") or {}
                    for key in ("Cluster_Line", "Cluster_Block", "Cluster_Line_Raw"):
                        for item in (cluster_results.get(key) or []):
                            t = str(item.get("text") or item.get("text_raw") or "").lower()
                            if any(k in t for k in ("dwg", "drawing", "7008", "drg", "doc no")):
                                print(f"    [{key}] id={item.get('line_id') or item.get('block_id')} text={item.get('text_raw')} bbox={item.get('bbox')}")

        print("\n=== STEP 3: RESOLVED RESULTS ===")
        resolved_path = session_dir / "resolved_results.json"
        if resolved_path.exists():
            resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
            for doc_id, doc_agg in (resolved.get("document_results") or {}).items():
                print(f"\n  Document: {doc_id}")
                entries = (doc_agg.get("document_summary") or {}).get("entries") or []
                print(f"  Total entries: {len(entries)}")
                for e in entries:
                    print(f"    query={e.get('query_name')} | value={str(e.get('value'))[:80]} | conf={e.get('confidence')}")
                print("\n  -- Raw page results --")
                for pg_id, pg in (doc_agg.get("pages") or {}).items():
                    for e in (pg.get("results") or []):
                        print(f"    [{pg_id}] query={e.get('query_name')} | fmt={e.get('data_format')} | value={str(e.get('value'))[:60]} | conf={e.get('confidence')}")
        else:
            print("  No resolved_results.json found")

if __name__ == "__main__":
    main()
