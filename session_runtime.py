import argparse
import copy
import csv
import gc
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, AbstractSet, FrozenSet, Optional, Tuple

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in os.sys.path:
    os.sys.path.insert(0, _BASE_DIR)
for _p in ("input", "workers", "cleaners", "bootstrap"):
    _full = os.path.join(_BASE_DIR, _p)
    if _full not in os.sys.path:
        os.sys.path.insert(0, _full)

import cv2
import numpy as np
from process_input import (
    PageData,
    DocumentPrefetcher,
    SUPPORTED_EXT,
    cleanup_batch_temp_dirs,
    expected_total_pages,
    scan_input_local,
)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pipeline import (
    BatchExtractionResult,
    DocumentExtractionResult,
    InternalValidator,
    PageExtractionResult,
    aggregate_document_results,
    build_digitised_document_results,
    extract_document_pages_only,
    process_input_local,
    process_page_data_cleaning,
    process_page_relationship_resolution,
    sidecar_export_document,
    gpu_health_check,
    _get_max_workers,
    _init_worker,
    ResilientProcessPool,
)
# Dedicated post-processing worker (workers/ is on sys.path via the pipeline
# import above). post_process_document is ALSO the inline implementation when
# the worker is disabled — one code path for both modes.
from post_worker import init_post_worker, post_process_document

APP_ROOT = Path(os.environ.get("DEC_APPDATA_ROOT", Path.home() / "AppData" / "Local" / "DocumentExtractionConsole"))
SESSIONS_ROOT = APP_ROOT / "sessions"
ACTIVITY_CACHE_PATH = APP_ROOT / "activity_cache.json"
EXPECTED_QUERY_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parent
QUERY_SET_PATH = PROJECT_ROOT / "config" / "query_set.yaml"
TASK_REGISTRY_PATH = PROJECT_ROOT / "config" / "task_registry.json"
SETTINGS_PATH = APP_ROOT / "settings.json"
_CROSS_REFERENCE_MEMORY: Dict[str, Dict[str, Any]] = {}
_DEBUG_LOGS_ENABLED = os.environ.get("PIPELINE_DEBUG_LOGS", "0").strip() == "1"

PAGE_PENDING = "pending"
PAGE_EXTRACTING = "extracting"
PAGE_EXTRACTED = "extracted"
PAGE_PARTIAL = "partial"
PAGE_SKIPPED = "skipped"
PAGE_HITL_PENDING = "hitl_pending"
PAGE_HITL_CORRECTED = "hitl_corrected"


def is_export_complete_only(status_counts: Dict[str, int], hitl_state: str = "none", human_verified: bool = False) -> bool:
    incomplete_keys = {"pending", "extracting", PAGE_SKIPPED, PAGE_PARTIAL, PAGE_HITL_PENDING}
    for key, value in status_counts.items():
        if key in incomplete_keys and int(value or 0) > 0:
            return False
    if hitl_state == PAGE_HITL_PENDING:
        return False
    if hitl_state == PAGE_HITL_CORRECTED:
        return True
    return human_verified or bool(status_counts.get(PAGE_CLEANED, 0) > 0 or status_counts.get(PAGE_EXTRACTED, 0) > 0)
PAGE_CLEANED = "cleaned"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _completed_documents_dir(session_dir: Path) -> Path:
    d = session_dir / "completed_documents"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Longest full path this module will construct. Windows MAX_PATH is 260, and
# long-path support is OFF by default and not enabled on the production machines
# (HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled). 240 leaves
# room for the ".tmp" form and a little headroom for a deeper session root.
_MAX_OUTPUT_PATH = 240

# How many GPU health probes to spread across one run.
#
# The probe costs ~5.7s (it spawns a process, imports torch and onnxruntime and builds
# a CUDA session), and it used to run before EVERY document -- about 19 minutes of
# probing on a 200-document batch. That cadence was inherited from before crash
# isolation existed: its job was to take a native crash in a throwaway process rather
# than in a long-lived worker. ResilientProcessPool now handles a dying worker
# reactively and for free, and _note_page_failure covers a GPU that goes sick without
# killing anything, so the probe is now only a periodic sanity check.
_GPU_PROBES_PER_RUN = 8


def _sanitize_component_for_filename(name: str) -> str:
    """Make a single name safe to use as a Windows filename."""
    out = []
    for ch in (name or "").strip():
        out.append("_" if (ch in '<>:"/\\|?*' or ord(ch) < 32) else ch)
    return "".join(out).strip(". ")   # Windows: no trailing dot/space


def _completed_document_stem(rel: str, out_dir: Path) -> str:
    """File stem for a completed document: the source file's OWN name, with the
    folder path dropped and the length budgeted against ``out_dir``.

    This previously flattened the entire relative path into the stem, separators
    and all. On a corpus nested several folders deep that produced names long
    enough to push the full path past MAX_PATH, and the write failed with
    FileNotFoundError -- which was not caught, so a single document ended a run
    that had been going for days. Measured on the failing batch: a 164-character
    relative path produced a 281-character output path.

    Dropping the folders costs nothing: the full relative path is still carried
    inside the payload as ``relative_path`` / ``relative_folder``, so the export
    can still rebuild the folder structure. Two documents sharing a basename are
    handled by the caller's collision guard.

    The stem is truncated to whatever the output directory leaves room for, with
    a hash of the full relative path appended so a truncated name stays unique.
    """
    rel = (rel or "").strip().replace("\\", "/").strip("/")
    base = rel.rsplit("/", 1)[-1] if rel else ""
    stem = _sanitize_component_for_filename(base) or "document"

    # Longest suffix the caller can add downstream: ".json.gz" + ".tmp", plus the
    # "__xxxxxxxx" the collision guard may insert.
    reserve = len(".json.gz.tmp") + len("__") + 8
    budget = _MAX_OUTPUT_PATH - len(str(out_dir)) - 1 - reserve

    if budget < 24:
        # The session root alone has eaten the budget. Nothing sensible is left
        # to trim, so warn rather than silently emitting a path that will fail.
        logging.warning(
            "Completed-documents path is close to the Windows limit (%d chars): %s",
            len(str(out_dir)), out_dir,
        )
        budget = 24

    if len(stem) > budget:
        digest = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:8]
        keep = max(1, budget - len("__") - len(digest))
        stem = (stem[:keep].strip(". ") or "document") + "__" + digest

    return stem


def _derive_session_status(session_meta: Dict[str, Any]) -> str:
    """Normalized session status derived from phase + exit_code."""
    phase = str(session_meta.get("phase", "") or "")
    exit_code = session_meta.get("exit_code", 0)
    if phase == "cancelled":
        return "cancelled"
    if phase.endswith("_failed") or phase == "failed" or (isinstance(exit_code, int) and exit_code != 0):
        return "failed"
    if phase == "resolve_complete":
        return "completed"
    return "processing"


def _write_session_meta(path: Path, session_meta: Dict[str, Any]) -> None:
    """Single writer for session_meta.json: stamps a normalized `status` and a
    one-shot `completed_utc` (derived from phase/exit_code) so every write site
    stays consistent, then persists. All session_meta writes go through here."""
    status = _derive_session_status(session_meta)
    session_meta["status"] = status
    if status == "completed" and not session_meta.get("completed_utc"):
        session_meta["completed_utc"] = utc_now()
    write_json(path, session_meta)


def _session_source_hash(session_dir: Path) -> str:
    """Batch-level source hash: sha256 over the input manifest (relative paths +
    sizes + mtimes) already written to input_fingerprint.json. Cheap — no file
    bytes are re-read; uniquely identifies the uploaded folder for this run."""
    try:
        ifp = json.loads((session_dir / "input_fingerprint.json").read_text(encoding="utf-8"))
        return _stable_hash(ifp)
    except Exception:
        return ""


def _skipped_pages_notes(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Structured record of every skipped/partial/errored page across the run,
    for session_meta `notes`. Keyed by the ORIGINAL document identity."""
    notes: List[Dict[str, Any]] = []
    for d in documents or []:
        ident = d.get("original_document_id") or d.get("relative_path") or d.get("document_id")
        for pg in d.get("page_statuses", d.get("pages", [])) or []:
            st = pg.get("status", "")
            errs = pg.get("errors") or []
            if st in {PAGE_SKIPPED, PAGE_PARTIAL} or errs:
                notes.append({
                    "original_document_id": ident,
                    "folder_id": d.get("folder_id") or d.get("relative_folder") or "",
                    "page_id": pg.get("page_id"),
                    "page_number": pg.get("page_number"),
                    "status": st,
                    "errors": errs,
                })
    return notes


def _persist_completed_document(session_dir: Path, doc_entry: Dict[str, Any]) -> None:
    """Write one document's FULL extraction result to disk immediately.

    Called right after each document finishes extraction. If the process
    crashes later (e.g. native 0xC0000005, OOM kill, server restart), every
    document persisted here before the crash is safely on disk and can be
    picked up on resume — including its resolved results, not just a marker.

    Written gzip-compressed: resolved-results JSON compresses ~10x, so a full
    multi-thousand-document run costs tens of MB, not GB, and session
    retention pruning removes it after a few days regardless.

    The FILE is named by the sanitized original relative path (human-readable),
    NOT the internal hash id. The internal ``document_id`` is kept inside the
    payload so the resume loader / queue still key on it — the filename change
    does not affect resume dedup.
    """
    import gzip
    internal_id = doc_entry.get("document_id", "unknown_doc")
    out_dir = _completed_documents_dir(session_dir)
    stem = _completed_document_stem(
        doc_entry.get("original_document_id")
        or doc_entry.get("relative_path")
        or internal_id,
        out_dir,
    )
    out_path = out_dir / f"{stem}.json.gz"
    # Collision guard: if a DIFFERENT document already owns this sanitized name
    # (two logical paths sanitizing to the same string — rare), disambiguate with
    # a short internal-id suffix. A same-id re-persist overwrites in place.
    if out_path.exists():
        existing_id = None
        try:
            with gzip.open(out_path, "rb") as f:
                existing_id = (json.loads(f.read().decode("utf-8")) or {}).get("document_id")
        except Exception:
            existing_id = None
        if existing_id and existing_id != internal_id:
            suffix = hashlib.sha256(str(internal_id).encode("utf-8")).hexdigest()[:8]
            out_path = out_dir / f"{stem}__{suffix}.json.gz"
    tmp_path = out_dir / f"{out_path.name}.tmp"
    payload = json.dumps(_json_safe(doc_entry), ensure_ascii=False).encode("utf-8")
    with gzip.open(tmp_path, "wb", compresslevel=6) as f:
        f.write(payload)
    tmp_path.replace(out_path)  # never leave a half-written entry


def _completed_document_id_from_name(name: str) -> Optional[str]:
    if name.endswith(".json.gz"):
        return name[: -len(".json.gz")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return None


def _load_completed_document_ids(session_dir: Path) -> set[str]:
    """Internal document_ids already persisted from a previous attempt at this run.

    The filename is now the human-readable relative path, not the id, so the id
    is read from INSIDE each file. Legacy hash-named files still carry the same
    id in their payload, so this is backward-compatible; if a file can't be
    parsed, fall back to the legacy name-derived id."""
    import gzip
    d = session_dir / "completed_documents"
    if not d.exists():
        return set()
    ids: set[str] = set()
    for p in d.iterdir():
        if p.name.endswith(".tmp"):
            continue
        did = None
        try:
            if p.name.endswith(".json.gz"):
                with gzip.open(p, "rb") as f:
                    did = (json.loads(f.read().decode("utf-8")) or {}).get("document_id")
            elif p.name.endswith(".json"):
                did = (json.loads(p.read_text(encoding="utf-8")) or {}).get("document_id")
            else:
                continue
        except Exception:
            did = None
        if not did:
            did = _completed_document_id_from_name(p.name)  # legacy fallback
        if did:
            ids.add(did)
    return ids


def _load_persisted_extraction_docs(session_dir: Path) -> List[Dict[str, Any]]:
    """Load all previously-persisted per-document results, for resuming a run.

    Reads gzip entries (current format) and plain .json (older builds); when a
    document exists in both formats the gzip entry wins.
    """
    import gzip
    d = session_dir / "completed_documents"
    if not d.exists():
        return []
    by_id: Dict[str, Dict[str, Any]] = {}
    gz_ids: set[str] = set()
    for p in sorted(d.iterdir()):
        if p.name.endswith(".tmp"):
            continue
        is_gz = p.name.endswith(".json.gz")
        if not is_gz and not p.name.endswith(".json"):
            continue
        try:
            if is_gz:
                with gzip.open(p, "rb") as f:
                    entry = json.loads(f.read().decode("utf-8"))
            else:
                entry = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            logging.warning("Failed to load persisted document %s; skipping it on resume", p)
            continue
        # Key on the INTERNAL document_id from inside the payload (filename is now
        # the human-readable relative path); fall back to the legacy name-derived id.
        doc_id = (entry or {}).get("document_id") or _completed_document_id_from_name(p.name)
        if not doc_id:
            continue
        if is_gz:
            by_id[doc_id] = entry
            gz_ids.add(doc_id)
        elif doc_id not in gz_ids:
            by_id.setdefault(doc_id, entry)  # plain .json only if no gzip entry won
    return [by_id[k] for k in sorted(by_id)]


def _persist_run_start_time(session_dir: Path, started_all: float) -> None:
    """Persist the original run start time so ETA/rate calculations are correct on resume."""
    marker = session_dir / "_extraction_start_time.txt"
    marker.write_text(str(started_all), encoding="utf-8")


def _load_run_start_time(session_dir: Path, default: float) -> float:
    """Load the original run start time, or return default if none was persisted."""
    marker = session_dir / "_extraction_start_time.txt"
    if marker.exists():
        try:
            return float(marker.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default


def _persist_expected_document_count(session_dir: Path, count: int) -> None:
    """Persist the expected total document count so a crashed run can later
    report "X of Y documents resolved" even after the live BatchData is gone."""
    write_json(session_dir / "expected_document_count.json", {"count": count})


def _read_expected_document_count(session_dir: Path) -> Optional[int]:
    p = session_dir / "expected_document_count.json"
    if not p.exists():
        return None
    try:
        return int(json.loads(p.read_text(encoding="utf-8")).get("count"))
    except Exception:
        return None


def _persist_batch_id(session_dir: Path, batch_id: str) -> None:
    write_json(session_dir / "batch_id.json", {"batch_id": batch_id})


def _read_batch_id(session_dir: Path, fallback: str) -> str:
    p = session_dir / "batch_id.json"
    if not p.exists():
        return fallback
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("batch_id", fallback))
    except Exception:
        return fallback


def _debug_c9d468_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    if not _DEBUG_LOGS_ENABLED:
        return
    try:
        rec = {
            "sessionId": "c9d468",
            "runId": str(run_id),
            "hypothesisId": str(hypothesis_id),
            "location": str(location),
            "message": str(message),
            "data": _json_safe(data),
            "timestamp": int(time.time() * 1000),
        }
        with open(str(PROJECT_ROOT / "debug-c9d468.log"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    APP_ROOT.mkdir(parents=True, exist_ok=True)


def get_session_retention_days() -> int:
    """Days a session survives a `purge` that was given no explicit --days.

    Effectively unlimited by design. Automatic purging was removed because the digitised
    document is a RETAINED OUTPUT, not working state, and the purge deleted whole session
    folders -- digitised documents, completed documents, review data and exports together,
    on every application start, for anything older than two days. An explicit
    `purge --days N` or `purge --all` still does exactly what it is asked to.
    """
    if SETTINGS_PATH.exists():
        try:
            payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            val = int(payload.get("SessionRetentionDays", 3650))
            return max(1, val)
        except Exception:
            return 3650
    return 3650


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _stable_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_input_fingerprint(input_path: str) -> Dict[str, Any]:
    base = Path(input_path)
    files: List[Dict[str, Any]] = []
    if base.exists():
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            # Only files the pipeline will actually ingest. _collect_input_entries skips
            # anything outside SUPPORTED_EXT, so recording them here described a corpus that
            # does not exist -- a folder holding 2 PDFs beside 6 stray files fingerprinted as
            # 8 -- and dropping an unrelated .db or .xlsx next to the inputs invalidated the
            # fingerprint, forcing a re-extraction that could not change the result.
            # Archives stay: .zip/.tar are in SUPPORTED_EXT and their contents feed the run.
            if p.suffix.lower() not in SUPPORTED_EXT:
                continue
            rel = str(p.relative_to(base)).replace("\\", "/")
            stat = p.stat()
            files.append({"path": rel, "size": stat.st_size, "mtime": int(stat.st_mtime)})
    return {"input_path": str(base.resolve()), "files": files, "file_count": len(files)}


def build_extract_config(module_a_enabled: bool, page_timeout_sec: int) -> Dict[str, Any]:
    return {
        "module_a_enabled": bool(module_a_enabled),
        "page_extract_timeout_sec": int(page_timeout_sec),
    }


def apply_page_timeout(page_timeout_sec: int) -> int:
    """Make --page-timeout-sec actually govern extraction.

    It previously did not. `build_extract_config` wrote `page_extract_timeout_sec` into the
    session fingerprint and NOTHING read it back, while pipeline used its own module-level
    default from the env var. So the flag (and its default) were inert and the effective
    timeout was always pipeline's, regardless of what the caller asked for or what the
    fingerprint on disk recorded.

    `pipeline._run_with_timeout` reads PIPELINE_PAGE_EXTRACT_TIMEOUT_SEC at CALL time, so
    assigning the module attribute here works even though pipeline is imported at the top
    of this file. Returns the value in force, for logging.
    """
    try:
        import pipeline as _pl
        value = int(page_timeout_sec)
        if value > 0:
            _pl.PIPELINE_PAGE_EXTRACT_TIMEOUT_SEC = value
        return int(_pl.PIPELINE_PAGE_EXTRACT_TIMEOUT_SEC)
    except Exception:
        logging.warning("Could not apply page timeout %s; pipeline default stands",
                        page_timeout_sec, exc_info=True)
        return int(page_timeout_sec)


def write_session_fingerprints(
    session_dir: Path,
    *,
    input_path: str,
    module_a_enabled: bool,
    page_timeout_sec: int,
) -> None:
    input_fp = build_input_fingerprint(input_path)
    write_json(session_dir / "input_fingerprint.json", input_fp)
    extract_cfg = build_extract_config(module_a_enabled, page_timeout_sec)
    _in_force = apply_page_timeout(page_timeout_sec)
    logging.info("Per-page extraction timeout in force: %ss", _in_force)
    write_json(session_dir / "extraction_config.json", extract_cfg)
    write_json(session_dir / "extract_config_fingerprint.json", {"sha256": _stable_hash(extract_cfg)})


def append_event(session_dir: Path, event: Dict[str, Any]) -> None:
    event_path = session_dir / f"{session_dir.name}_events.ndjson"
    with event_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(event), ensure_ascii=False) + "\n")
        f.flush()


def emit_run_failed(
    session_dir: Path,
    *,
    reason: str,
    message: str,
    phase: str,
    exit_code: int,
    expected_schema_version: int | None = None,
    got: Any = None,
) -> None:
    event: Dict[str, Any] = {
        "type": "run_failed",
        "run_id": session_dir.name,
        "reason": reason,
        "phase": phase,
        "exit_code": exit_code,
        "message": message,
        "ts": utc_now(),
    }
    if expected_schema_version is not None:
        event["expected_schema_version"] = expected_schema_version
    if got is not None:
        event["got"] = got
    append_event(session_dir, event)


def load_activity_cache() -> List[Dict[str, Any]]:
    if not ACTIVITY_CACHE_PATH.exists():
        return []
    with ACTIVITY_CACHE_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return payload.get("rows", [])


def save_activity_cache(rows: List[Dict[str, Any]]) -> None:
    write_json(ACTIVITY_CACHE_PATH, {"rows": rows, "updated_utc": utc_now()})


def upsert_activity_row(run_id: str, batch_name: str, status: str, session_dir: Path) -> None:
    rows = load_activity_cache()
    now = utc_now()
    found = False
    for row in rows:
        if row.get("run_id") == run_id:
            row["status"] = status
            row["updated_utc"] = now
            found = True
            break
    if not found:
        rows.append(
            {
                "run_id": run_id,
                "batch_name": batch_name,
                "status": status,
                "created_utc": now,
                "updated_utc": now,
                "session_path": str(session_dir),
                "input_path": "",
            }
        )
    save_activity_cache(rows)


def set_activity_input_path(run_id: str, input_path: str) -> None:
    rows = load_activity_cache()
    changed = False
    for row in rows:
        if row.get("run_id") == run_id:
            row["input_path"] = input_path
            row["updated_utc"] = utc_now()
            changed = True
            break
    if changed:
        save_activity_cache(rows)


def _active_structured_field_names(query_package: Dict[str, Any]) -> Optional[FrozenSet[str]]:
    """Return None = score all YAML fields (legacy). Empty frozenset = no structured fields selected."""
    if "structured_fields" not in query_package:
        return None
    names = [
        str(f.get("field_name", "")).strip()
        for f in (query_package.get("structured_fields") or [])
        if isinstance(f, dict) and f.get("field_name")
    ]
    return frozenset(names)


def _extract_query_values_for_export(
    aggregated: Any,
    allowed_names: Optional[AbstractSet[str]],
) -> Dict[str, Any]:
    """Best (highest-confidence) value per query_name from document_summary.entries."""
    if not isinstance(aggregated, dict):
        return {}
    entries = (aggregated.get("document_summary") or {}).get("entries") or []
    best: Dict[str, tuple] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        qn = e.get("query_name")
        if not qn:
            continue
        if allowed_names is not None and qn not in allowed_names:
            continue
        try:
            conf = float(e.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        val = e.get("value")
        if qn not in best or conf > best[qn][0]:
            best[qn] = (conf, val)
    return {k: v[1] for k, v in best.items()}


def _parse_page_number_from_page_id(page_id: Any) -> Optional[int]:
    pid = str(page_id or "")
    m = re.search(r"_PG(\d+)$", pid, re.I)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _select_hitl_issue_page(doc_result: DocumentExtractionResult, aggregated: Any) -> tuple[str, Optional[int]]:
    # Prefer pages that already have extraction failures/partial status.
    for page_id, pr in (doc_result.page_results or {}).items():
        raw = str(getattr(pr, "status", "") or "").upper()
        if raw in {"SKIPPED", "ERROR", "PARTIAL"}:
            pn = getattr(pr, "page_number", None)
            try:
                page_number = int(pn) if pn is not None else None
            except (TypeError, ValueError):
                page_number = None
            if page_number is None:
                page_number = _parse_page_number_from_page_id(page_id)
            return str(page_id), page_number

    # Otherwise pick the lowest-confidence extracted entry page as HITL landing page.
    entries = (aggregated.get("document_summary") or {}).get("entries") or [] if isinstance(aggregated, dict) else []
    best_pid = ""
    best_conf = float("inf")
    for e in entries:
        if not isinstance(e, dict):
            continue
        pid = str(e.get("page_id") or "").strip()
        if not pid:
            continue
        try:
            conf = float(e.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < best_conf:
            best_conf = conf
            best_pid = pid
    if best_pid:
        return best_pid, _parse_page_number_from_page_id(best_pid)
    return "", None


def _collect_all_hitl_issue_pages(
    doc_result: Any,
    aggregated: Any,
    doc_id: str,
    extraction_docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collect every page that has issues: skipped/partial status, low-confidence resolved
    field values, or missing field values.  Returns list ordered skipped/partial first then
    by ascending page number.  Each entry carries source_path, overlay_rects (all OCR
    rectangles on that page), issue_fields, and issue_reasons."""
    # Build page_id -> {source_path, all_rects} from extraction_docs.
    # NEW FORMAT: use skipped_pages_rect_data (only kept for skipped/error pages).
    # LEGACY FORMAT: use pages[].rect_data.
    page_info: Dict[str, Dict[str, Any]] = {}
    for d in extraction_docs:
        if str(d.get("document_id")) != str(doc_id):
            continue
        # NEW FORMAT: skipped_pages_rect_data
        skipped_rect_data = d.get("skipped_pages_rect_data") or {}
        page_statuses = d.get("page_statuses", d.get("pages", []))
        for ps in page_statuses:
            pid = str(ps.get("page_id") or "")
            if not pid:
                continue
            rects: List[List[float]] = []
            # Try skipped_pages_rect_data first (new format), then legacy pages[].rect_data
            rect_data = skipped_rect_data.get(pid) or ps.get("rect_data") or {}
            if isinstance(rect_data, dict):
                rectangles_raw = rect_data.get("rectangles") or {}
                rect_items = rectangles_raw.values() if isinstance(rectangles_raw, dict) else rectangles_raw
                for rect in rect_items:
                    bbox = rect.get("bbox") if isinstance(rect, dict) else None
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        rects.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
            page_info[pid] = {"source_path": str(ps.get("source_path") or ""), "all_rects": rects}

    # Build page_id -> field_name -> anchor_bbox from resolved entries (field-specific highlighting)
    field_anchor_bboxes: Dict[str, Dict[str, List[float]]] = {}
    entries = (
        (aggregated.get("document_summary") or {}).get("entries") or []
        if isinstance(aggregated, dict)
        else []
    )
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pid = str(entry.get("page_id") or "").strip()
        qname = str(entry.get("query_name") or "").strip()
        abbox = entry.get("anchor_bbox")
        if pid and qname and isinstance(abbox, (list, tuple)) and len(abbox) >= 4:
            field_anchor_bboxes.setdefault(pid, {})[qname] = [float(v) for v in abbox[:4]]

    issue_pages: List[Dict[str, Any]] = []
    seen: set = set()

    def _add_or_merge(page_id: str, reason: str, field: str = "") -> None:
        if page_id in seen:
            for ip in issue_pages:
                if ip["page_id"] == page_id:
                    if reason not in ip["issue_reasons"]:
                        ip["issue_reasons"].append(reason)
                    if field and field not in ip["issue_fields"]:
                        ip["issue_fields"].append(field)
            return
        seen.add(page_id)
        pn = _parse_page_number_from_page_id(page_id)
        info = page_info.get(page_id, {})
        issue_pages.append({
            "page_id": page_id,
            "page_number": pn,
            "source_path": info.get("source_path", ""),
            "overlay_rects": [],  # populated in post-processing below
            "issue_fields": [field] if field else [],
            "issue_reasons": [reason],
        })

    # Phase 1: pages that were skipped / errored / partial during extraction
    for page_id, pr in (doc_result.page_results or {}).items():
        raw = str(getattr(pr, "status", "") or "").upper()
        if raw in {"SKIPPED", "ERROR", "PARTIAL"}:
            _add_or_merge(str(page_id), "skipped_or_partial_page")

    # Phase 2: low-confidence or empty resolved field values
    entries = (
        (aggregated.get("document_summary") or {}).get("entries") or []
        if isinstance(aggregated, dict)
        else []
    )
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        conf = str(entry.get("confidence") or "").lower()
        query_name = str(entry.get("query_name") or "")
        val = entry.get("value")
        is_low_conf = conf == "low"
        is_empty = val is None or val == "" or val == []
        if not (is_low_conf or is_empty) or not query_name:
            continue
        page_id = str(entry.get("page_id") or "").strip()
        reason = f"low_confidence:{query_name}" if is_low_conf else f"missing_value:{query_name}"
        if page_id:
            _add_or_merge(page_id, reason, query_name)

    # Sort: skipped/partial pages first, then ascending page number
    def _sort_key(ip: Dict[str, Any]) -> tuple:
        has_skip = any("skipped_or_partial" in r for r in ip.get("issue_reasons", []))
        return (0 if has_skip else 1, ip.get("page_number") or 999)

    issue_pages.sort(key=_sort_key)

    # Post-process: assign overlay_rects as field-specific anchor bboxes (ordered to match
    # issue_fields so rect index == field index for click-to-field mapping in the UI).
    # Fall back to all OCR rects only for skipped/partial pages that have no field bboxes.
    for ip in issue_pages:
        pid = ip["page_id"]
        page_field_bboxes = field_anchor_bboxes.get(pid, {})
        field_rects: List[List[float]] = []
        for f in ip.get("issue_fields", []):
            fb = page_field_bboxes.get(f)
            if isinstance(fb, list) and len(fb) >= 4:
                field_rects.append(fb)
        if field_rects:
            ip["overlay_rects"] = field_rects
        else:
            # Fallback: all OCR rects (e.g. skipped/partial pages with no resolved anchors)
            ip["overlay_rects"] = page_info.get(pid, {}).get("all_rects", [])

    return issue_pages


def parse_query_package(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    # region agent log
    _debug_c9d468_log(
        "pre-fix",
        "H5",
        "session_runtime.py:parse_query_package",
        "raw_query_package_freetext_shape",
        {
            "query_package_path": str(path),
            "freetext_type": type(payload.get("freetext_queries")).__name__,
            "freetext_count": len(payload.get("freetext_queries", []) or []),
            "freetext_sample": (payload.get("freetext_queries", []) or [])[:10],
        },
    )
    # endregion
    schema_version = payload.get("schema_version")
    if schema_version != EXPECTED_QUERY_SCHEMA_VERSION:
        raise ValueError(f"schema_version_mismatch:{schema_version}")
    allowed_fields = load_allowed_query_fields()
    allowed_tasks = load_allowed_task_ids()
    unknown_fields = [
        f.get("field_name")
        for f in payload.get("structured_fields", [])
        if isinstance(f, dict) and f.get("field_name") not in allowed_fields
    ]
    if unknown_fields:
        raise ValueError(f"unknown_structured_fields:{','.join(sorted(set(str(x) for x in unknown_fields)))}")
    unknown_tasks = [t for t in payload.get("task_ids", []) if t not in allowed_tasks]
    if unknown_tasks:
        raise ValueError(f"unknown_task_ids:{','.join(sorted(set(str(x) for x in unknown_tasks)))}")
    return payload


def load_allowed_query_fields() -> set[str]:
    if not QUERY_SET_PATH.exists():
        return set()
    fields: set[str] = set()
    for line in QUERY_SET_PATH.read_text(encoding="utf-8").splitlines():
        trimmed = line.strip()
        if trimmed.startswith("- name:"):
            fields.add(trimmed.split(":", 1)[1].strip())
        elif trimmed.startswith("- "):
            # fallback for simple queries list
            value = trimmed[2:].strip()
            if value and ":" not in value:
                fields.add(value)
    return fields


def load_allowed_task_ids() -> set[str]:
    if not TASK_REGISTRY_PATH.exists():
        return set()
    try:
        payload = json.loads(TASK_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return set()
    return {
        str(item.get("task_id"))
        for item in payload.get("tasks", [])
        if isinstance(item, dict) and item.get("task_id")
    }


def _extract_in_memory(session_dir: Path, run_id: str, input_path: str) -> Dict[str, Any]:
    # A cancel.request left over from a previous attempt must not kill this
    # attempt — it belongs to the run that was cancelled, not to us.
    try:
        (session_dir / "cancel.request").unlink(missing_ok=True)
    except Exception:
        pass

    # Streaming ingest: scan builds metadata-only stubs for the whole corpus
    # (cheap), and pages are rendered just-in-time by a bounded prefetcher in
    # the loop below. Peak page-image memory is ~(INGEST_PREFETCH_DOCS + 1)
    # documents regardless of corpus size — eager ingestion of an 8000-document
    # upload previously materialized every 300-DPI page up front (45GB+).
    # Resume: reuse the original batch_id so document_ids (hash of
    # batch_id|relative_path) match the completion markers persisted by the
    # previous attempt.
    prior_batch_id = _read_batch_id(session_dir, "")
    batch = scan_input_local(input_path, batch_id=(prior_batch_id or None))
    _persist_expected_document_count(session_dir, batch.document_count)
    _persist_batch_id(session_dir, batch.batch_id)
    total_pages = expected_total_pages(batch)
    append_event(
        session_dir,
        {
            "type": "scan_complete",
            "run_id": run_id,
            "batch_id": batch.batch_id,
            "documents": batch.document_count,
            "expected_pages": total_pages,
            "resumed_batch_id": bool(prior_batch_id),
            "phase": "extract",
            "ts": utc_now(),
        },
    )
    processed_pages = 0
    started_all = time.time()
    extraction_docs: List[Dict[str, Any]] = []

    # Resume support: skip documents already persisted from a previous attempt at this run_id.
    already_done_ids = _load_completed_document_ids(session_dir)
    if already_done_ids:
        extraction_docs = _load_persisted_extraction_docs(session_dir)
        processed_pages = sum(len(doc.get("pages", [])) for doc in extraction_docs)
        started_all = _load_run_start_time(session_dir, started_all)
        logging.info(
            "Resuming run %s: %d documents already completed, skipping them",
            run_id, len(already_done_ids),
        )

    # Dedicated resolver worker (PIPELINE_RESOLVER_WORKER=1): cleaning +
    # resolution for document N run in their own PROCESS while this process
    # drives document N+1's extraction. Process isolation means Paddle (OCR
    # workers) and torch (resolver) never share an address space, and the OCR
    # pool gives up PIPELINE_RESOLVER_RESERVE slots (default 1) so the total
    # process count stays within the core budget.
    resolver_enabled = os.environ.get("PIPELINE_RESOLVER_WORKER", "").strip() == "1"
    _resolver_reserve = 0
    if resolver_enabled:
        _resolver_reserve = max(1, int(os.environ.get("PIPELINE_RESOLVER_RESERVE", "1") or 1))
        logging.info(
            "PIPELINE_RESOLVER_WORKER=1: dedicated post-processing worker enabled "
            "(OCR pool gives up %d slot(s))", _resolver_reserve,
        )
    _ocr_workers = max(1, _get_max_workers() - _resolver_reserve)

    executor = ResilientProcessPool(max_workers=_ocr_workers, initializer=_init_worker)
    resolver_pool: Optional[ProcessPoolExecutor] = (
        ProcessPoolExecutor(max_workers=1, initializer=init_post_worker) if resolver_enabled else None
    )
    first_document = True
    consecutive_gpu_health_failures = 0
    # Spread _GPU_PROBES_PER_RUN probes evenly across the documents this run will
    # actually process (resumed runs skip the already-done ones). Small runs fall back
    # to probing every document, which costs little when there are few of them.
    _docs_to_process = max(1, len(batch.documents) - len(set(already_done_ids or [])))
    gpu_health_probe_interval = max(1, _docs_to_process // _GPU_PROBES_PER_RUN)
    documents_since_gpu_probe = 0
    logging.info(
        "GPU health probe: every %d document(s) across %d to process (~%d probes this run)",
        gpu_health_probe_interval, _docs_to_process,
        max(1, _docs_to_process // gpu_health_probe_interval),
    )
    # Bounded render-ahead: the prefetcher renders the next documents' pages in
    # background threads while the current document is in OCR, and skips
    # already-completed documents BEFORE rendering them (fast resume).
    prefetcher = DocumentPrefetcher(
        batch.documents, skip_document_ids=set(already_done_ids or [])
    )
    # Rolling window for ETA: cumulative rate is polluted by stalls and by
    # resumed runs (dead time between attempts); recent throughput is what the
    # remaining-time estimate should reflect.
    from collections import deque as _deque
    _recent_pages: Any = _deque(maxlen=200)

    def _eta_seconds(processed: int) -> Optional[int]:
        _recent_pages.append((time.time(), processed))
        remaining = max(0, total_pages - processed)
        if remaining == 0:
            return 0
        if len(_recent_pages) >= 2:
            (t0, c0), (t1, c1) = _recent_pages[0], _recent_pages[-1]
            if (t1 - t0) >= 30.0 and c1 > c0:
                return int(remaining / ((c1 - c0) / (t1 - t0)))
        elapsed_all_ = max(0.001, time.time() - started_all)
        rate_ = processed / elapsed_all_
        return int(remaining / rate_) if rate_ > 0 else None

    # Overlap extraction with post-processing (lookahead-1 pipelining):
    # document N+1 extracts in the worker pool while document N is
    # cleaned/resolved here. STRICTLY OPT-IN via PIPELINE_OVERLAP_POST=1.
    # It defaulted on for CPU hosts once and caused a production incident:
    # extraction workers + render threads + main-thread torch/BLAS
    # oversubscribed a small CPU server (~15x slowdown), and the parallel
    # path's main-process OCR fallback then ran PaddleOCR concurrently with
    # sentence-transformers on another thread — two OpenMP runtimes in one
    # process, native crash 0x80000003. Sequential phases are the safe
    # default; enable overlap only on hosts with headroom, after testing.
    _overlap_env = os.environ.get("PIPELINE_OVERLAP_POST", "").strip()
    overlap_enabled = _overlap_env == "1"
    if resolver_enabled and overlap_enabled:
        # The resolver worker is the safe superset of thread-overlap: same
        # pipelining benefit, but post runs in an isolated process.
        logging.info("PIPELINE_RESOLVER_WORKER supersedes PIPELINE_OVERLAP_POST; using resolver worker")
        overlap_enabled = False
    if overlap_enabled:
        logging.info("PIPELINE_OVERLAP_POST=1: extraction/post-processing overlap enabled")
    extract_runner: Optional[ThreadPoolExecutor] = (
        ThreadPoolExecutor(max_workers=1) if overlap_enabled else None
    )
    # pending holds (document, future, started_ts) in overlap mode (future =
    # extraction) or (document, future, started_ts, raw_results) in resolver
    # mode (future = post-processing; raw results retained for fallback).
    pending: Optional[Tuple[Any, ...]] = None

    def _complete_pending_post(
        p_doc: Any, p_future: Any, p_started: float, p_results: Dict[str, Any]
    ) -> None:
        """Finish a resolver-worker document; on worker failure, recycle the
        pool and post-process inline so the document is never lost."""
        nonlocal resolver_pool
        post: Optional[Dict[str, Any]] = None
        try:
            post = p_future.result()
        except Exception:
            logging.exception(
                "Resolver worker failed for %s; recycling pool and resolving inline",
                p_doc.document_id,
            )
            if resolver_pool is not None:
                resolver_pool.shutdown(wait=False)
                resolver_pool = ProcessPoolExecutor(max_workers=1, initializer=init_post_worker)
        _finish_document(p_doc, p_results, p_started, post=post)

    def _finish_failed_document(
        p_doc: Any, p_started: float, reason: str, exc: BaseException
    ) -> None:
        """A whole document's EXTRACTION failed. Record it and keep the run going.

        The document loop is ``try: ... finally:`` with no ``except``, so every bare
        call inside it ends the run. Page-level failures have always been recorded
        as evidence and stepped over; document-level extraction failures were not,
        which is the same defect that let one unwritable filename kill a multi-day
        batch.

        Finishing with no page results is deliberate: ``_finish_document`` already
        materialises a SKIPPED result for every page it cannot find, so the document
        is reported with its pages marked rather than vanishing from the run.
        """
        doc_id = getattr(p_doc, "document_id", "unknown_doc")
        logging.exception("%s for %s; recording it and continuing", reason, doc_id)
        try:
            append_event(session_dir, {
                "type": "document_extraction_failed",
                "run_id": run_id,
                "document_id": doc_id,
                "relative_path": str((getattr(p_doc, "metadata", None) or {}).get("relative_path") or ""),
                "reason": reason,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "phase": "extract",
                "ts": utc_now(),
            })
        except Exception:
            # Never let the record of a failure become a second failure.
            logging.warning("Could not record document extraction failure", exc_info=True)
        try:
            _finish_document(p_doc, {}, p_started)
        except Exception:
            logging.exception(
                "Could not finalise failed document %s; continuing to the next", doc_id
            )

    def _post_doc_meta(document: Any) -> Dict[str, Any]:
        # source_filename rides along so cross-reference validation can match references
        # against real document names rather than the hash half of document_id.
        _meta = getattr(document, "metadata", None) or {}
        _name = str(_meta.get("filename") or "") or os.path.basename(
            str(getattr(document, "source_path", "") or "")
        )
        return {
            "document_id": document.document_id,
            "batch_id": batch.batch_id,
            "source_filename": _name,
        }

    def _post_page_stubs(document: Any) -> List[Dict[str, Any]]:
        """Image-free page descriptors for post-processing (pickle-light)."""
        return [
            {
                "page_id": p.page_id,
                "document_id": p.document_id,
                "batch_id": p.batch_id,
                "page_number": p.page_number,
                "width": p.width,
                "height": p.height,
                "storage_path": str(getattr(p, "storage_path", "") or ""),
                "metadata": dict(p.metadata or {}),
            }
            for p in document.pages
        ]

    def _strip_heavy_debug(results_by_page_id: Dict[str, Any]) -> Dict[str, Any]:
        """Drop page-sized debug arrays before pickling results to the resolver
        worker — cleaning/resolution consume OCR results, never pixels."""
        for r in results_by_page_id.values():
            r.masked_image = None
            r.frame_mask = None
            r.rect_vis = None
        return results_by_page_id

    def _finish_document(
        document: Any,
        results_by_page_id: Dict[str, Any],
        doc_started_ts: float,
        post: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Events, cleaning + resolution, aggregation, persistence, release for one extracted document.

        ``post`` is the post_process_document payload; when None it is computed
        inline here (default mode). Resolver-worker mode passes the payload
        computed in the dedicated process.
        """
        nonlocal processed_pages
        if post is None:
            try:
                post = post_process_document(
                    _post_doc_meta(document), _post_page_stubs(document), results_by_page_id
                )
            except Exception as _post_exc:
                # A document that breaks cleaning or resolution must not end the run.
                # The page-level contract has always been that a failure becomes evidence --
                # the page is recorded with its reason and the document continues -- but that
                # was never applied at document level. post_process_document runs the whole
                # cleaning and resolution stack, and _finish_document is called bare at all
                # four of its call sites, so one unusual document propagated to
                # run_orchestrate and killed a batch that had been running for days.
                #
                # The extracted text is kept. What is lost for this document is cleaning,
                # resolution and the digitised document -- recorded, not silent.
                logging.exception(
                    "Post-processing failed for %s; keeping extracted text and continuing",
                    document.document_id,
                )
                try:
                    append_event(session_dir, {
                        "type": "document_post_processing_failed",
                        "run_id": run_id,
                        "document_id": document.document_id,
                        "relative_path": str((getattr(document, "metadata", None) or {}).get("relative_path") or ""),
                        "error": str(_post_exc),
                        "error_type": type(_post_exc).__name__,
                        "phase": "extract",
                        "ts": utc_now(),
                    })
                except Exception:
                    # Never let the record of a failure become a second failure.
                    logging.warning("Could not record post-processing failure event", exc_info=True)
                post = {
                    "page_results": results_by_page_id,
                    "aggregated_results": {
                        "document_summary": {"entries": []},
                        "post_processing_failed": True,
                        "post_processing_error": str(_post_exc),
                        "post_processing_error_type": type(_post_exc).__name__,
                    },
                    "runtime_params_by_page": {},
                    "digitised_document": None,
                }
        # Page results evolved through cleaning/resolution supersede the raw
        # extraction results for everything below (events, persistence, sidecar).
        results_by_page_id = post["page_results"]
        doc_elapsed = max(0.001, time.time() - doc_started_ts)
        _doc_meta = getattr(document, "metadata", None) or {}
        _evt_rel_path = str(_doc_meta.get("relative_path") or "")
        _evt_rel_folder = str(_doc_meta.get("relative_folder") or "")
        doc_pages = []
        for page in document.pages:
            result = results_by_page_id.get(page.page_id)
            if result is None:
                result = PageExtractionResult(
                    page_id=page.page_id,
                    document_id=document.document_id,
                    batch_id=batch.batch_id,
                    status="SKIPPED",
                    errors=["not_extracted"],
                )
            elapsed = round(
                sum((result.extraction_timing or {}).values()) if result.extraction_timing else doc_elapsed / max(1, len(document.pages)),
                3,
            )
            processed_pages += 1
            eta_sec = _eta_seconds(processed_pages)
            page_status = PAGE_EXTRACTED if result.status == "EXTRACTED" else PAGE_SKIPPED
            if result.status in {"SKIPPED", "ERROR"}:
                append_event(
                    session_dir,
                    {
                        "type": "page_skipped",
                        "run_id": run_id,
                        "batch_id": batch.batch_id,
                        "document_id": document.document_id,
                        "relative_path": _evt_rel_path,
                        "relative_folder": _evt_rel_folder,
                        "page_id": page.page_id,
                        "page_status": page_status,
                        "reason": "stage_error",
                        "detail": "; ".join(result.errors)[:400],
                        "phase": "extract",
                        "stage": PAGE_SKIPPED,
                        "pct": round((processed_pages / max(1, total_pages)) * 100, 1),
                        "eta_sec": eta_sec,
                        "ts": utc_now(),
                    },
                )
            else:
                append_event(
                    session_dir,
                    {
                        "type": "page_completed",
                        "run_id": run_id,
                        "batch_id": batch.batch_id,
                        "document_id": document.document_id,
                        "relative_path": _evt_rel_path,
                        "relative_folder": _evt_rel_folder,
                        "page_id": page.page_id,
                        "pct": 100,
                        "elapsed_sec": elapsed,
                        "phase": "extract",
                        "stage": PAGE_EXTRACTED,
                        "progress_pct": round((processed_pages / max(1, total_pages)) * 100, 1),
                        "eta_sec": eta_sec,
                        "ts": utc_now(),
                    },
                )
            doc_pages.append(
                {
                    "page_id": page.page_id,
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "source_path": str(page.storage_path) if getattr(page, "storage_path", None) else str(input_path),
                    "source_page_index": page.page_number - 1,
                    "status": PAGE_SKIPPED if result.status in {"SKIPPED", "ERROR"} else PAGE_EXTRACTED,
                    "errors": result.errors,
                    "extraction_timing": result.extraction_timing,
                    "rect_data": result.rect_data,
                    "cluster_results": result.cluster_results,
                    "mask_info": result.mask_info,
                    "region_results": result.region_results,
                }
            )

        # Cleaning + resolution + aggregation already ran (inline above, or in
        # the dedicated resolver worker). Persist per-document REVIEW DATA here
        # (server/app path only) — the region bboxes+classes + OCR lines the HITL
        # classification-review stage needs. This is pure serialization of
        # already-computed output: NO re-extraction.
        #
        # The "NO sidecar" this comment used to claim stopped being true on 2026-11-06. The
        # graph export was moved out on 2026-07-22 and left "for reuse by the review-stage
        # finalize step"; that reuse was never built, so sidecar_export_document sat defined,
        # imported and called from nowhere, and the sidecar produced no parquet at all for
        # four months. It is called below, in this loop, because its own docstring requires
        # page.image to still be in memory and this is the last point before the release.
        try:
            from review_data import write_review_data  # sidecar dir is on sys.path via pipeline import
            write_review_data(str(session_dir), document, post["page_results"])
        except Exception:
            logging.debug("Review-data persist failed for %s", document.document_id, exc_info=True)

        # Page-quality rows for augmentation calibration. Must run BEFORE the page.image /
        # page.metadata release below (line ~1379), because the measurement is parked in
        # page.metadata["classifier_signals"]["page_quality"]. Reads metadata only - no
        # raster - and is gated on the same sidecar switch. Never raises.
        try:
            from page_quality_export import write_page_quality
            _pq_n = write_page_quality(str(session_dir), document)
            if _pq_n:
                logging.debug("page_quality: +%d row(s) for %s", _pq_n, document.document_id)
            # NOT finalised here. finalize_page_quality() rewrites the whole parquet, so
            # calling it per document would restore exactly the O(n^2) cost the append-only
            # JSONL was introduced to remove. The JSONL is the record and the inference side
            # reads it directly; convert once at the end of a capture run if a parquet is
            # wanted for transfer.
        except Exception:
            logging.debug("page_quality persist failed for %s", document.document_id, exc_info=True)

        # LayoutML graph export: nodes / pages / tables / edges / junctions parquet. This is
        # what produces the teacher proposals DOMAIN_DATA_PLAN relies on for pre-labelling,
        # and it had been dead code since 2026-07-22.
        #
        # Placed here, last before the page.image and page.metadata release below, because
        # sidecar_export_document's docstring requires the raster to still be resident.
        #
        # Honest about what that buys: the parquet tables need no raster and will be complete
        # either way. PAGE-IMAGE persistence and phash are best-effort - the raster is
        # already gone for pages extracted in a worker (pipeline.py:1027) or materialised
        # from a stub (:1883), and the exporter falls back to copying page.storage_path.
        # So expect complete tables and partial page images, not a guarantee of both.
        #
        # Gated by the same OCULI_SIDECAR_DISABLED switch (checked inside), and documented
        # as "Never raises" - but wrapped anyway, because a training-data export must never
        # be the reason a production extraction fails.
        try:
            sidecar_export_document(
                document,
                post["page_results"],
                runtime_params_by_page=post.get("runtime_params_by_page") or {},
            )
        except Exception:
            logging.debug("Sidecar graph export failed for %s",
                          document.document_id, exc_info=True)

        meta = getattr(document, "metadata", None) or {}
        # Build stripped extraction entry — only resolved results + minimal metadata.
        # rect_data is kept ONLY for skipped/error pages (needed for HITL overlay).
        stripped_pages = []
        skipped_rect_data = {}
        for p in doc_pages:
            stripped_pages.append({
                "page_id": p["page_id"],
                "page_number": p["page_number"],
                "source_path": p["source_path"],
                "status": p["status"],
                "errors": p["errors"] or [],
            })
            if p["status"] in {PAGE_SKIPPED, PAGE_PARTIAL} or (p.get("errors") and len(p["errors"]) > 0):
                skipped_rect_data[p["page_id"]] = p.get("rect_data")

        doc_entry = {
            # Original, human-readable identity at the TOP. `document_id` (the
            # internal batch|path hash) is kept below for the queue/resume loader.
            "folder_id": meta.get("relative_folder") or "",
            "original_document_id": meta.get("relative_path") or "",
            "document_id": document.document_id,
            "relative_path": meta.get("relative_path") or "",
            "relative_folder": meta.get("relative_folder") or "",
            "aggregated_results": post["aggregated_results"],
            "page_statuses": stripped_pages,
            "skipped_pages_rect_data": skipped_rect_data,
            "page_count": len(doc_pages),
            "completed_utc": utc_now(),
        }
        extraction_docs.append(doc_entry)

        # Persist the FULL result entry, not just a completion marker. The
        # resume loader seeds extraction_docs from these files — a marker
        # without aggregated_results means every document completed before
        # a crash silently loses its resolved data in the resumed run's
        # output, and a dead run leaves nothing viewable at all.
        #
        # Failure here must not end the batch. 1.3.2 §9 established that for
        # post-processing but it was never applied to persistence, so one
        # unwritable filename raised straight out of run_orchestrate and killed
        # a run that had been going for days. The document itself is NOT lost:
        # it is already in extraction_docs above, so this run still reports and
        # exports it — what is lost is its crash-recovery copy, which only
        # matters if the run later dies and is resumed.
        try:
            _persist_completed_document(session_dir, doc_entry)
        except Exception:
            logging.exception(
                "Persisting completed document failed for %s; continuing. "
                "This document will be missing from a resumed run.",
                document.document_id,
            )

        # Write this document's digitised file NOW rather than at
        # end-of-run, so the Export tab can show partial results even if
        # the process dies or the app closes mid-run.
        try:
            _write_digitised_fallback(session_dir, [doc_entry])
        except Exception:
            logging.warning("Incremental digitised write failed for %s", document.document_id, exc_info=True)

        # Rich digitised document (classified regions + text) → the file the
        # Tab-4 "Export Digitised" button prefers ({doc}.export_digitised.json).
        # Without this, the export falls back to the lightweight HITL file
        # (page statuses + overlay rects only), which is NOT the digitised
        # content. Keyed by document_id so it survives across resume attempts.
        digitised_document = post.get("digitised_document")
        if digitised_document:
            try:
                _dig_dir = session_dir / "digitised"
                _dig_dir.mkdir(parents=True, exist_ok=True)
                # Gzipped to match completed_documents: an LLM input dominated by
                # per-line geometry, which compresses heavily.
                import gzip as _gz
                with _gz.open(
                    _dig_dir / f"{document.document_id}.export_digitised.json.gz",
                    "wt", encoding="utf-8",
                ) as _f:
                    json.dump(_json_safe(digitised_document), _f, ensure_ascii=False)
            except Exception:
                logging.warning("export_digitised write failed for %s", document.document_id, exc_info=True)

        # Release raw extraction data immediately
        doc_pages.clear()

        for page in document.pages:
            page.image = None
            if hasattr(page, 'metadata') and isinstance(page.metadata, dict):
                page.metadata.clear()

        gc.collect()

    try:
        for document in prefetcher:
            # Graceful cancel: stop between documents. Everything persisted so
            # far stays on disk; the same run_id resumes from this point.
            if (session_dir / "cancel.request").exists():
                logging.info("cancel.request detected; stopping extraction after %d documents (resumable)", len(extraction_docs))
                append_event(session_dir, {
                    "type": "extract_cancelled",
                    "run_id": run_id,
                    "documents_completed": len(extraction_docs),
                    "documents_total": batch.document_count,
                    "phase": "extract",
                    "ts": utc_now(),
                })
                break

            # Heartbeat: one event per document with process memory so silent
            # process deaths leave a diagnosable trail.
            try:
                import psutil as _psutil
                _rss_mb = int(_psutil.Process().memory_info().rss / (1024 * 1024))
            except Exception:
                _rss_mb = -1
            append_event(session_dir, {
                "type": "doc_started",
                "run_id": run_id,
                "batch_id": batch.batch_id,
                "document_id": document.document_id,
                "relative_path": str((document.metadata or {}).get("relative_path") or ""),
                "documents_completed": len(extraction_docs),
                "documents_total": batch.document_count,
                "rss_mb": _rss_mb,
                "phase": "extract",
                "ts": utc_now(),
            })

            # Health checks assume an idle pool between documents, which no
            # longer holds when pipelining overlaps extraction with
            # post-processing. Overlap is only on for CPU hosts (no CUDA), so
            # nothing is lost by skipping the GPU probe there.
            # Paddle isolation: the worker pool holds only torch now (paddle runs in
            # its own subprocess), so the paddle-CUDA health probe checks the wrong
            # process and would import paddle needlessly. Skip it when isolated.
            _paddle_isolated = str(os.environ.get("OCULI_PADDLE_SUBPROCESS", "")).strip().lower() in ("1", "true", "yes", "on")
            if extract_runner is None and not first_document and not _paddle_isolated:
                # A periodic sanity check, not the recovery mechanism. Recovery is
                # reactive and free: a worker that dies raises BrokenProcessPool and
                # ResilientProcessPool replaces itself, and a GPU that goes sick
                # without killing anything is caught by _note_page_failure after
                # three consecutive page failures. This probe exists only to notice a
                # stack that has quietly stopped being usable, so it runs
                # _GPU_PROBES_PER_RUN times spread across the run rather than before
                # every document -- the latter cost ~5.7s each and bought nothing the
                # two reactive paths do not already cover.
                #
                # It also no longer latches off. It previously disabled itself for the
                # whole run after two consecutive failures, logging "GPU confirmed
                # unusable" -- wording inherited from a period when the probe itself
                # was faulty and condemned healthy hardware.
                documents_since_gpu_probe += 1
                if documents_since_gpu_probe >= gpu_health_probe_interval:
                    documents_since_gpu_probe = 0
                    if not gpu_health_check():
                        consecutive_gpu_health_failures += 1
                        # Recycle on every failure: the document about to run should
                        # never inherit a pool that just failed its check.
                        logging.warning(
                            "GPU health check failed before document %s (failure #%d); "
                            "recycling worker pool",
                            document.document_id,
                            consecutive_gpu_health_failures,
                        )
                        try:
                            executor.shutdown(wait=True)
                            executor = ResilientProcessPool(
                                max_workers=_ocr_workers, initializer=_init_worker
                            )
                        except Exception:
                            # Keep the existing pool rather than ending the run; it may
                            # still be serviceable, and the next probe will try again.
                            logging.exception(
                                "Could not recycle the worker pool after a failed GPU health "
                                "check; continuing with the existing pool",
                            )
                    else:
                        if consecutive_gpu_health_failures:
                            logging.info(
                                "GPU health check passed again after %d consecutive failure(s)",
                                consecutive_gpu_health_failures,
                            )
                        consecutive_gpu_health_failures = 0
            first_document = False

            # Persist the run start time on the first actually-processed document
            # so resumed runs keep the original timestamp for ETA/rate calculations.
            if not already_done_ids and processed_pages == 0:
                _persist_run_start_time(session_dir, started_all)

            _doc_meta = getattr(document, "metadata", None) or {}
            _evt_rel_path = str(_doc_meta.get("relative_path") or "")
            _evt_rel_folder = str(_doc_meta.get("relative_folder") or "")
            for page in document.pages:
                # append_event opens a file and writes with no handling of its own, so
                # a transient lock or a full disk here would end the run. Progress
                # reporting must never be able to do that; the same guard is already
                # applied to the heartbeat above and the failure events below.
                try:
                    append_event(
                        session_dir,
                        {
                            "type": "page_started",
                            "run_id": run_id,
                            "batch_id": batch.batch_id,
                            "document_id": document.document_id,
                            "relative_path": _evt_rel_path,
                            "relative_folder": _evt_rel_folder,
                            "page_id": page.page_id,
                            "phase": "extract",
                            "stage": PAGE_EXTRACTING,
                            "pct": round((processed_pages / max(1, total_pages)) * 100, 1),
                            "ts": utc_now(),
                        },
                    )
                except Exception:
                    logging.warning(
                        "Could not write page_started event for %s", page.page_id, exc_info=True
                    )
            started = time.time()
            if resolver_pool is not None:
                # Dedicated resolver worker: extraction runs here (driving the
                # OCR pool), post-processing of this document runs in its own
                # process while the NEXT document extracts. Raw results are
                # kept alongside the future so a worker failure can fall back
                # to inline post-processing without losing the document.
                try:
                    results_by_page_id = extract_document_pages_only(document, shared_executor=executor)
                except Exception as _exc:
                    _finish_failed_document(document, started, "Extraction failed", _exc)
                    continue
                _strip_heavy_debug(results_by_page_id)
                try:
                    post_future = resolver_pool.submit(
                        post_process_document,
                        _post_doc_meta(document),
                        _post_page_stubs(document),
                        results_by_page_id,
                    )
                except Exception:
                    # The resolver worker is already dead, so submit raised rather than
                    # the future. _complete_pending_post recycles the pool when the
                    # FUTURE fails, but nothing covered this side of it. Post-process
                    # inline instead: the extraction above succeeded and must not be
                    # thrown away.
                    logging.exception(
                        "Resolver pool rejected %s; post-processing it inline",
                        document.document_id,
                    )
                    try:
                        resolver_pool.shutdown(wait=False)
                        resolver_pool = ProcessPoolExecutor(
                            max_workers=1, initializer=init_post_worker
                        )
                    except Exception:
                        logging.exception("Could not recycle the resolver pool; continuing")
                    _finish_document(document, results_by_page_id, started)
                    continue
                if pending is not None:
                    p_doc, p_future, p_started, p_results = pending
                    pending = None
                    _complete_pending_post(p_doc, p_future, p_started, p_results)
                pending = (document, post_future, started, results_by_page_id)
            elif extract_runner is not None:
                # Pipelined: submit this document's extraction to the worker
                # pool, then post-process the PREVIOUS document while it runs.
                try:
                    future = extract_runner.submit(
                        extract_document_pages_only, document, shared_executor=executor
                    )
                except Exception as _exc:
                    _finish_failed_document(document, started, "Extraction could not be submitted", _exc)
                    continue
                if pending is not None:
                    p_doc, p_future, p_started = pending
                    pending = None
                    # p_future.result() re-raises whatever extraction raised in the
                    # worker. The resolver-worker branch above guards its equivalent;
                    # this one did not, so a single failed document ended the run in
                    # pipelined mode only.
                    try:
                        _p_results = p_future.result()
                    except Exception as _exc:
                        _finish_failed_document(p_doc, p_started, "Extraction failed", _exc)
                    else:
                        _finish_document(p_doc, _p_results, p_started)
                pending = (document, future, started)
            else:
                try:
                    results_by_page_id = extract_document_pages_only(document, shared_executor=executor)
                except Exception as _exc:
                    _finish_failed_document(document, started, "Extraction failed", _exc)
                    continue
                _finish_document(document, results_by_page_id, started)

        # Drain the pipelined tail — also runs when cancel broke the loop, so
        # an already-extracted document is persisted instead of discarded.
        if pending is not None:
            if resolver_pool is not None:
                p_doc, p_future, p_started, p_results = pending
                pending = None
                _complete_pending_post(p_doc, p_future, p_started, p_results)
            else:
                p_doc, p_future, p_started = pending
                pending = None
                # Same guard as the in-loop drain: the tail must not be the one place
                # a failed document can still end the run.
                try:
                    _p_results = p_future.result()
                except Exception as _exc:
                    _finish_failed_document(p_doc, p_started, "Extraction failed", _exc)
                else:
                    _finish_document(p_doc, _p_results, p_started)
    finally:
        prefetcher.close()
        if extract_runner is not None:
            extract_runner.shutdown(wait=False)
        if resolver_pool is not None:
            resolver_pool.shutdown(wait=False)
        executor.shutdown(wait=True)
        cleanup_batch_temp_dirs(batch)
        try:
            _cache_dir = session_dir / "completed_documents"
            if _cache_dir.exists():
                _cache_bytes = sum(f.stat().st_size for f in _cache_dir.iterdir() if f.is_file())
                logging.info(
                    "Result cache: %d documents, %.1f MB (gzip) in %s",
                    len(list(_cache_dir.iterdir())), _cache_bytes / (1024 * 1024), _cache_dir,
                )
        except Exception:
            pass

    return {"batch": batch, "documents": extraction_docs}


def _write_digitised_fallback(
    session_dir: Path,
    extraction_docs: List[Dict[str, Any]],
    issue_page_ids: Optional[Dict[str, str]] = None,  # {doc_id: issue_page_id}
) -> None:
    digitised_dir = session_dir / "digitised"
    digitised_dir.mkdir(parents=True, exist_ok=True)
    for doc in extraction_docs:
        doc_id = doc.get("document_id", "unknown_doc")
        pages = []
        issue_pid = (issue_page_ids or {}).get(doc_id, "")
        # NEW FORMAT: page_statuses + skipped_pages_rect_data
        # LEGACY FORMAT: pages with rect_data
        page_statuses = doc.get("page_statuses", doc.get("pages", []))
        skipped_rect_data = doc.get("skipped_pages_rect_data", {})
        for page in page_statuses:
            pid = page.get("page_id", "")
            # rect_data from skipped_pages_rect_data (new) or pages[].rect_data (legacy)
            rect_data = skipped_rect_data.get(pid) or page.get("rect_data")
            rects = ((rect_data or {}).get("rectangles") or []) if isinstance(rect_data, dict) else []
            overlay_rects: List[List[float]] = []
            for rect in rects:
                bbox = rect.get("bbox") if isinstance(rect, dict) else None
                if isinstance(bbox, list) and len(bbox) >= 4:
                    overlay_rects.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
            pages.append({
                "page_id": pid,
                "page_number": page.get("page_number"),
                "source_path": page.get("source_path", ""),
                "source_page_index": page.get("source_page_index", 0),
                "status": page.get("status", ""),
                "errors": page.get("errors"),
                "overlay_rects": overlay_rects,
                "is_issue_page": (pid == issue_pid) if issue_pid else False,
            })
        payload = {
            "run_id": session_dir.name,
            "document_id": doc_id,
            "relative_path": doc.get("relative_path") or "",
            "relative_folder": doc.get("relative_folder") or "",
            "pages": pages,
            "created_utc": utc_now(),
        }
        write_json(digitised_dir / f"{doc_id}.digitised_document.json", payload)

    # NOTE: extraction_cache.json.gz removed - the output folder's digitised_document.json
    # (written by pipeline.py during extraction) contains the full schema for re-querying.


def _resolve_from_memory(
    extraction_docs: List[Dict[str, Any]],
    batch_id: str,
    query_package: Dict[str, Any],
) -> Dict[str, Any]:
    active_structured = _active_structured_field_names(query_package)
    batch_result = BatchExtractionResult(batch_id=batch_id, status="PROCESSING")
    target_doc_ids = {
        str(did)
        for did in (query_package.get("document_ids", []) or [])
        if isinstance(did, str) and did.strip()
    }
    for doc in extraction_docs:
        doc_id = doc.get("document_id", "unknown_doc")
        if target_doc_ids and doc_id not in target_doc_ids:
            continue
        doc_result = DocumentExtractionResult(
            document_id=doc_id,
            batch_id=batch_id,
            status="PROCESSING",
            # Persisted completed documents carry the real name; cross-reference needs it.
            source_filename=os.path.basename(
                str(doc.get("relative_path") or doc.get("original_document_id") or "")
            ),
        )

        # NEW FORMAT: pre-resolved data (aggregated_results stored during extraction)
        if doc.get("aggregated_results"):
            doc_result.aggregated_results = doc["aggregated_results"]
            # Reconstruct minimal page_results from page_statuses for export row building
            for ps in doc.get("page_statuses", []):
                page_id = ps.get("page_id", "unknown_page")
                doc_result.page_results[page_id] = PageExtractionResult(
                    page_id=page_id,
                    document_id=doc_id,
                    batch_id=batch_id,
                    status=ps.get("status", "EXTRACTED"),
                    errors=ps.get("errors") or [],
                )
            doc_result.status = "COMPLETE"
            batch_result.document_results[doc_id] = doc_result
            continue

        # LEGACY FORMAT: raw pages with rect_data/cluster_results/mask_info/region_results
        for p in doc.get("pages", []):
            page_id = p.get("page_id", "unknown_page")
            width = int(p.get("width") or 1)
            height = int(p.get("height") or 1)
            page_number = int(p.get("page_number") or 1)
            synthetic_page = PageData(
                page_id=page_id,
                document_id=doc_id,
                batch_id=batch_id,
                page_number=page_number,
                image=np.zeros((height, width, 3), dtype=np.uint8),
                width=width,
                height=height,
                storage_path="",
                patches=0,
                status="READY_FOR_EXTRACTION",
            )
            page_result = PageExtractionResult(
                page_id=page_id,
                document_id=doc_id,
                batch_id=batch_id,
                rect_data=p.get("rect_data"),
                cluster_results=p.get("cluster_results"),
                mask_info=p.get("mask_info"),
                region_results=p.get("region_results"),
                status=p.get("status", "EXTRACTED"),
                errors=p.get("errors") or [],
                extraction_timing=p.get("extraction_timing"),
            )
            if page_result.status.upper() == "EXTRACTED":
                page_result = process_page_data_cleaning(
                    synthetic_page,
                    page_result,
                    active_query_field_names=active_structured,
                )
                if page_result.status in {"CLEANED", "COMPLETE"}:
                    page_result = process_page_relationship_resolution(synthetic_page, page_result)
            doc_result.page_results[page_id] = page_result
        doc_result.aggregated_results = aggregate_document_results(doc_result)
        doc_result.status = "COMPLETE"
        batch_result.document_results[doc_id] = doc_result

    # Only run validation if at least one validation task is selected
    task_ids = set(query_package.get("task_ids", []) or [])
    validation_tasks = {"validate_revision", "check_missing_data", "cross_reference_validation"}
    should_validate = bool(task_ids & validation_tasks)
    
    if should_validate:
        try:
            validator = InternalValidator(batch_result)
            existing_cross_reference = _CROSS_REFERENCE_MEMORY.get(batch_id)
            validation = validator.validate(existing_cross_reference_results=existing_cross_reference)
            cross_ref = validation.get("cross_reference_results") if isinstance(validation, dict) else None
            if isinstance(cross_ref, dict):
                _CROSS_REFERENCE_MEMORY[batch_id] = cross_ref
            batch_result.validation_results = validation if isinstance(validation, dict) else {}
        except Exception as exc:
            batch_result.validation_results = {"validation_status": "failed", "error": str(exc), "maiden_id": batch_id}
    else:
        # No validation tasks selected - skip validation entirely
        batch_result.validation_results = {
            "maiden_id": batch_id,
            "validation_status": "skipped",
            "batch_missing_data": [],
            "schema_discrepancies": [],
            "cross_reference_results": {
                "manifest": [],
                "extracted_references": [],
                "resolution_records": [],
                "anomalies": [],
                "summary": {
                    "missing_documents_count": 0,
                    "orphaned_documents_count": 0,
                    "low_confidence_count": 0,
                    "pass_state": True,
                    "hitl_tooltip": "",
                },
            },
            "validation_summary": {
                "missing_data_candidates": 0,
                "schema_discrepancies_count": 0,
                "cross_reference_missing_documents_count": 0,
                "cross_reference_orphaned_documents_count": 0,
                "cross_reference_low_confidence_count": 0,
                "validation_status": "skipped",
            },
        }

    # Extract cross-reference summary (already filtered by task_ids during validation)
    cross_ref_summary = {}
    if isinstance(batch_result.validation_results, dict):
        cross_ref_summary = (
            (batch_result.validation_results.get("cross_reference_results") or {}).get("summary", {})
            if isinstance(batch_result.validation_results.get("cross_reference_results"), dict)
            else {}
        )
    cross_ref_missing = int(cross_ref_summary.get("missing_documents_count", 0) or 0)
    cross_ref_orphans = int(cross_ref_summary.get("orphaned_documents_count", 0) or 0)
    cross_ref_low_conf = int(cross_ref_summary.get("low_confidence_count", 0) or 0)
    has_cross_ref_issues = (cross_ref_missing + cross_ref_orphans + cross_ref_low_conf) > 0
    cross_ref_tooltip = str(cross_ref_summary.get("hitl_tooltip") or "").strip()

    _doc_paths = {str(d.get("document_id")): d for d in extraction_docs if d.get("document_id")}

    export_rows: List[Dict[str, Any]] = []
    digitised_documents: Dict[str, Any] = {}
    for doc_id, doc_result in batch_result.document_results.items():
        issues: List[str] = []
        status_counts: Dict[str, int] = {}
        preview_raster = ""
        for pr in doc_result.page_results.values():
            raw = str(pr.status or "UNKNOWN").upper()
            if raw in {"COMPLETE", "CLEANED"}:
                key = PAGE_CLEANED
            elif raw == "EXTRACTED":
                key = PAGE_EXTRACTED
            elif raw in {"SKIPPED", "ERROR"}:
                key = PAGE_SKIPPED
            elif raw == "PARTIAL":
                key = PAGE_PARTIAL
            else:
                key = raw.lower()
            status_counts[key] = status_counts.get(key, 0) + 1
        
        # Collect all issue pages (skipped/partial + low-confidence/missing fields)
        agg = batch_result.document_results[doc_id].aggregated_results
        all_issue_pages = _collect_all_hitl_issue_pages(doc_result, agg, doc_id, extraction_docs)

        # Use first issue page as the primary page for backward compatibility
        first_issue = all_issue_pages[0] if all_issue_pages else {}
        issue_page_id = first_issue.get("page_id", "")
        issue_page_number = first_issue.get("page_number")
        preview_raster = first_issue.get("source_path", "")
        issue_overlay_rects = first_issue.get("overlay_rects", [])

        # Build issues list
        has_skipped = any(k in {PAGE_SKIPPED, PAGE_PARTIAL} for k in status_counts.keys())
        if has_skipped:
            issues.append("Extraction has skipped/partial pages")
        if has_cross_ref_issues:
            issues.append(cross_ref_tooltip or "Cross-reference issues detected in this batch.")

        # Collect unique low-confidence and missing field names across all issue pages
        low_conf_fields: List[str] = []
        missing_fields: List[str] = []
        for ip in all_issue_pages:
            for reason in ip.get("issue_reasons", []):
                if reason.startswith("low_confidence:"):
                    field = reason[len("low_confidence:"):]
                    if field and field not in low_conf_fields:
                        low_conf_fields.append(field)
                elif reason.startswith("missing_value:"):
                    field = reason[len("missing_value:"):]
                    if field and field not in missing_fields:
                        missing_fields.append(field)
        if low_conf_fields:
            issues.append(f"Low confidence: {', '.join(low_conf_fields)}")
        if missing_fields:
            issues.append(f"Missing values: {', '.join(missing_fields)}")

        has_field_issues = bool(low_conf_fields or missing_fields)
        hitl_state = PAGE_HITL_PENDING if (has_skipped or has_cross_ref_issues or has_field_issues) else "none"
        is_complete = is_export_complete_only(status_counts, hitl_state=hitl_state, human_verified=False)
        query_values = _extract_query_values_for_export(agg, active_structured)
        _src = _doc_paths.get(str(doc_id)) or {}
        _rf = str(_src.get("relative_folder") or "").replace("\\", "/")
        _rp = str(_src.get("relative_path") or "").replace("\\", "/")
        export_rows.append(
            {
                "source_batch": batch_id,
                "document_id": doc_id,
                "relative_folder": _rf,
                "relative_path": _rp,
                "status": "partial" if issues else "complete",
                "issues": "; ".join(issues),
                "page_status_counts": status_counts,
                "is_export_complete_only": is_complete,
                "preview_raster_path": preview_raster,
                "issue_page_id": issue_page_id,
                "issue_page_number": issue_page_number,
                "issue_source_path": preview_raster,
                "issue_source_page_index": (issue_page_number - 1) if isinstance(issue_page_number, int) else 0,
                "issue_overlay_rects": issue_overlay_rects,
                "issue_pages": all_issue_pages,
                "hitl_state": hitl_state,
                "human_verified": False,
                "query_fields": [f.get("field_name") for f in query_package.get("structured_fields", []) if isinstance(f, dict)],
                "query_values": query_values,
                "task_ids": query_package.get("task_ids", []),
            }
        )
        # NOTE: digitised_document.json is already written by pipeline.py during extraction
        # to outputs/{batch}/{doc_id}/digitised_document.json. No need to rebuild here.
        digitised_documents[doc_id] = {}

    return {
        "schema_version": 1,
        "batch_id": batch_id,
        "query_package": query_package,
        "resolved_utc": utc_now(),
        "export_rows": export_rows,
        "document_results": {k: v.aggregated_results for k, v in batch_result.document_results.items()},
        "digitised_documents": digitised_documents,
        "validation_results": batch_result.validation_results,
        "original_task_ids": query_package.get("task_ids", []),
    }


def _freetext_query_metrics(query_package: Dict[str, Any]) -> Dict[str, Any]:
    freetext = [q for q in query_package.get("freetext_queries", []) if isinstance(q, dict)]
    return {
        "freetext_count": len(freetext),
        "freetext_ids": [q.get("id") for q in freetext],
        "match_modes": sorted({q.get("match_mode", "fuzzy") for q in freetext}),
    }


def run_extract(args: argparse.Namespace) -> int:
    ensure_dirs()
    run_id = args.run_id or str(uuid.uuid4())
    session_dir = SESSIONS_ROOT / run_id
    session_dir.mkdir(parents=True, exist_ok=True)
    write_session_fingerprints(
        session_dir,
        input_path=args.input_path,
        module_a_enabled=args.module_a_enabled,
        page_timeout_sec=args.page_timeout_sec,
    )
    batch_name = Path(args.input_path).name
    upsert_activity_row(run_id, batch_name, "processing", session_dir)
    set_activity_input_path(run_id, args.input_path)

    session_meta = {
        "run_id": run_id,
        "phase": "extract",
        "input_path": args.input_path,
        "source_folder_name": batch_name,
        "source_file_hash": _session_source_hash(session_dir),
        "started_utc": utc_now(),
        "last_activity_utc": utc_now(),
        "exit_code": 0,
    }
    _write_session_meta(session_dir / "session_meta.json", session_meta)
    append_event(session_dir, {"type": "run_started", "run_id": run_id, "phase": "extract", "ts": utc_now()})

    try:
        extracted = _extract_in_memory(session_dir, run_id, args.input_path)
        batch = extracted["batch"]
        _write_digitised_fallback(session_dir, extracted["documents"])
        session_meta["notes"] = _skipped_pages_notes(extracted["documents"])
        session_meta["phase"] = "extract_complete"
        session_meta["last_extract_utc"] = utc_now()
        _write_session_meta(session_dir / "session_meta.json", session_meta)
        append_event(session_dir, {"type": "extract_complete", "run_id": run_id, "batch_id": batch.batch_id, "phase": "extract", "ts": utc_now()})
        upsert_activity_row(run_id, batch_name, "complete", session_dir)
        print(run_id)
        return 0
    except Exception as exc:
        session_meta["exit_code"] = 1
        session_meta["last_error"] = str(exc)
        _write_session_meta(session_dir / "session_meta.json", session_meta)
        emit_run_failed(
            session_dir,
            reason="extract_exception",
            message=str(exc),
            phase="extract",
            exit_code=1,
        )
        upsert_activity_row(run_id, batch_name, "failed", session_dir)
        logging.exception("extract failed")
        return 1


def run_resolve(args: argparse.Namespace) -> int:
    ensure_dirs()
    session_dir = SESSIONS_ROOT / args.run_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Unknown run_id: {args.run_id}")
    session_meta_path = session_dir / "session_meta.json"
    session_meta = json.loads(session_meta_path.read_text(encoding="utf-8")) if session_meta_path.exists() else {}

    try:
        payload = parse_query_package(Path(args.query_package))
        write_json(session_dir / "query_package.json", payload)
        resolution_checkpoint = {
            "schema_version": 1,
            "run_id": args.run_id,
            "phase": "resolve",
            "query_package": payload,
            "document_count": 0,
            "created_utc": utc_now(),
        }
        write_json(session_dir / "resolution_checkpoint.json", resolution_checkpoint)
        append_event(
            session_dir,
            {
                "type": "resolve_complete",
                "run_id": args.run_id,
                "phase": "resolve",
                "message": "Resolution checkpoint updated",
                "ts": utc_now(),
            },
        )
        session_meta["phase"] = "resolve_complete"
        session_meta["exit_code"] = 0
        session_meta["last_activity_utc"] = utc_now()
        _write_session_meta(session_meta_path, session_meta)
        return 0
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("schema_version_mismatch:"):
            got = msg.split(":", 1)[1]
            emit_run_failed(
                session_dir,
                reason="schema_version_mismatch",
                message="Query package version not supported.",
                phase="resolve",
                exit_code=2,
                expected_schema_version=EXPECTED_QUERY_SCHEMA_VERSION,
                got=got,
            )
            session_meta["phase"] = "resolve_failed"
            session_meta["exit_code"] = 2
            session_meta["last_error"] = "schema_version_mismatch"
            _write_session_meta(session_meta_path, session_meta)
            return 2
        if msg.startswith("unknown_structured_fields:") or msg.startswith("unknown_task_ids:"):
            emit_run_failed(
                session_dir,
                reason="query_validation_error",
                message=msg,
                phase="resolve",
                exit_code=2,
            )
            session_meta["phase"] = "resolve_failed"
            session_meta["exit_code"] = 2
            session_meta["last_error"] = msg
            _write_session_meta(session_meta_path, session_meta)
            return 2
        raise
    except Exception as exc:
        emit_run_failed(
            session_dir,
            reason="resolve_exception",
            message=str(exc),
            phase="resolve",
            exit_code=1,
        )
        session_meta["phase"] = "resolve_failed"
        session_meta["exit_code"] = 1
        session_meta["last_error"] = str(exc)
        _write_session_meta(session_meta_path, session_meta)
        logging.exception("resolve failed")
        return 1


def _try_resolve_from_persisted_session(
    session_dir: Path, run_id: str, query_package_path: str
) -> Optional[int]:
    """Rebuild resolved_results.json from the session's own persisted results.

    The server/app path persists each document's FULL results (aggregated
    fields included) to completed_documents/*.json.gz during extraction. The
    legacy recovery commands only knew about the CLI path's outputs/ tree, so
    a crashed server run could never show its partial results even though
    everything needed was on disk. This resolves through the same
    _resolve_from_memory machinery the live hibernate loop uses, so the output
    schema (export_rows etc.) is identical to a normal resolve.

    Returns 0 on success, None when this session has no persisted results
    (caller falls back to the legacy outputs/ path).
    """
    extraction_docs = [
        d for d in _load_persisted_extraction_docs(session_dir)
        if d.get("aggregated_results")
    ]
    if not extraction_docs:
        return None

    batch_id = _read_batch_id(session_dir, run_id)
    payload: Dict[str, Any] = {}
    if query_package_path:
        try:
            payload = parse_query_package(Path(query_package_path))
            write_json(session_dir / "query_package.json", payload)
        except Exception:
            logging.warning("Query package unreadable; resolving all fields from persisted results")
            payload = {}

    resolved = _resolve_from_memory(extraction_docs, batch_id, payload)
    resolved.pop("digitised_documents", None)

    completed_ids = _load_completed_document_ids(session_dir)
    total_expected = _read_expected_document_count(session_dir)
    is_partial = total_expected is not None and len(completed_ids) < total_expected
    resolved["is_partial"] = is_partial
    resolved["completed_document_count"] = len(completed_ids)
    if total_expected is not None:
        resolved["expected_document_count"] = total_expected
    write_json(session_dir / "resolved_results.json", resolved)

    append_event(
        session_dir,
        {
            "type": "resolve_complete",
            "run_id": run_id,
            "phase": "resolve",
            "batch_id": batch_id,
            "is_partial": is_partial,
            "completed_document_count": len(completed_ids),
            "resolved_results_path": str(session_dir / "resolved_results.json"),
            "message": (
                f"Resolved {len(completed_ids)} of {total_expected} documents from persisted session results (partial)"
                if is_partial
                else f"Resolved {len(completed_ids)} documents from persisted session results"
            ),
            "ts": utc_now(),
        },
    )
    session_meta_path = session_dir / "session_meta.json"
    session_meta = json.loads(session_meta_path.read_text(encoding="utf-8")) if session_meta_path.exists() else {}
    session_meta["phase"] = "resolve_complete"
    session_meta["exit_code"] = 0
    session_meta["last_activity_utc"] = utc_now()
    _write_session_meta(session_meta_path, session_meta)
    logging.info(
        "Resolved %d persisted documents for run %s (partial=%s)",
        len(extraction_docs), run_id, is_partial,
    )
    return 0


def run_resolve_from_disk(args: argparse.Namespace) -> int:
    """Load existing digitised documents from outputs/ and optionally filter by new query package.
    
    The digitised documents in outputs/{batch}/{doc}/digitised_document.json contain the
    normalized, formatted document layout (TABLE, INFO_CLUSTER, LIST, PARAGRAPH, etc.).
    This command simply loads them and optionally filters by document_ids in the new query package.
    
    No re-extraction or re-resolution needed - the digitised documents are the final output.
    """
    ensure_dirs()
    session_dir = SESSIONS_ROOT / args.run_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Unknown run_id: {args.run_id}")

    # Server/app path first: full per-document results persisted during
    # extraction live in the session itself. The outputs/ tree below only
    # exists for CLI (pipeline_main) runs.
    try:
        rc = _try_resolve_from_persisted_session(
            session_dir, args.run_id, str(getattr(args, "query_package", "") or "")
        )
        if rc == 0:
            return 0
    except Exception:
        logging.exception("Persisted-session resolve failed; trying outputs/ fallback")

    # Get batch_id from existing resolved_results.json, or use run_id as fallback
    batch_id = args.run_id
    resolved_path = session_dir / "resolved_results.json"
    if resolved_path.exists():
        try:
            existing = json.loads(resolved_path.read_text(encoding="utf-8"))
            batch_id = existing.get("batch_id", args.run_id)
        except Exception:
            pass

    # Load digitised documents from outputs/ where pipeline.py writes the full schema
    outputs_digitised_dir = Path("outputs") / batch_id
    if not outputs_digitised_dir.exists():
        logging.error("No output documents found at %s. Re-extraction required.", outputs_digitised_dir)
        return 1
    
    digitised_files = (list(outputs_digitised_dir.glob("*/digitised_document.json.gz"))
                       or list(outputs_digitised_dir.glob("*/digitised_document.json")))
    if not digitised_files:
        logging.error("No digitised documents found in outputs. Re-extraction required.")
        return 1
    
    digitised_documents = {}
    for file_path in digitised_files:
        try:
            doc_id = file_path.parent.name
            import gzip as _gz
            _open = _gz.open if file_path.suffix == ".gz" else open
            with _open(file_path, "rt", encoding="utf-8") as _f:
                digitised_documents[doc_id] = json.load(_f)
        except Exception:
            logging.warning("Failed to load digitised document %s; skipping", file_path)
    
    if not digitised_documents:
        logging.error("No valid digitised documents loaded. Re-extraction required.")
        return 1
    
    # Load query package to get document_ids filter (if specified)
    try:
        payload = parse_query_package(Path(args.query_package))
        write_json(session_dir / "query_package.json", payload)
        target_doc_ids = {
            str(did)
            for did in (payload.get("document_ids", []) or [])
            if isinstance(did, str) and did.strip()
        }
        
        # Filter digitised documents by target_doc_ids if specified
        if target_doc_ids:
            digitised_documents = {
                doc_id: doc_data
                for doc_id, doc_data in digitised_documents.items()
                if doc_id in target_doc_ids
            }
    except Exception:
        logging.warning("Failed to parse query package; returning all digitised documents")
    
    resolved_path = session_dir / "resolved_results.json"
    batch_id = args.run_id
    if resolved_path.exists():
        try:
            existing = json.loads(resolved_path.read_text(encoding="utf-8"))
            batch_id = existing.get("batch_id", args.run_id)
        except Exception:
            pass
    
    session_meta_path = session_dir / "session_meta.json"
    session_meta = json.loads(session_meta_path.read_text(encoding="utf-8")) if session_meta_path.exists() else {}
    
    try:
        # Build minimal resolved_results.json pointing to digitised documents
        resolved = {
            "batch_id": batch_id,
            "run_id": args.run_id,
            "document_count": len(digitised_documents),
            "digitised_documents": digitised_documents,
            "note": "Loaded from existing digitised documents. No re-resolution performed.",
        }
        write_json(session_dir / "resolved_results.json", resolved)
        
        append_event(
            session_dir,
            {
                "type": "resolve_complete",
                "run_id": args.run_id,
                "phase": "resolve",
                "batch_id": batch_id,
                "resolved_results_path": str(session_dir / "resolved_results.json"),
                "note": "Loaded from digitised documents",
            },
        )
        session_meta["phase"] = "resolve_complete"
        session_meta["exit_code"] = 0
        session_meta["last_activity_utc"] = utc_now()
        _write_session_meta(session_meta_path, session_meta)
        logging.info("Loaded %d digitised documents from disk", len(digitised_documents))
        return 0
    except Exception as exc:
        emit_run_failed(
            session_dir,
            reason="resolve_from_disk_exception",
            message=str(exc),
            phase="resolve",
            exit_code=1,
        )
        session_meta["phase"] = "resolve_failed"
        session_meta["exit_code"] = 1
        session_meta["last_error"] = str(exc)
        _write_session_meta(session_meta_path, session_meta)
        logging.exception("resolve_from_disk failed")
        return 1


def run_resolve_partial(args: argparse.Namespace) -> int:
    """Load whatever digitised documents have been written so far for a crashed run.
    
    Loads from outputs/{batch_id}/*/digitised_document.json where pipeline.py writes
    the full schema during extraction."""
    ensure_dirs()
    session_dir = SESSIONS_ROOT / args.run_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Unknown run_id: {args.run_id}")

    # Server/app path first: crashed runs persist full per-document results in
    # the session; use them directly (identical schema to a live resolve).
    try:
        rc = _try_resolve_from_persisted_session(
            session_dir, args.run_id, str(getattr(args, "query_package", "") or "")
        )
        if rc == 0:
            return 0
    except Exception:
        logging.exception("Persisted-session resolve failed; trying outputs/ fallback")

    batch_id = _read_batch_id(session_dir, args.run_id)
    outputs_digitised_dir = Path("outputs") / batch_id
    if not outputs_digitised_dir.exists():
        logging.error("No output documents found for run %s; nothing to resolve.", args.run_id)
        return 1
    
    digitised_files = (list(outputs_digitised_dir.glob("*/digitised_document.json.gz"))
                       or list(outputs_digitised_dir.glob("*/digitised_document.json")))
    if not digitised_files:
        logging.error("No digitised documents found for run %s; nothing to resolve.", args.run_id)
        return 1
    
    digitised_documents = {}
    for file_path in digitised_files:
        try:
            doc_id = file_path.parent.name
            import gzip as _gz
            _open = _gz.open if file_path.suffix == ".gz" else open
            with _open(file_path, "rt", encoding="utf-8") as _f:
                digitised_documents[doc_id] = json.load(_f)
        except Exception:
            logging.warning("Failed to load digitised document %s; skipping", file_path)
    
    if not digitised_documents:
        logging.error("No valid digitised documents loaded for run %s; nothing to resolve.", args.run_id)
        return 1

    completed_ids = _load_completed_document_ids(session_dir)
    total_expected = _read_expected_document_count(session_dir)

    session_meta_path = session_dir / "session_meta.json"
    session_meta = json.loads(session_meta_path.read_text(encoding="utf-8")) if session_meta_path.exists() else {}

    try:
        payload = parse_query_package(Path(args.query_package))
        write_json(session_dir / "query_package.json", payload)
        
        # Filter by document_ids if specified in query package
        target_doc_ids = {
            str(did)
            for did in (payload.get("document_ids", []) or [])
            if isinstance(did, str) and did.strip()
        }
        if target_doc_ids:
            digitised_documents = {
                doc_id: doc_data
                for doc_id, doc_data in digitised_documents.items()
                if doc_id in target_doc_ids
            }

        is_partial = total_expected is not None and len(completed_ids) < total_expected
        
        resolved = {
            "batch_id": batch_id,
            "run_id": args.run_id,
            "document_count": len(digitised_documents),
            "digitised_documents": digitised_documents,
            "is_partial": is_partial,
            "completed_document_count": len(completed_ids),
            "note": "Loaded from digitised documents (partial run recovery).",
        }
        if total_expected is not None:
            resolved["expected_document_count"] = total_expected

        write_json(session_dir / "resolved_results.json", resolved)

        append_event(
            session_dir,
            {
                "type": "resolve_complete",
                "run_id": args.run_id,
                "phase": "resolve",
                "batch_id": batch_id,
                "is_partial": is_partial,
                "completed_document_count": len(completed_ids),
                "resolved_results_path": str(session_dir / "resolved_results.json"),
                "message": (
                    f"Resolved {len(completed_ids)} of {total_expected} documents (partial)"
                    if is_partial else f"Resolved {len(completed_ids)} documents"
                ),
                "ts": utc_now(),
            },
        )
        session_meta["phase"] = "resolve_complete"
        session_meta["exit_code"] = 0
        session_meta["last_activity_utc"] = utc_now()
        _write_session_meta(session_meta_path, session_meta)
        return 0
    except Exception as exc:
        emit_run_failed(
            session_dir, reason="resolve_partial_exception", message=str(exc),
            phase="resolve", exit_code=1,
        )
        session_meta["phase"] = "resolve_failed"
        session_meta["exit_code"] = 1
        session_meta["last_error"] = str(exc)
        _write_session_meta(session_meta_path, session_meta)
        logging.exception("resolve_partial failed")
        return 1


def _ensure_is_partial_flag(session_dir: Path, is_partial: bool) -> None:
    """run_resolve_from_disk's output predates the is_partial field and doesn't
    set it. The frontend needs one consistent field to branch on regardless of
    which path produced resolved_results.json, so backfill it here rather than
    modifying run_resolve_from_disk itself (which has other callers and is
    correct as-is for its own purpose)."""
    resolved_path = session_dir / "resolved_results.json"
    if not resolved_path.exists():
        return
    try:
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if "is_partial" not in resolved:
        resolved["is_partial"] = is_partial
        write_json(resolved_path, resolved)


def run_resolve_for_view(args: argparse.Namespace) -> int:
    """Single entry point for the 'View Results' / 'Export Results' button.

    Picks the right resolve path automatically so the frontend never has to
    decide:
      - If the run finished extraction cleanly (extraction_cache.json.gz
        exists), resolve the complete results via run_resolve_from_disk.
      - Otherwise, if any documents were persisted before the run stopped
        (completed_documents/ has entries), resolve a partial result via
        run_resolve_partial.
      - Otherwise, there is nothing to show — raise.

    Either path writes the same resolved_results.json shape that export and
    preview_csv already read; this function does not change what they do.
    """
    ensure_dirs()
    session_dir = SESSIONS_ROOT / args.run_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Unknown run_id: {args.run_id}")

    cache_path = session_dir / "extraction_cache.json.gz"
    if cache_path.exists():
        logging.info(
            "Run %s: full extraction cache present, resolving complete results",
            args.run_id,
        )
        exit_code = run_resolve_from_disk(args)
        if exit_code == 0:
            _ensure_is_partial_flag(session_dir, is_partial=False)
        return exit_code

    completed_ids = _load_completed_document_ids(session_dir)
    if completed_ids:
        logging.info(
            "Run %s: no full extraction cache, but %d completed document(s) found; "
            "resolving partial results",
            args.run_id, len(completed_ids),
        )
        return run_resolve_partial(args)

    raise FileNotFoundError(
        f"No results available for run {args.run_id}: no completed extraction "
        f"cache and no completed documents persisted."
    )


def run_orchestrate(args: argparse.Namespace) -> int:
    ensure_dirs()
    run_id = args.run_id or str(uuid.uuid4())
    session_dir = SESSIONS_ROOT / run_id
    session_dir.mkdir(parents=True, exist_ok=True)
    write_session_fingerprints(
        session_dir,
        input_path=args.input_path,
        module_a_enabled=args.module_a_enabled,
        page_timeout_sec=args.page_timeout_sec,
    )
    batch_name = Path(args.input_path).name
    upsert_activity_row(run_id, batch_name, "processing", session_dir)
    set_activity_input_path(run_id, args.input_path)
    session_meta_path = session_dir / "session_meta.json"

    session_meta = {
        "run_id": run_id,
        "phase": "extract",
        "input_path": args.input_path,
        "source_folder_name": batch_name,
        "source_file_hash": _session_source_hash(session_dir),
        "started_utc": utc_now(),
        "last_activity_utc": utc_now(),
        "exit_code": 0,
        "mode": "single_process_in_memory",
    }
    _write_session_meta(session_meta_path, session_meta)
    append_event(session_dir, {"type": "run_started", "run_id": run_id, "phase": "extract", "ts": utc_now()})
    print(run_id, flush=True)

    try:
        extracted = _extract_in_memory(session_dir, run_id, args.input_path)
        batch = extracted["batch"]
        extraction_docs = extracted["documents"]
        _write_digitised_fallback(session_dir, extraction_docs)
        session_meta["notes"] = _skipped_pages_notes(extraction_docs)
        session_meta["phase"] = "hibernating_waiting_for_query_package"
        session_meta["last_extract_utc"] = utc_now()
        _write_session_meta(session_meta_path, session_meta)
        append_event(session_dir, {"type": "extract_complete", "run_id": run_id, "batch_id": batch.batch_id, "phase": "extract", "ts": utc_now()})

        query_package_path = session_dir / "query_package.json"
        last_query_mtime: float | None = None
        while True:
            if (session_dir / "cancel.request").exists():
                emit_run_failed(
                    session_dir,
                    reason="cancelled",
                    message="Run cancelled by user.",
                    phase="resolve",
                    exit_code=3,
                )
                session_meta["phase"] = "cancelled"
                session_meta["exit_code"] = 3
                _write_session_meta(session_meta_path, session_meta)
                upsert_activity_row(run_id, batch_name, "failed", session_dir)
                return 3

            if not query_package_path.exists():
                time.sleep(0.25)
                continue

            mtime = query_package_path.stat().st_mtime
            if last_query_mtime is not None and mtime <= last_query_mtime:
                time.sleep(0.25)
                continue
            last_query_mtime = mtime

            payload = parse_query_package(query_package_path)
            # region agent log
            _debug_c9d468_log(
                run_id,
                "H5",
                "session_runtime.py:run_orchestrate",
                "freetext_after_parse_before_resolve",
                {
                    "freetext_count": len(payload.get("freetext_queries", []) or []),
                    "freetext_sample": (payload.get("freetext_queries", []) or [])[:10],
                    "structured_count": len(payload.get("structured_fields", []) or []),
                },
            )
            # endregion
            resolved = _resolve_from_memory(extraction_docs, batch.batch_id, payload)
            digitised_docs_payload = resolved.pop("digitised_documents", {})
            write_json(session_dir / "resolved_results.json", resolved)
            resolved_docs = len(resolved.get("export_rows", []))
            qmetrics = _freetext_query_metrics(payload)
            append_event(
                session_dir,
                {
                    "type": "resolve_complete",
                    "run_id": run_id,
                    "phase": "resolve",
                    "batch_id": batch.batch_id,
                    "task_ids": payload.get("task_ids", []),
                    "resolved_results_path": str(session_dir / "resolved_results.json"),
                    "query_metrics": qmetrics,
                    "message": f"Resolved {resolved_docs} documents from in-memory extraction.",
                    "ts": utc_now(),
                },
            )
            session_meta["phase"] = "resolve_complete"
            session_meta["exit_code"] = 0
            session_meta["last_activity_utc"] = utc_now()
            _write_session_meta(session_meta_path, session_meta)
            upsert_activity_row(run_id, batch_name, "complete", session_dir)
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("schema_version_mismatch:"):
            got = msg.split(":", 1)[1]
            emit_run_failed(
                session_dir,
                reason="schema_version_mismatch",
                message="Query package version not supported.",
                phase="resolve",
                exit_code=2,
                expected_schema_version=EXPECTED_QUERY_SCHEMA_VERSION,
                got=got,
            )
            session_meta["phase"] = "resolve_failed"
            session_meta["exit_code"] = 2
            session_meta["last_error"] = "schema_version_mismatch"
            _write_session_meta(session_meta_path, session_meta)
            upsert_activity_row(run_id, batch_name, "failed", session_dir)
            return 2
        if msg.startswith("unknown_structured_fields:") or msg.startswith("unknown_task_ids:"):
            emit_run_failed(
                session_dir,
                reason="query_validation_error",
                message=msg,
                phase="resolve",
                exit_code=2,
            )
            session_meta["phase"] = "resolve_failed"
            session_meta["exit_code"] = 2
            session_meta["last_error"] = msg
            _write_session_meta(session_meta_path, session_meta)
            upsert_activity_row(run_id, batch_name, "failed", session_dir)
            return 2
        raise
    except Exception as exc:
        emit_run_failed(
            session_dir,
            reason="extract_exception",
            message=str(exc),
            phase=session_meta.get("phase", "extract"),
            exit_code=1,
        )
        session_meta["phase"] = "failed"
        session_meta["exit_code"] = 1
        session_meta["last_error"] = str(exc)
        _write_session_meta(session_meta_path, session_meta)
        upsert_activity_row(run_id, batch_name, "failed", session_dir)
        logging.exception("orchestrate failed")
        return 1


def run_purge(args: argparse.Namespace) -> int:
    ensure_dirs()
    days = args.days if args.days is not None else get_session_retention_days()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = load_activity_cache()
    for session_dir in SESSIONS_ROOT.glob("*"):
        if not session_dir.is_dir():
            continue
        if not args.all:
            mtime = datetime.fromtimestamp(session_dir.stat().st_mtime, timezone.utc)
            if mtime >= cutoff:
                continue
        run_id = session_dir.name
        shutil.rmtree(session_dir, ignore_errors=True)
        for row in rows:
            if row.get("run_id") == run_id:
                row["status"] = "purged"
                row["purged_utc"] = utc_now()
                row.pop("session_path", None)
    save_activity_cache(rows)
    return 0


def _internal_rows_from_payload(payload: Dict[str, Any], export_rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    validation_results = payload.get("validation_results", {}) if isinstance(payload.get("validation_results", {}), dict) else {}
    rows: List[Dict[str, Any]] = []
    
    # Build document_id -> filename mapping from export_rows
    doc_id_to_filename: Dict[str, str] = {}
    if export_rows:
        for r in export_rows:
            doc_id = str(r.get("document_id") or "")
            rel_path = str(r.get("relative_path") or "")
            if doc_id and rel_path:
                doc_id_to_filename[doc_id] = Path(rel_path).name
            elif doc_id:
                doc_id_to_filename[doc_id] = doc_id

    def _map_doc_id(doc_id: str) -> str:
        return doc_id_to_filename.get(doc_id, doc_id)

    # Cross-reference anomalies
    cross_ref = validation_results.get("cross_reference_results", {}) if isinstance(validation_results.get("cross_reference_results", {}), dict) else {}
    for r in (cross_ref.get("anomalies") or []):
        if isinstance(r, dict) and r.get("anomaly_type", "NONE") != "NONE":
            row = {"category": "cross_reference", **r}
            # Map source_document and referenced_document to filenames
            if row.get("source_document"):
                row["source_document"] = _map_doc_id(str(row["source_document"]))
            if row.get("referenced_document"):
                row["referenced_document"] = _map_doc_id(str(row["referenced_document"]))
            rows.append(row)

    # Missing data candidates
    for item in (validation_results.get("batch_missing_data") or []):
        if not isinstance(item, dict):
            continue
        phrase = str(item.get("phrase") or "")
        docs = item.get("found_in_documents") or []
        mapped_docs = [_map_doc_id(str(d)) for d in docs] if docs else []
        rows.append({
            "category": "missing_data",
            "source_document": ", ".join(mapped_docs) if mapped_docs else "",
            "referenced_document": phrase,
            "anomaly_type": "MISSING_DATA",
            "reference_type": None,
            "raw_reference_string": phrase,
            "closest_manifest_match": None,
            "page_or_section": None,
            "severity": "high" if item.get("relevance_score", 0) >= 0.7 else "medium",
            "notes": str(item.get("note") or f"Phrase '{phrase}' appears unassigned in {item.get('unique_documents', 0)} document(s) (score={item.get('relevance_score', 0):.2f})."),
        })

    # Schema discrepancies (revision/release/version)
    for item in (validation_results.get("schema_discrepancies") or []):
        if not isinstance(item, dict):
            continue
        field = str(item.get("data_definition") or "")
        values = item.get("values") or []
        sibling = str(item.get("sibling_id") or "")
        values_str = ", ".join(str(v) for v in values) if values else ""
        rows.append({
            "category": "schema_discrepancy",
            "source_document": _map_doc_id(sibling),
            "referenced_document": field,
            "anomaly_type": "SCHEMA_DISCREPANCY",
            "reference_type": None,
            "raw_reference_string": values_str,
            "closest_manifest_match": None,
            "page_or_section": None,
            "severity": str(item.get("severity") or "low"),
            "notes": str(item.get("note") or f"Field '{field}' has conflicting values across documents: {values_str}."),
        })

    return rows


def _row_batch_key(row: Dict[str, Any]) -> str:
    s = str(row.get("source_batch") or "").strip()
    return s if s else "__default__"


def _page_label_for_entry(entry: Dict[str, Any]) -> str:
    span = entry.get("page_span")
    if isinstance(span, dict) and span.get("spans_pages"):
        sp, ep = span.get("start_page"), span.get("end_page")
        try:
            sp_i = int(sp) if sp is not None else None
            ep_i = int(ep) if ep is not None else None
        except (TypeError, ValueError):
            sp_i = ep_i = None
        if sp_i is not None and ep_i is not None and sp_i != ep_i:
            return f"pg {sp_i}–{ep_i}"
        if sp_i is not None:
            return f"pg {sp_i}"
    pid = str(entry.get("page_id") or "")
    m = re.search(r"_PG(\d+)$", pid, re.I)
    if m:
        return f"pg {int(m.group(1))}"
    return pid if pid else "—"


def _field_pages_for_doc(doc_results: Dict[str, Any], doc_id: str) -> Dict[str, str]:
    agg = (doc_results or {}).get(doc_id) or {}
    entries = (agg.get("document_summary") or {}).get("entries") or []
    field_pages: Dict[str, List[str]] = {}
    for ent in (entries if isinstance(entries, list) else []):
        qn = str(ent.get("query_name") or "").strip()
        if not qn:
            continue
        pg = _page_label_for_entry(ent)
        if pg and pg != "\u2014":
            field_pages.setdefault(qn, [])
            if pg not in field_pages[qn]:
                field_pages[qn].append(pg)
    return {k: ", ".join(v) for k, v in field_pages.items()}


def _format_friendly_entry_value(entry: Dict[str, Any]) -> str:
    val = entry.get("value")
    if val is None:
        return ""
    if isinstance(val, list):
        parts: List[str] = []
        for i, item in enumerate(val[:80]):
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            rec = item.get("record")
            if isinstance(rec, dict):
                inner = "; ".join(
                    f"{k}: {v}" for k, v in rec.items() if v is not None and str(v).strip() != ""
                )
                parts.append(f"Row {item.get('row_index', i)}: {inner}".strip())
            else:
                parts.append(str(rec))
        if len(val) > 80:
            parts.append(f"… (+{len(val) - 80} more rows)")
        return " | ".join(parts) if parts else ""
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def _resolve_preview_document_ids(
    session_dir: Path,
    mode: str,
    document_ids_csv: str,
    batch_key: str,
    restrict_to_relative_folder: bool,
    relative_folder: str,
) -> tuple[List[Dict[str, Any]], set[str]]:
    prep_rows, _ = _prepare_export_rows(session_dir, mode)
    all_ids = {str(r["document_id"]) for r in prep_rows if r.get("document_id")}
    ids_csv = str(document_ids_csv or "").strip()
    bk = str(batch_key or "").strip()
    if ids_csv:
        want = {x.strip() for x in ids_csv.split(",") if x.strip()}
        return prep_rows, all_ids & want
    if bk:
        filtered = [r for r in prep_rows if _row_batch_key(r) == bk]
        if restrict_to_relative_folder:
            norm = str(relative_folder or "").replace("\\", "/")
            filtered = [
                r
                for r in filtered
                if str(r.get("relative_folder") or "").replace("\\", "/").lower() == norm.lower()
            ]
        return prep_rows, {str(r["document_id"]) for r in filtered if r.get("document_id")}
    return prep_rows, all_ids


def _write_friendly_preview_csv(
    path: Path,
    prep_rows: List[Dict[str, Any]],
    allowed_doc_ids: AbstractSet[str],
    document_results: Dict[str, Any],
) -> None:
    label_for = {
        str(r.get("document_id")): (str(r.get("relative_path") or "") or str(r.get("document_id")))
        for r in prep_rows
        if r.get("document_id")
    }
    path.parent.mkdir(parents=True, exist_ok=True)

    def _sort_key(ent: Dict[str, Any]) -> tuple:
        return _entry_sort_key(ent)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Document", "Search", "Result", "Location"])
        for doc_id in sorted(allowed_doc_ids):
            agg = document_results.get(doc_id)
            if not isinstance(agg, dict):
                continue
            entries = (agg.get("document_summary") or {}).get("entries") or []
            doc_label = label_for.get(doc_id, doc_id)
            for ent in sorted((e for e in entries if isinstance(e, dict)), key=_sort_key):
                qn = str(ent.get("query_name") or "").strip() or "—"
                w.writerow([doc_label, qn, _format_friendly_entry_value(ent), _page_label_for_entry(ent)])


def _merge_hitl_into_export_rows(session_dir: Path, rows: List[Dict[str, Any]]) -> None:
    hitl_state_path = session_dir / "hitl_state.json"
    if not hitl_state_path.exists():
        return
    try:
        state_rows = json.loads(hitl_state_path.read_text(encoding="utf-8"))
        by_doc = {
            str(r.get("DocumentId") or r.get("document_id") or ""): r
            for r in state_rows
            if isinstance(r, dict)
        }
        for row in rows:
            doc_id = str(row.get("document_id", ""))
            st = by_doc.get(doc_id)
            if not st:
                continue
            hitl_state = str(st.get("HitlState") or st.get("hitl_state") or row.get("hitl_state") or "none")
            human_verified = bool(st.get("HumanVerified") or st.get("human_verified"))
            notes = str(st.get("Notes") or st.get("notes") or "")
            corrections = st.get("Corrections") or st.get("corrections") or []
            row["hitl_state"] = hitl_state
            row["human_verified"] = human_verified
            row["hitl_notes"] = notes
            row["hitl_corrections"] = corrections
            counts = row.get("page_status_counts", {}) if isinstance(row.get("page_status_counts", {}), dict) else {}
            row["is_export_complete_only"] = is_export_complete_only(counts, hitl_state=hitl_state, human_verified=human_verified)
            if hitl_state == PAGE_HITL_CORRECTED:
                row["status"] = PAGE_HITL_CORRECTED
                row["issues"] = ""
    except Exception:
        pass


def _prepare_export_rows(session_dir: Path, mode: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    resolved_path = session_dir / "resolved_results.json"
    if not resolved_path.exists():
        # Self-heal: a run that died before its resolve phase has no
        # resolved_results.json, but its per-document results are persisted —
        # rebuild instead of failing the user's View Results with a traceback.
        try:
            _try_resolve_from_persisted_session(session_dir, session_dir.name, "")
        except Exception:
            logging.exception("Self-heal resolve failed for %s", session_dir.name)
    if not resolved_path.exists():
        raise FileNotFoundError(f"resolved_results.json not found for run {session_dir.name}")
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = copy.deepcopy(payload.get("export_rows") or [])
    
    # Filter validation results based on original_task_ids from the initial run
    validation_results = payload.get("validation_results", {}) if isinstance(payload.get("validation_results", {}), dict) else {}
    original_task_ids = set(payload.get("original_task_ids", []) or [])
    filtered_validation = dict(validation_results) if isinstance(validation_results, dict) else {}
    if isinstance(filtered_validation, dict):
        if "validate_revision" not in original_task_ids:
            filtered_validation["schema_discrepancies"] = []
        if "check_missing_data" not in original_task_ids:
            filtered_validation["batch_missing_data"] = []
            filtered_validation["missing_queries"] = []
        if "cross_reference_validation" not in original_task_ids:
            filtered_validation["cross_reference_results"] = {
                "manifest": [],
                "extracted_references": [],
                "resolution_records": [],
                "anomalies": [],
                "summary": {
                    "missing_documents_count": 0,
                    "orphaned_documents_count": 0,
                    "low_confidence_count": 0,
                    "pass_state": True,
                    "hitl_tooltip": "",
                },
            }
    # Use filtered validation for internal rows
    payload_with_filtered_validation = dict(payload)
    payload_with_filtered_validation["validation_results"] = filtered_validation
    
    internal_rows = _internal_rows_from_payload(payload_with_filtered_validation, rows)
    _merge_hitl_into_export_rows(session_dir, rows)
    if mode != "all":
        rows = [r for r in rows if bool(r.get("is_export_complete_only"))]
    return rows, internal_rows




def _export_internal_headers() -> List[str]:
    return [
        "category",
        "source_document",
        "referenced_document",
        "anomaly_type",
        "reference_type",
        "raw_reference_string",
        "closest_manifest_match",
        "page_or_section",
        "severity",
        "notes",
    ]


def _entry_sort_key(ent: Dict[str, Any]) -> tuple:
    pid = str(ent.get("page_id") or "")
    m = re.search(r"_PG(\d+)$", pid, re.I)
    pn = int(m.group(1)) if m else 999999
    return (pn, str(ent.get("query_name") or ""))


def _build_export_headers(rows: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
    return ["document", "status", "human_reviewed", "field", "value", "page"], []


def _write_export_csv_files(
    main_path: Path,
    internal_path: Path,
    rows: List[Dict[str, Any]],
    internal_rows: List[Dict[str, Any]],
    doc_results: Optional[Dict[str, Any]] = None,
) -> None:
    headers, _ = _build_export_headers(rows)
    internal_headers = _export_internal_headers()
    main_path.parent.mkdir(parents=True, exist_ok=True)
    with main_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            raw_path = str(row.get("relative_path") or row.get("document_id") or "")
            doc_label = Path(raw_path).name if raw_path else str(row.get("document_id", ""))
            status = row.get("status", "")
            human_reviewed = bool(row.get("human_verified"))
            agg = (doc_results or {}).get(str(row.get("document_id", ""))) or {}
            entries = (agg.get("document_summary") or {}).get("entries") or []
            valid = sorted((e for e in entries if isinstance(e, dict)), key=_entry_sort_key)
            if valid:
                for ent in valid:
                    writer.writerow([
                        doc_label, status, human_reviewed,
                        str(ent.get("query_name") or "").strip() or "\u2014",
                        _format_friendly_entry_value(ent),
                        _page_label_for_entry(ent),
                    ])
            else:
                writer.writerow([doc_label, status, human_reviewed, "", "", ""])
    if internal_rows:
        with internal_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(internal_headers)
            for row in internal_rows:
                writer.writerow(
                    [
                        row.get("category", ""),
                        row.get("source_document"),
                        row.get("referenced_document"),
                        row.get("anomaly_type"),
                        row.get("reference_type"),
                        row.get("raw_reference_string"),
                        row.get("closest_manifest_match"),
                        row.get("page_or_section"),
                        row.get("severity"),
                        row.get("notes"),
                    ]
                )


def run_preview_csv(args: argparse.Namespace) -> int:
    """Friendly CSV preview under the OS temp dir; scope by selected batch / folder / document from the UI."""
    ensure_dirs()
    session_dir = SESSIONS_ROOT / args.run_id
    resolved_path = session_dir / "resolved_results.json"
    if not resolved_path.exists():
        # Self-heal from persisted per-document results (crashed/partial runs).
        try:
            _try_resolve_from_persisted_session(session_dir, args.run_id, "")
        except Exception:
            logging.exception("Self-heal resolve failed for %s", args.run_id)
    if not resolved_path.exists():
        raise FileNotFoundError(f"resolved_results.json not found for run {args.run_id}")
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    doc_results = payload.get("document_results") if isinstance(payload.get("document_results"), dict) else {}
    prep_rows, allowed = _resolve_preview_document_ids(
        session_dir,
        args.mode,
        getattr(args, "document_ids", ""),
        getattr(args, "batch_key", ""),
        bool(getattr(args, "restrict_to_relative_folder", False)),
        str(getattr(args, "relative_folder", "") or ""),
    )
    tmp_root = Path(tempfile.mkdtemp(prefix="dec_csv_preview_"))
    main_csv = tmp_root / "results_preview.csv"
    _write_friendly_preview_csv(main_csv, prep_rows, allowed, doc_results)
    print(str(main_csv.resolve()))
    return 0


def run_export(args: argparse.Namespace) -> int:
    ensure_dirs()
    session_dir = SESSIONS_ROOT / args.run_id
    rows, internal_rows = _prepare_export_rows(session_dir, args.mode)

    # Load document_results for per-field page info.
    try:
        _raw = json.loads((session_dir / "resolved_results.json").read_text(encoding="utf-8"))
        _doc_results: Dict[str, Any] = _raw.get("document_results") or {} if isinstance(_raw.get("document_results"), dict) else {}
    except Exception:
        _doc_results = {}

    # Apply selection scope (document / batch / folder) — mirrors preview_csv scoping.
    doc_ids_csv = str(getattr(args, "document_ids", "") or "").strip()
    batch_key   = str(getattr(args, "batch_key", "") or "").strip()
    if doc_ids_csv:
        want = {x.strip() for x in doc_ids_csv.split(",") if x.strip()}
        rows = [r for r in rows if str(r.get("document_id", "")) in want]
    elif batch_key:
        rows = [r for r in rows if _row_batch_key(r) == batch_key]
        if bool(getattr(args, "restrict_to_relative_folder", False)):
            norm = str(getattr(args, "relative_folder", "") or "").replace("\\", "/")
            rows = [
                r for r in rows
                if str(r.get("relative_folder") or "").replace("\\", "/").lower() == norm.lower()
            ]

    out_opt = str(getattr(args, "output", None) or "").strip()
    if out_opt:
        xlsx_path = Path(out_opt)
        if xlsx_path.suffix.lower() != ".xlsx":
            xlsx_path = xlsx_path.with_suffix(".xlsx")
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        csv_fallback_path = xlsx_path.with_suffix(".csv")
        internal_csv_path = xlsx_path.parent / f"{xlsx_path.stem}_internal_val.csv"
    else:
        export_dir = session_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        xlsx_path = export_dir / f"resolved_{args.mode}_{ts}.xlsx"
        csv_fallback_path = export_dir / f"resolved_{args.mode}_{ts}.csv"
        internal_csv_path = export_dir / f"internal_val_results_{args.mode}_{ts}.csv"
    headers, all_selected_fields = _build_export_headers(rows)
    internal_headers = _export_internal_headers()

    try:
        import openpyxl  # type: ignore

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Results"
        ws.append(headers)
        for row in rows:
            raw_path = str(row.get("relative_path") or row.get("document_id") or "")
            doc_label = Path(raw_path).name if raw_path else str(row.get("document_id", ""))
            status = row.get("status", "")
            human_reviewed = bool(row.get("human_verified"))
            agg = _doc_results.get(str(row.get("document_id", ""))) or {}
            entries = (agg.get("document_summary") or {}).get("entries") or []
            valid = sorted((e for e in entries if isinstance(e, dict)), key=_entry_sort_key)
            if valid:
                for ent in valid:
                    ws.append([
                        doc_label, status, human_reviewed,
                        str(ent.get("query_name") or "").strip() or "\u2014",
                        _format_friendly_entry_value(ent),
                        _page_label_for_entry(ent),
                    ])
            else:
                ws.append([doc_label, status, human_reviewed, "", "", ""])
        if internal_rows:
            ws_internal = wb.create_sheet(title="Internal Validation")
            ws_internal.append(internal_headers)
            for row in internal_rows:
                ws_internal.append(
                    [
                        row.get("source_document"),
                        row.get("referenced_document"),
                        row.get("anomaly_type"),
                        row.get("reference_type"),
                        row.get("raw_reference_string"),
                        row.get("closest_manifest_match"),
                        row.get("page_or_section"),
                        row.get("severity"),
                        row.get("notes"),
                    ]
                )
        wb.save(str(xlsx_path))
        print(str(xlsx_path))
        return 0
    except Exception:
        _write_export_csv_files(csv_fallback_path, internal_csv_path, rows, internal_rows, _doc_results)
        print(str(csv_fallback_path))
        return 0


def run_export_stamped(args: argparse.Namespace) -> int:
    """Export documents with stamp applied to first page."""
    ensure_dirs()
    session_dir = SESSIONS_ROOT / args.run_id
    
    if not session_dir.exists():
        logging.error(f"Session directory not found: {session_dir}")
        _append_event(session_dir, "stamp_export_error", {"error": "session_not_found"})
        return 1
    
    stamp_path = Path(args.stamp_image_path)
    if not stamp_path.exists():
        logging.error(f"Stamp image not found: {stamp_path}")
        _append_event(session_dir, "stamp_export_error", {"error": "stamp_not_found", "path": str(stamp_path)})
        return 1
    
    # Load stamp image
    try:
        stamp_image_bytes = stamp_path.read_bytes()
    except Exception as e:
        logging.error(f"Failed to read stamp image: {e}")
        _append_event(session_dir, "stamp_export_error", {"error": "stamp_read_failed", "detail": str(e)})
        return 1
    
    # Import stamping module
    try:
        from stamping import apply_stamp_to_pdf, convert_image_to_pdf_with_stamp
        from stamping.stamp_applicator import validate_stamp_image
    except ImportError as e:
        logging.error(f"Stamping module not available: {e}")
        _append_event(session_dir, "stamp_export_error", {"error": "stamping_module_unavailable", "detail": str(e)})
        return 1
    
    # Validate stamp image
    is_valid, error_msg = validate_stamp_image(stamp_image_bytes)
    if not is_valid:
        logging.error(f"Invalid stamp image: {error_msg}")
        _append_event(session_dir, "stamp_export_error", {"error": "invalid_stamp", "detail": error_msg})
        return 1
    
    # Load document manifest
    manifest_path = session_dir / "document_manifest.json"
    if not manifest_path.exists():
        logging.error(f"Document manifest not found: {manifest_path}")
        _append_event(session_dir, "stamp_export_error", {"error": "manifest_not_found"})
        return 1
    
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.error(f"Failed to load manifest: {e}")
        _append_event(session_dir, "stamp_export_error", {"error": "manifest_load_failed", "detail": str(e)})
        return 1
    
    documents = manifest.get("documents", [])
    if not documents:
        logging.warning("No documents in manifest")
        _append_event(session_dir, "stamp_export_complete", {"success_count": 0, "hitl_queue_count": 0, "error_count": 0})
        return 0
    
    # Filter by document IDs if specified
    doc_ids_csv = str(getattr(args, "document_ids", "") or "").strip()
    if doc_ids_csv:
        want_ids = {x.strip() for x in doc_ids_csv.split(",") if x.strip()}
        documents = [d for d in documents if d.get("document_id") in want_ids]
    
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    _append_event(session_dir, "stamp_export_started", {"document_count": len(documents)})
    
    success_count = 0
    hitl_queue_count = 0
    error_count = 0
    
    # Load HITL state for tracking
    hitl_path = session_dir / "hitl_state.json"
    try:
        hitl_state = json.loads(hitl_path.read_text(encoding="utf-8")) if hitl_path.exists() else {}
    except Exception:
        hitl_state = {}
    
    for idx, doc in enumerate(documents):
        doc_id = doc.get("document_id", f"unknown_{idx}")
        source_path = doc.get("source_path") or doc.get("file_path")
        
        if not source_path:
            logging.warning(f"No source path for document {doc_id}")
            error_count += 1
            continue
        
        source_path = Path(source_path)
        if not source_path.exists():
            logging.warning(f"Source file not found: {source_path}")
            error_count += 1
            continue
        
        # Determine output path
        if output_dir:
            out_path = output_dir / f"{doc_id}_stamped.pdf"
        else:
            # Overwrite original (convert to PDF if needed)
            if source_path.suffix.lower() == ".pdf":
                out_path = source_path
            else:
                out_path = source_path.with_suffix(".pdf")
        
        _append_event(session_dir, "stamp_export_progress", {
            "current": idx + 1,
            "total": len(documents),
            "document_id": doc_id,
            "status": "processing"
        })
        
        try:
            if source_path.suffix.lower() == ".pdf":
                # Stamp PDF directly
                result = apply_stamp_to_pdf(
                    str(source_path),
                    stamp_image_bytes,
                    str(out_path)
                )
            else:
                # Convert image to PDF with stamp
                result = convert_image_to_pdf_with_stamp(
                    str(source_path),
                    stamp_image_bytes,
                    str(out_path)
                )
            
            if result:
                success_count += 1
                logging.info(f"Stamped document: {doc_id} -> {out_path}")
            else:
                # No suitable whitespace found - add to HITL queue
                hitl_queue_count += 1
                hitl_state[doc_id] = {
                    "status": "stamp_placement_required",
                    "reason": "insufficient_whitespace",
                    "source_path": str(source_path),
                }
                _append_event(session_dir, "stamp_needs_hitl", {
                    "document_id": doc_id,
                    "reason": "insufficient_whitespace"
                })
                logging.info(f"Document {doc_id} needs manual stamp placement")
                
        except Exception as e:
            logging.error(f"Failed to stamp document {doc_id}: {e}")
            error_count += 1
            _append_event(session_dir, "stamp_export_progress", {
                "current": idx + 1,
                "total": len(documents),
                "document_id": doc_id,
                "status": "error",
                "error": str(e)
            })
    
    # Save updated HITL state
    if hitl_queue_count > 0:
        try:
            hitl_path.write_text(json.dumps(hitl_state, indent=2), encoding="utf-8")
        except Exception as e:
            logging.warning(f"Failed to save HITL state: {e}")
    
    _append_event(session_dir, "stamp_export_complete", {
        "success_count": success_count,
        "hitl_queue_count": hitl_queue_count,
        "error_count": error_count
    })
    
    logging.info(f"Stamp export complete: {success_count} stamped, {hitl_queue_count} need HITL, {error_count} errors")
    
    # Print summary for UI consumption
    print(json.dumps({
        "success_count": success_count,
        "hitl_queue_count": hitl_queue_count,
        "error_count": error_count
    }))
    
    return 0 if error_count == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("session_runtime")
    sub = parser.add_subparsers(dest="cmd", required=True)

    extract = sub.add_parser("extract")
    extract.add_argument("--input-path", required=True)
    extract.add_argument("--run-id")
    extract.add_argument("--module-a-enabled", action="store_true")
    extract.add_argument("--page-timeout-sec", type=int, default=int(os.environ.get("PIPELINE_PAGE_EXTRACT_TIMEOUT_SEC", "600")))

    resolve = sub.add_parser("resolve")
    resolve.add_argument("--run-id", required=True)
    resolve.add_argument("--query-package", required=True)

    resolve_disk = sub.add_parser("resolve_from_disk")
    resolve_disk.add_argument("--run-id", required=True)
    resolve_disk.add_argument("--query-package", required=True)

    resolve_partial = sub.add_parser("resolve_partial")
    resolve_partial.add_argument("--run-id", required=True)
    resolve_partial.add_argument("--query-package", required=True)

    resolve_view = sub.add_parser("resolve_for_view")
    resolve_view.add_argument("--run-id", required=True)
    resolve_view.add_argument("--query-package", required=True)

    orchestrate = sub.add_parser("orchestrate")
    orchestrate.add_argument("--input-path", required=True)
    orchestrate.add_argument("--run-id")
    orchestrate.add_argument("--module-a-enabled", action="store_true")
    orchestrate.add_argument("--page-timeout-sec", type=int, default=int(os.environ.get("PIPELINE_PAGE_EXTRACT_TIMEOUT_SEC", "600")))

    purge = sub.add_parser("purge")
    purge.add_argument("--days", type=int, default=None)
    purge.add_argument("--all", action="store_true")

    export_cmd = sub.add_parser("export")
    export_cmd.add_argument("--run-id", required=True)
    export_cmd.add_argument("--mode", choices=["complete", "all"], default="complete")
    export_cmd.add_argument(
        "--output",
        default=None,
        help="Write the XLSX (or CSV fallback) to this path instead of session exports/",
    )
    export_cmd.add_argument(
        "--document-ids",
        default="",
        help="Comma-separated document ids (single-document scope); highest priority if non-empty",
    )
    export_cmd.add_argument(
        "--batch-key",
        default="",
        help="Limit to export rows for this batch key (__default__ or batch id)",
    )
    export_cmd.add_argument(
        "--restrict-to-relative-folder",
        action="store_true",
        help="With --batch-key, also filter by --relative-folder",
    )
    export_cmd.add_argument(
        "--relative-folder",
        default="",
        help="POSIX-style folder path within the batch when --restrict-to-relative-folder is set",
    )

    preview_csv_cmd = sub.add_parser("preview_csv")
    preview_csv_cmd.add_argument("--run-id", required=True)
    preview_csv_cmd.add_argument("--mode", choices=["complete", "all"], default="complete")
    preview_csv_cmd.add_argument(
        "--document-ids",
        default="",
        help="Comma-separated document ids (single-document scope); highest priority if non-empty",
    )
    preview_csv_cmd.add_argument(
        "--batch-key",
        default="",
        help="Limit to export rows for this batch key (__default__ or batch id)",
    )
    preview_csv_cmd.add_argument(
        "--restrict-to-relative-folder",
        action="store_true",
        help="With --batch-key, also filter by --relative-folder (use empty string for batch root)",
    )
    preview_csv_cmd.add_argument(
        "--relative-folder",
        default="",
        help="POSIX-style folder path within the batch when --restrict-to-relative-folder is set",
    )

    # Stamped export command
    export_stamped_cmd = sub.add_parser("export_stamped")
    export_stamped_cmd.add_argument("--run-id", required=True)
    export_stamped_cmd.add_argument("--stamp-image-path", required=True, help="Path to stamp image file (PNG or JPG)")
    export_stamped_cmd.add_argument(
        "--document-ids",
        default="",
        help="Comma-separated document ids to stamp; if empty, stamps all documents",
    )
    export_stamped_cmd.add_argument(
        "--output-dir",
        default="",
        help="Output directory for stamped PDFs; defaults to overwriting originals",
    )

    build_ann = sub.add_parser("build_annotations")
    build_ann.add_argument("--run-id", required=True)
    build_ann.add_argument("--doc-id", default="", help="Build one document; default = all docs in the run")
    build_ann.add_argument("--rebuild", action="store_true",
                           help="Regenerate the graph from review_data even if one exists (DISCARDS saved human edits)")

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--doc-id", required=True)
    finalize.add_argument("--force", action="store_true", help="Finalize even if not marked verified (testing)")

    return parser


def _page_number_from_page_id(page_id: str) -> Optional[int]:
    """Page number from the ``..._PG001`` suffix process_input assigns (see page_id format)."""
    m = re.search(r"_PG(\d+)$", str(page_id or ""))
    return int(m.group(1)) if m else None


def _ensure_page_rasters(session_dir: Path, rd: Dict[str, Any]) -> Dict[str, str]:
    """Make sure every page of this document has a review raster, re-rendering from the
    ORIGINAL source file where it is missing. Returns {page_id: png path}.

    Why this exists: review_data only persists the render array when ``page.image`` is still
    resident, and the pipeline releases page images as soon as a page is extracted (decoded in
    the worker, dropped straight after). So ``render_image_path`` comes back null and the review
    tab falls back to a blank grid. Re-rendering through render_page_for_extraction reproduces
    the SAME array extraction ran on (same normalisation, same deskew), so the boxes still align.

    Purely additive: writes PNGs under review_data/rasters/ and returns paths. It never edits a
    page's regions, classes or OCR lines, and it never raises — a page it cannot rebuild simply
    keeps the blank canvas it has now.
    """
    raster_dir = session_dir / "review_data" / "rasters"
    out: Dict[str, str] = {}
    try:
        from process_input import render_page_for_extraction
        from review_data import _save_render_image
    except Exception:
        logging.debug("ensure_rasters: render/save helpers unavailable", exc_info=True)
        return out

    for pg in rd.get("pages", []) or []:
        page_id = str(pg.get("page_id") or "")
        if not page_id:
            continue
        existing = pg.get("render_image_path")
        if existing and os.path.exists(existing):
            out[page_id] = existing
            continue
        cached = raster_dir / f"{page_id}.png"
        if cached.exists():
            out[page_id] = str(cached)
            continue

        src = pg.get("storage_path")
        if not src or not os.path.exists(src):
            logging.info("ensure_rasters: %s source unavailable (%s)", page_id, src)
            continue
        page_number = _page_number_from_page_id(page_id)
        if page_number is None:
            logging.info("ensure_rasters: %s has no page number in its id", page_id)
            continue

        try:
            ext = os.path.splitext(src)[1].lower()
            pd_ = render_page_for_extraction(
                src, ext, page_number,
                str(pg.get("document_id") or rd.get("document_id") or ""),
                str(pg.get("batch_id") or rd.get("batch_id") or ""))
            img = getattr(pd_, "image", None)
            if img is None or not hasattr(img, "shape"):
                logging.info("ensure_rasters: %s re-render produced no array", page_id)
                continue

            # The stored boxes live in the review_data canvas space. Scale the re-render onto
            # exactly those dimensions so every bbox lands where it did at extraction time.
            want_w = int(float(pg.get("render_width") or pg.get("width") or 0) or 0)
            want_h = int(float(pg.get("render_height") or pg.get("height") or 0) or 0)
            got_h, got_w = int(img.shape[0]), int(img.shape[1])
            if want_w > 1 and want_h > 1 and (got_w, got_h) != (want_w, want_h):
                ar_src = got_w / float(got_h)
                ar_dst = want_w / float(want_h)
                if abs(ar_src - ar_dst) / max(ar_dst, 1e-6) > 0.02:
                    logging.warning(
                        "ensure_rasters: %s aspect mismatch (render %dx%d vs canvas %dx%d) - "
                        "refusing, boxes would not align", page_id, got_w, got_h, want_w, want_h)
                    continue
                import cv2 as _cv2
                img = _cv2.resize(img, (want_w, want_h), interpolation=_cv2.INTER_AREA)

            path = _save_render_image(img, str(raster_dir), page_id)
            if path:
                out[page_id] = path
                logging.info("ensure_rasters: rebuilt %s", page_id)
        except Exception:
            logging.debug("ensure_rasters: %s re-render failed", page_id, exc_info=True)

    return out


def run_build_annotations(args: argparse.Namespace) -> int:
    """Review-stage entry: build editable annotation graphs (annotations/<doc>.json) from the
    persisted review_data. Maps OCULI's classes -> the 3 detector classes via build_page_regions.
    Reads review_data only — never re-extracts. The frontend calls this when a doc is opened for
    review; it then edits + saves annotations/<doc>.json back. An already-verified graph is NOT
    rebuilt (never clobber human edits)."""
    import gzip as _gz
    ensure_dirs()
    session_dir = SESSIONS_ROOT / args.run_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Unknown run_id: {args.run_id}")
    # sidecar dir is on sys.path via the pipeline import at module load
    from annotation_graph import build_page_regions, build_document_graph, frame_bboxes

    review_dir = session_dir / "review_data"
    out_dir = session_dir / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_id = (getattr(args, "doc_id", "") or "").strip()
    files = [review_dir / f"{doc_id}.json.gz"] if doc_id else sorted(review_dir.glob("*.json.gz"))

    built = 0
    for rf in files:
        if not rf.exists():
            logging.warning("build_annotations: review_data missing: %s", rf)
            continue
        with _gz.open(rf, "rt", encoding="utf-8") as fh:
            rd = json.load(fh)
        this_doc = rd.get("document_id")
        out_path = out_dir / f"{this_doc}.json"

        # Make the page images available BEFORE either branch below. The review tab cannot
        # display a PDF directly, so each page is rendered from its original source once and
        # cached as a PNG. Additive only.
        rasters = _ensure_page_rasters(session_dir, rd)

        # An existing graph may hold human edits, so it is never regenerated (see below). It
        # must still be told where its images are: patch ONLY image_path, per page, and only
        # when the current one is unusable. Regions, classes and the verified flag are never
        # touched here — that is the difference between fixing the backdrop and destroying a
        # fortnight of review.
        if out_path.exists() and not getattr(args, "rebuild", False) and rasters:
            try:
                graph = json.loads(out_path.read_text(encoding="utf-8"))
                patched = 0
                for page in graph.get("pages", []) or []:
                    pid = str(page.get("page_id") or "")
                    cur = page.get("image_path")
                    if cur and os.path.exists(cur):
                        continue
                    new = rasters.get(pid)
                    if new:
                        page["image_path"] = new
                        patched += 1
                if patched:
                    write_json(out_path, graph)
                    logging.info("build_annotations: patched image_path on %d page(s) of %s",
                                 patched, out_path.name)
            except Exception:
                logging.warning("build_annotations: image_path patch failed for %s",
                                out_path.name, exc_info=True)
        # Never clobber an existing graph — it may hold human edits that were SAVED but not
        # yet verified. Reopening a doc must NOT rebuild from review_data (that was the root
        # cause of "my edits reverted on reopen / save didn't work"). Pass --rebuild to force
        # regeneration, e.g. after re-extraction changed the review_data.
        if out_path.exists() and not getattr(args, "rebuild", False):
            logging.info("build_annotations: keep existing graph %s (pass --rebuild to regenerate)", out_path.name)
            continue
        pages = []
        for pg in rd.get("pages", []):
            regions = build_page_regions(
                pg.get("cleaning_inputs", {}), pg.get("classification"),
                pg.get("consolidation"), composite_regions=None,
                anchors=pg.get("anchors"),
            )
            # Clamp every box to the page canvas. Region-OCR detections could previously run
            # past the raster edge (docTR unclips each box; the patch->page conversion did not
            # clamp), so a minority of text boxes extended beyond the page and were drawn off
            # it in the review overlay. A box outside the image cannot be correct, so trimming
            # it strictly reduces error. Applied only to a FRESHLY built graph: a graph that
            # already exists is never re-derived here, so no reviewed box is ever moved.
            _cw = float(pg.get("render_width") or pg.get("width") or 0) or 0.0
            _ch = float(pg.get("render_height") or pg.get("height") or 0) or 0.0
            if _cw > 1 and _ch > 1:
                # Trim only where the box survives with area. A region lying ENTIRELY outside the
                # canvas keeps its original coordinates instead of being clamped away: those
                # boxes carry real OCR text whose position is wrong, and deleting them would
                # lose the reading as well. Nothing here ever removes a region.
                def _clamp_bb(bb):
                    if not bb or len(bb) < 4:
                        return bb
                    x1 = min(max(float(bb[0]), 0.0), _cw)
                    y1 = min(max(float(bb[1]), 0.0), _ch)
                    x2 = min(max(float(bb[2]), 0.0), _cw)
                    y2 = min(max(float(bb[3]), 0.0), _ch)
                    return [x1, y1, x2, y2] if (x2 > x1 and y2 > y1) else list(bb[:4])
                for _r in regions:
                    _r["bbox"] = _clamp_bb(_r.get("bbox"))
                    for _t in (_r.get("table_internal") or []):
                        _t["bbox"] = _clamp_bb(_t.get("bbox"))
            pages.append({
                "page_id": pg.get("page_id"),
                # use the RENDER space (the bbox pixel space) so boxes align on the
                # frontend + synthetic canvas; fall back to PageMetadata dims.
                "width": pg.get("render_width") or pg.get("width"),
                "height": pg.get("render_height") or pg.get("height"),
                # coordinate-authoritative render array (pixel-aligned with all bboxes). Only a
                # raster guaranteed to BE the canvas size is acceptable: the ensured one, or the
                # one review_data recorded.
                #
                # storage_path is deliberately NOT a fallback. It is the original file, so for a
                # PDF the review tab cannot decode it at all, and for an image it loads at the
                # ORIGINAL dimensions — which need not equal the render canvas the bboxes live
                # in. Nothing rescales it, so every box would be silently misaligned over a
                # plausible-looking page. A blank canvas at the right size is the safer failure:
                # it is obviously wrong rather than quietly wrong.
                "image_path": (rasters.get(str(pg.get("page_id") or ""))
                               or pg.get("render_image_path")),
                "regions": regions,
                # Sheet frame (rendered, never labelled) for the in-place synthetic. Prefer the
                # frame_detection service's result (pg["frame_bbox"]); fall back to a BORDER_FRAME
                # classification if that's absent (older review_data).
                "frame_bboxes": ([pg["frame_bbox"]] if pg.get("frame_bbox")
                                 else frame_bboxes(pg.get("classification"), pg.get("cleaning_inputs", {}))),
            })
        write_json(out_path, build_document_graph(this_doc, pages))
        built += 1
    logging.info("build_annotations: wrote %d graph(s) to %s", built, out_dir)
    print(json.dumps({"built": built, "annotations_dir": str(out_dir)}), flush=True)
    return 0


def run_finalize(args: argparse.Namespace) -> int:
    """Per-doc VERIFIED GATE: after the human verifies classifications in the annotation tab,
    recompute region features (Gate 1) then generate the synthetic IP-free labeled page(s)
    (Gates 3-5). Requires the annotation graph to be verified (frontend sets verified=true)
    unless --force. Per document only; idempotent (re-run overwrites the synthetic)."""
    import gzip as _gz
    ensure_dirs()
    session_dir = SESSIONS_ROOT / args.run_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Unknown run_id: {args.run_id}")
    from annotation_graph import recompute_region_features
    from synthetic_page_generator import generate_document

    doc_id = (getattr(args, "doc_id", "") or "").strip()
    if not doc_id:
        raise ValueError("finalize requires --doc-id")
    ann_path = session_dir / "annotations" / f"{doc_id}.json"
    review_path = session_dir / "review_data" / f"{doc_id}.json.gz"
    if not ann_path.exists():
        raise FileNotFoundError(f"annotation graph missing (build_annotations first): {ann_path}")
    graph = json.loads(ann_path.read_text(encoding="utf-8"))
    if not graph.get("verified") and not getattr(args, "force", False):
        raise RuntimeError(f"doc {doc_id} not verified — finalize runs only after HITL verification (or pass --force)")

    # Gate 1: recompute region features (text via OCR spatial query) from review_data ocr_lines
    ocr_by_page: Dict[str, Any] = {}
    if review_path.exists():
        with _gz.open(review_path, "rt", encoding="utf-8") as fh:
            review = json.load(fh)
        ocr_by_page = {p.get("page_id"): p.get("ocr_lines", []) for p in review.get("pages", [])}
    for page in graph.get("pages", []):
        ol = ocr_by_page.get(page.get("page_id"), [])
        for r in page.get("regions", []):
            recompute_region_features(r, ol)
            r["human_verified"] = True   # doc verified -> all regions verified
    write_json(ann_path, graph)

    # Gates 3-5: synthetic IP-free labeled page(s) — the training sample(s)
    out_dir = session_dir / "synthetic"
    res = generate_document(str(ann_path), str(review_path), str(out_dir))
    write_json(session_dir / "synthetic" / f"{doc_id}.finalize.json",
               {"finalized": True, "doc_id": doc_id, "ts": utc_now(), **res})
    logging.info("finalize: doc %s -> %s", doc_id, res)
    print(json.dumps(res), flush=True)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.cmd == "extract":
        return run_extract(args)
    if args.cmd == "resolve":
        return run_resolve(args)
    if args.cmd == "resolve_from_disk":
        return run_resolve_from_disk(args)
    if args.cmd == "resolve_partial":
        return run_resolve_partial(args)
    if args.cmd == "resolve_for_view":
        return run_resolve_for_view(args)
    if args.cmd == "orchestrate":
        return run_orchestrate(args)
    if args.cmd == "purge":
        return run_purge(args)
    if args.cmd == "export":
        return run_export(args)
    if args.cmd == "preview_csv":
        return run_preview_csv(args)
    if args.cmd == "export_stamped":
        return run_export_stamped(args)
    if args.cmd == "build_annotations":
        return run_build_annotations(args)
    if args.cmd == "finalize":
        return run_finalize(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
