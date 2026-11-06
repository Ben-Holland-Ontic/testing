# -*- coding: utf-8 -*-
"""Write page_quality.parquet: one row per page, for augmentation calibration.

WHY THIS IS A SEPARATE MODULE. The graph exporter's own entry point,
``pipeline.sidecar_export_document``, is dead code - defined once, imported once by
session_runtime, and called from nowhere. The inline call was removed on 2026-07-22 and left
"for reuse by the review-stage finalize step"; that reuse was never built, so the parquet
export has silently produced nothing since. This capture path deliberately does not depend
on it, so it cannot regress the same way.

WHY IT READS METADATA, NOT THE RASTER. ``page.image`` is released at several points during
extraction (pipeline.py:1027, :1883, input/process_input.py:1045), well before the
document-end cleanup at pipeline.py:3336. Anything that assumes a live raster this late is
the bug that already cost us the classify-regions page images. The measurement is therefore
taken inside ``_classify_and_branch``, where the raster exists by construction, and parked
in ``page.metadata["classifier_signals"]["page_quality"]``. This module only reads that
dict, which survives until page.metadata.clear().

Output: ``<session_dir>/page_quality.parquet``, ~33 metric columns plus the identifiers and
stratification keys needed to fit per-stratum distributions without joining another table.

Input coverage, and the one deliberate omission
-----------------------------------------------
    .pdf                    measured, and carries scan_branch from the classifier
    .jpg .jpeg .png .bmp    measured; no scan_branch, because images skip the branch
    .tif .tiff .webp .gif   classifier by design. Stratify these on `source` instead.
    .zip .tar               expanded upstream; contents are routed by their own extension
    .docx .doc .docm        UNSUPPORTED_WORD_EXT - not ingested for OCR at all
    .txt .csv               NOT measured, deliberately

`.txt` and `.csv` do produce a raster - `_extract_txt_pages` renders the text to an image -
so a hook there would "work". It is omitted on purpose: that raster is a synthetic,
perfectly clean render of plain text, not a captured page. Those rows would land in the
clean stratum, and because the clean floors are derived from that stratum's p97
(`calibration.floors_from`), they would raise the floors and make genuine degradation read
as less severe than it is. Collecting them would quietly bias every fit.

If they are ever wanted, nothing needs rebuilding: every row already carries `source`, so
they can be filtered at fit time rather than at capture time.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("page_quality_export")

# Reuse the graph exporter's writer: it already handles pyarrow with a JSONL fallback, and
# coerces mixed/None columns rather than failing the run.
try:
    from oculi_graph_exporter import _Writer          # type: ignore
    _WRITER_OK = True
except Exception:                                     # pragma: no cover
    _Writer = None                                    # type: ignore
    _WRITER_OK = False


def _enabled() -> bool:
    """Same switch as the rest of the sidecar (pipeline.py:2187)."""
    return str(os.environ.get("OCULI_SIDECAR_DISABLED", "")).strip().lower() \
        not in ("1", "true", "yes")


def _sf(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None                  # NaN -> None, so parquet stays typed
    except Exception:
        return None


def _page_number(page: Any, fallback: int) -> int:
    for attr in ("page_number", "page_index", "index"):
        v = getattr(page, attr, None)
        if isinstance(v, int):
            return v
    return fallback


def _resolve_doc_type(doc_meta: Dict[str, Any]) -> str:
    """document_type, resolved the same way the graph exporter resolves it.

    Reused rather than reimplemented: DOMAIN_DATA_PLAN stratifies the review budget over
    document_type, so a second definition that disagreed would silently split the strata.
    """
    try:
        from oculi_graph_exporter import _resolve_document_type
        return _resolve_document_type(doc_meta)
    except Exception:
        explicit = str(doc_meta.get("ingest_document_class", "") or "").strip()
        if explicit:
            return explicit
        rel = str(doc_meta.get("relative_folder", "") or "").strip()
        if rel:
            parts = rel.replace("\\", "/").strip("/").split("/")
            return parts[0] if parts else "Unknown"
        return "Unknown"


def _resolve_batch_id(session_dir: str) -> str:
    """batch_id as session_runtime persisted it (batch_id.json)."""
    try:
        import json
        p = Path(session_dir) / "batch_id.json"
        if p.is_file():
            return str(json.loads(p.read_text(encoding="utf-8")).get("batch_id", "") or "")
    except Exception:
        pass
    return ""


def collect_document(document: Any, session_dir: str = "") -> List[Dict[str, Any]]:
    """-> one row per page that carries a measurement. Never raises."""
    rows: List[Dict[str, Any]] = []
    try:
        pages = list(getattr(document, "pages", None) or [])
    except Exception:
        return rows

    doc_meta = getattr(document, "metadata", {}) or {}
    if not isinstance(doc_meta, dict):
        doc_meta = {}
    src_path = getattr(document, "source_path", "") or ""
    doc_id = getattr(document, "document_id", "") or ""
    filename = doc_meta.get("filename", "") or (os.path.basename(src_path) if src_path else "")
    rel_path = doc_meta.get("relative_path", "") or ""
    doc_type = _resolve_doc_type(doc_meta)
    batch_id = _resolve_batch_id(session_dir) if session_dir else ""

    for i, page in enumerate(pages):
        pmeta = getattr(page, "metadata", None)
        if not isinstance(pmeta, dict):
            continue
        signals = pmeta.get("classifier_signals") or {}
        if not isinstance(signals, dict):
            continue
        pq = signals.get("page_quality")
        if not isinstance(pq, dict) or not pq:
            continue

        row: Dict[str, Any] = {
            "page_id": getattr(page, "page_id", "") or "",
            "doc_id": doc_id,
            "batch_id": batch_id,
            "page_number": _page_number(page, i),
            "filename": filename,
            "relative_path": rel_path,
            # --- stratification keys. The corpus is roughly two-thirds born-digital, so a
            # pooled average would badly understate degradation; these are what let the fit
            # be done per stratum instead. DOMAIN_DATA_PLAN stratifies the review budget over
            # document_type x quality_bucket and dedups on phash, so all three live here and
            # the parquet needs no join to be usable.
            "document_type": doc_type,
            # quality_bucket IS scan_branch - the graph exporter writes it under that name
            # (oculi_graph_exporter.py:620). Emitted under both so neither doc is wrong.
            # Empty for image inputs: they skip the branch classifier by design, so there is
            # no verdict to report. Stratify those on `source` instead.
            "quality_bucket": str(pmeta.get("scan_branch", "") or ""),
            "scan_branch": str(pmeta.get("scan_branch", "") or ""),
            # "pdf" or "image" - the two ingestion paths measure the same way but only the
            # PDF path carries a scan_branch, so this is what separates them.
            "source": str(pmeta.get("source", "") or "pdf"),
            "scan_confidence": _sf(pmeta.get("scan_confidence")),
            "render_dpi": _sf(pmeta.get("render_dpi")),
            "target_dpi": _sf(pmeta.get("target_dpi")),
            "deskew_angle_deg": _sf(pmeta.get("deskew_angle_deg")),
        }
        for k, v in pq.items():
            key = k if k not in row else "pq_" + k
            # Only coerce numerics. An earlier version sent every non-int through _sf(),
            # which turned the hex phash string into None - the column existed, was typed,
            # and carried nothing. Exactly the silent-null failure this capture path exists
            # to stop, so strings and bools pass through untouched.
            if isinstance(v, bool) or isinstance(v, int) or isinstance(v, str):
                row[key] = v
            else:
                row[key] = _sf(v)
        rows.append(row)
    return rows


def write_page_quality(session_dir: str, document: Any) -> int:
    """Append this document's rows to <session_dir>/page_quality.jsonl. Never raises.

    Called once per document from both processing paths, while page.metadata is still
    populated. Returns the number of rows written, 0 if disabled or nothing to write.

    APPEND-ONLY, deliberately. The first version re-read the parquet and rewrote it whole on
    every document, because _Writer.flush() replaces the file. Measured, that cost 6.5 ms at
    one document and 24.4 ms at sixty - linear per write, so O(n^2) over a session: about
    2.5 minutes of pure rewrite over 10,000 pages, and roughly an hour over 50,000. Appending
    a line is O(1) and the parquet is produced once, by finalize_page_quality().

    It is also more robust on a machine that sleeps mid-run: a crash leaves every completed
    line intact, where a half-written parquet rewrite could lose the whole session.
    """
    if not _enabled():
        return 0
    try:
        rows = collect_document(document, session_dir)
        if not rows:
            return 0
        out = Path(session_dir)
        out.mkdir(parents=True, exist_ok=True)
        import json
        with open(out / "page_quality.jsonl", "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
        return len(rows)
    except Exception:
        logger.debug("page_quality write failed for %s",
                     getattr(document, "document_id", "?"), exc_info=True)
        return 0


def finalize_page_quality(session_dir: str) -> int:
    """Convert the appended JSONL into page_quality.parquet. Never raises.

    Safe to call repeatedly and safe to never call - the JSONL is the real record and
    calibration.fit() reads either. Returns the row count written, or 0.
    """
    out = Path(session_dir)
    jl = out / "page_quality.jsonl"
    if not jl.is_file():
        return 0
    try:
        import json
        with open(jl, "r", encoding="utf-8") as fh:
            rows = [json.loads(ln) for ln in fh if ln.strip()]
        if not rows:
            return 0
        if not _WRITER_OK:
            logger.info("page_quality: parquet writer unavailable; JSONL kept as the record")
            return 0
        w = _Writer(out, "page_quality")
        w.extend(rows)
        w.flush()
        return len(rows)
    except Exception:
        logger.debug("page_quality finalize failed for %s", session_dir, exc_info=True)
        return 0


def _read_existing(session_dir: Path) -> List[Dict[str, Any]]:
    """Rows already on disk, so a per-document flush does not truncate the session."""
    pq = session_dir / "page_quality.parquet"
    if pq.is_file():
        try:
            import pyarrow.parquet as pq_mod
            return pq_mod.read_table(pq).to_pylist()
        except Exception:
            logger.debug("page_quality: could not read existing parquet", exc_info=True)
            return []
    jl = session_dir / "page_quality.jsonl"
    if jl.is_file():
        try:
            import json
            with open(jl, "r", encoding="utf-8") as fh:
                return [json.loads(ln) for ln in fh if ln.strip()]
        except Exception:
            logger.debug("page_quality: could not read existing jsonl", exc_info=True)
    return []
