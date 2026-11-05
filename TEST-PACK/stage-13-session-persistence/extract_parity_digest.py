"""Canonical digest of an `extract` run, for byte-level parity comparison.

DIAGNOSTIC. Reads a session's ``completed_documents/*.json.gz`` — the full
per-document extraction payload the runtime persists — and reduces it to the
things a refactor must not change: every extracted value, every unmatched OCR
line, and the per-page status. Everything volatile between two identical runs
(run ids, batch ids, timestamps, absolute source paths) is scrubbed, so two
digests of the same build compare equal.

    $PY = 'C:\OCULI.pd1\TESTING\.venv-nopaddle\Scripts\python.exe'
    & $PY extract_parity_digest.py <session_dir> -o baseline.json

Compare two digests:

    & $PY extract_parity_digest.py --diff baseline.json candidate.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ids are derived from a batch id that changes between runs; the text they label
# does not. Scrub the id, keep the structure.
_BATCH_RE = re.compile(r"BATCH_\d+")
_DOC_RE = re.compile(r"DOC_[0-9a-fA-F]+")


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return _DOC_RE.sub("DOC_X", _BATCH_RE.sub("BATCH_X", value))
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    return value


def _round_bbox(bbox: Any) -> Any:
    """Geometry to 3dp. Float noise below that is not a behaviour change."""
    if isinstance(bbox, list):
        return [round(float(v), 3) if isinstance(v, (int, float)) else v for v in bbox]
    return bbox


def digest_document(entry: Dict[str, Any]) -> Dict[str, Any]:
    agg = entry.get("aggregated_results") or {}

    entries: List[Dict[str, Any]] = []
    for e in (agg.get("document_summary") or {}).get("entries") or []:
        rec: Dict[str, Any] = {
            "page": _scrub(e.get("page_id")),
            "data_format": e.get("data_format"),
            "confidence": e.get("confidence"),
            "orientation": e.get("orientation"),
            "value": e.get("value"),
        }
        label = {
            "query_name": e.get("query_name"),
            "anchor_bbox": _round_bbox(e.get("anchor_bbox")),
        }
        if e.get("data_format") == "TABLE":
            # MEASURED, pre-existing: two identical runs of the same build emit a
            # TABLE entry with byte-identical rows but a DIFFERENT representative
            # query_name and anchor_bbox -- whichever column header won a tie.
            # Affected 5 entries across 4 of 8 documents on the reference corpus,
            # with every recognised character unchanged. Nothing to do with OCR, so
            # it is held apart from the stable hash rather than allowed to mask a
            # real text regression. `sha256_strict` still covers it.
            rec["unstable_table_label"] = label
        else:
            rec.update(label)
        entries.append(rec)
    # Stable order: the runtime's own ordering is the thing under test for OCR
    # output, but aggregation order across pages is not guaranteed, so sort on
    # content and compare as a multiset.
    entries.sort(key=lambda d: json.dumps(d, sort_keys=True, ensure_ascii=False))

    um = agg.get("unmatched_text") or {}
    lines: List[Dict[str, Any]] = []
    for ln in um.get("lines") or []:
        lines.append({
            "page": _scrub(ln.get("page_id")),
            "text": ln.get("text"),
            "bbox": _round_bbox(ln.get("bbox")),
            "source": ln.get("source"),
            "owner_id": ln.get("owner_id"),
            "region_label": ln.get("region_label"),
            "reason": ln.get("reason"),
        })
    lines.sort(key=lambda d: json.dumps(d, sort_keys=True, ensure_ascii=False))

    pages = []
    for p in entry.get("page_statuses") or []:
        pages.append({
            "page_number": p.get("page_number"),
            "status": p.get("status"),
            "errors": _scrub(p.get("errors")),
            "source_name": Path(str(p.get("source_path", ""))).name,
        })
    pages.sort(key=lambda d: (d.get("page_number") is None, d.get("page_number")))

    return {
        "relative_path": entry.get("relative_path") or entry.get("original_document_id"),
        "page_count": entry.get("page_count"),
        "total_pages": agg.get("total_pages"),
        "unmatched_total_lines": um.get("total_lines"),
        "unmatched_by_reason": um.get("by_reason"),
        "unmatched_lines": lines,
        "summary_entries": entries,
        "pages": pages,
    }


def digest_session(session_dir: Path) -> Dict[str, Any]:
    docs_dir = session_dir / "completed_documents"
    if not docs_dir.is_dir():
        raise SystemExit(f"no completed_documents/ under {session_dir}")
    docs = []
    for p in sorted(docs_dir.iterdir()):
        if p.name.endswith(".tmp"):
            continue
        if p.name.endswith(".json.gz"):
            raw = gzip.open(p, "rb").read().decode("utf-8")
        elif p.name.endswith(".json"):
            raw = p.read_text(encoding="utf-8")
        else:
            continue
        docs.append(digest_document(json.loads(raw)))
    docs.sort(key=lambda d: str(d.get("relative_path")))
    payload = {"document_count": len(docs), "documents": docs}
    strict = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)
    stable = json.dumps(_drop_unstable(payload), sort_keys=True, ensure_ascii=False, indent=2)
    payload["sha256"] = hashlib.sha256(stable.encode("utf-8")).hexdigest()
    payload["sha256_strict"] = hashlib.sha256(strict.encode("utf-8")).hexdigest()
    return payload


def _drop_unstable(obj: Any) -> Any:
    """The digest minus the fields two identical runs disagree on."""
    if isinstance(obj, dict):
        return {k: _drop_unstable(v) for k, v in obj.items() if k != "unstable_table_label"}
    if isinstance(obj, list):
        return [_drop_unstable(v) for v in obj]
    return obj


def _flatten(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "unmatched": [json.dumps(x, sort_keys=True, ensure_ascii=False) for x in doc["unmatched_lines"]],
        "entries": [json.dumps(_drop_unstable(x), sort_keys=True, ensure_ascii=False)
                    for x in doc["summary_entries"]],
    }


def diff(a_path: Path, b_path: Path) -> int:
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    if a.get("sha256") == b.get("sha256"):
        print(f"IDENTICAL  sha256={a.get('sha256')}  ({a['document_count']} documents)")
        if a.get("sha256_strict") != b.get("sha256_strict"):
            print("           (TABLE query_name/anchor_bbox differ -- known, run-to-run, "
                  "not OCR; see digest_document)")
        return 0
    print("DIFFERENT")
    by_a = {d["relative_path"]: d for d in a["documents"]}
    by_b = {d["relative_path"]: d for d in b["documents"]}
    for key in sorted(set(by_a) | set(by_b)):
        if key not in by_a:
            print(f"  + document only in candidate: {key}")
            continue
        if key not in by_b:
            print(f"  - document only in baseline:  {key}")
            continue
        fa, fb = _flatten(by_a[key]), _flatten(by_b[key])
        for field in ("unmatched", "entries"):
            sa, sb = set(fa[field]), set(fb[field])
            only_a, only_b = sorted(sa - sb), sorted(sb - sa)
            if not only_a and not only_b:
                continue
            print(f"  {key}  [{field}]  -{len(only_a)} +{len(only_b)}")
            for line in only_a[:10]:
                print(f"    - {line}")
            for line in only_b[:10]:
                print(f"    + {line}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="?", help="a sessions/<run_id> directory")
    ap.add_argument("-o", "--out", help="write the digest JSON here")
    ap.add_argument("--diff", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    args = ap.parse_args()

    if args.diff:
        return diff(Path(args.diff[0]), Path(args.diff[1]))

    if not args.session_dir:
        ap.error("session_dir is required unless --diff is used")
    payload = digest_session(Path(args.session_dir))
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"{args.out}  sha256={payload['sha256']}  documents={payload['document_count']}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
