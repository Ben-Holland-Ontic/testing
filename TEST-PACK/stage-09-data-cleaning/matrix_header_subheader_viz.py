"""Generate matrix-role overlays (main header/subheader/value/row anchor) for selected images.

Batch a folder:
  py -3 tools/matrix_header_subheader_viz.py --dir "C:\\path\\to\\images" [--recursive]

Writes per-image summaries under ``output/unified_diag/matrix_role_viz/<stem>/`` and
``output/unified_diag/matrix_role_viz/_batch_audit.json`` (classification + consolidation fingerprint).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from input.process_input import DocumentData, _extract_image_page  # noqa: E402
from pipeline import process_page_data_cleaning, process_page_extraction  # noqa: E402


DEFAULT_IMAGES = [
    r"C:\Users\benja\OneDrive\Desktop\data\Screenshot 2026-04-08 113432.png",
    r"C:\Users\benja\OneDrive\Desktop\data\Screenshot 2026-04-08 113409.png",
]

ROLE_COLORS = {
    "main_header": (0, 0, 255),      # red
    "sub_header": (0, 165, 255),     # orange
    "row_anchor": (255, 0, 0),       # blue
    "value": (0, 220, 0),            # green
    "matrix_t": (255, 0, 255),       # magenta
}


def _gather_image_paths(data_dir: Path, recursive: bool) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    paths: List[Path] = []
    if recursive:
        for p in sorted(data_dir.rglob("*")):
            if p.is_file() and p.suffix in exts:
                paths.append(p)
    else:
        for p in sorted(data_dir.iterdir()):
            if p.is_file() and p.suffix in exts:
                paths.append(p)
    return [str(p.resolve()) for p in paths]


def _fingerprint_consolidation(consolidation: Dict[str, Any]) -> Dict[str, Any]:
    tables = consolidation.get("tables") or []
    ec2 = [t for t in tables if str(t.get("ec_type") or "").upper() == "EC2"]
    by_type = Counter(str(t.get("ec_type") or "UNKNOWN").upper() for t in tables)
    out: Dict[str, Any] = {
        "n_tables": len(tables),
        "by_ec_type": dict(by_type),
        "ec2_table_count": len(ec2),
    }
    eco: List[Dict[str, Any]] = []
    for t in ec2:
        cells = t.get("cells") or []
        canon = (t.get("canonical_grid") or {}) if isinstance(t.get("canonical_grid"), dict) else {}
        ccells = canon.get("cells") if isinstance(canon.get("cells"), list) else []
        pids = sorted(str(c.get("poly_id") or "") for c in cells if c.get("poly_id"))
        payload = "".join(str(c.get("poly_id")) for c in cells)
        eco.append({
            "table_id": str(t.get("table_id")),
            "grid_shape": t.get("grid_shape"),
            "n_cells": len(cells),
            "n_canonical_cells": len(ccells),
            "poly_sha1": hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16],
        })
    out["ec2_tables"] = eco
    return out


def _fingerprint_classification(classification: Dict[str, Any]) -> Dict[str, Any]:
    clusters = classification.get("clusters") or {}
    if not isinstance(clusters, dict):
        return {"n_clusters": 0, "n_TABLE": 0, "n_INFO_CLUSTER": 0}
    n_table = sum(1 for c in clusters.values() if isinstance(c, dict) and c.get("label") == "TABLE")
    n_info = sum(1 for c in clusters.values() if isinstance(c, dict) and c.get("label") == "INFO_CLUSTER")
    return {"n_clusters": len(clusters), "n_TABLE": n_table, "n_INFO_CLUSTER": n_info}


def _length_like_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hits = []
    for c in cells:
        raw = str(c.get("text") or "")
        folded = "".join(raw.upper().split())
        if "LENGTH" in folded:
            hits.append({"poly_id": c.get("poly_id"), "text": raw[:160]})
    return hits


def _draw_roles_overlay(img_bgr: np.ndarray, cells: List[Dict[str, Any]], out_path: Path) -> None:
    vis = img_bgr.copy()
    for c in cells:
        bb = c.get("bbox") or []
        if len(bb) < 4:
            continue
        role = str(c.get("role") or "")
        color = ROLE_COLORS.get(role, (180, 180, 180))
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    y = 24
    for role, color in ROLE_COLORS.items():
        cv2.putText(vis, role, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 24
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def _run_one_page(path: str, out_root: Path) -> Tuple[Dict[str, Any], Any, Any]:
    """Returns (audit_row_dict, extraction_result_unused, extraction_result_unused2)."""

    audit: Dict[str, Any] = {"image_path": path, "error": None}
    if not os.path.isfile(path):
        audit["error"] = "missing_file"
        return audit, None, None

    doc = DocumentData(
        document_id=f"DOC_{Path(path).stem}",
        batch_id="MATRIX_ROLE_VIZ",
        source_path=path,
        file_size=os.path.getsize(path),
    )
    page = _extract_image_page(doc, path)[0]
    pr_ext = process_page_extraction(page)
    pr = process_page_data_cleaning(page, pr_ext)

    classification = getattr(pr, "classification", None) or {}
    consolidation = getattr(pr, "consolidation", None) or {}

    stem = Path(path).stem.replace(" ", "_")
    image_out = out_root / stem
    image_out.mkdir(parents=True, exist_ok=True)

    audit.update({
        "stem": stem,
        "classification_fingerprint": _fingerprint_classification(classification),
        "consolidation_fingerprint": _fingerprint_consolidation(consolidation),
        "viz_subdir": str(image_out.relative_to(ROOT)),
    })

    tables = consolidation.get("tables", []) or []
    ec2_tables = [t for t in tables if str(t.get("ec_type", "")).upper() == "EC2"]

    summary_tables: List[Dict[str, Any]] = []
    audit["screenshot_matrix_focus"] = {}
    low_name = stem.lower()

    for t in ec2_tables:
        cells = t.get("cells", []) or []
        role_counts: Dict[str, int] = {}
        for c in cells:
            role = str(c.get("role") or "")
            role_counts[role] = role_counts.get(role, 0) + 1

        table_id = str(t.get("table_id") or "table")
        overlay_path = image_out / f"{table_id}_matrix_roles_overlay.jpg"
        img_arr = np.asarray(page.image).copy()
        if len(img_arr.shape) == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
        _draw_roles_overlay(img_arr, cells, overlay_path)

        matrix_records = t.get("matrix_records") or {}
        header_groups = matrix_records.get("header_groups") if isinstance(matrix_records, dict) else []
        explicit_subheaders = 0
        explicit_main_headers = 0
        if isinstance(header_groups, list):
            explicit_main_headers = len(header_groups)
            explicit_subheaders = sum(
                len(g.get("sub_header_poly_ids") or []) for g in header_groups if isinstance(g, dict)
            )

        pids_sorted = sorted(str(c.get("poly_id")) for c in cells if c.get("poly_id"))
        length_cells = _length_like_cells(cells)

        row = {
            "table_id": table_id,
            "grid_shape": t.get("grid_shape"),
            "reading_path": t.get("reading_path"),
            "matrix_orientation": matrix_records.get("orientation") if isinstance(matrix_records, dict) else None,
            "role_counts": role_counts,
            "header_groups_n": len(header_groups) if isinstance(header_groups, list) else 0,
            "explicit_main_headers_from_groups": explicit_main_headers,
            "explicit_subheaders_from_groups": explicit_subheaders,
            "overlay_path": str(overlay_path),
            "n_cells": len(cells),
            "poly_ids_sorted_sample": pids_sorted[:48],
            "length_text_hits_in_cells_n": len(length_cells),
            "length_text_hits_sample": length_cells[:8],
            "corner_poly_tail": [
                pid for pid in pids_sorted
                if "-0016" in pid or "-0017" in pid or pid.endswith("016") or pid.endswith("017")
            ],
        }
        summary_tables.append(row)

        focus = audit["screenshot_matrix_focus"]
        if "113432" in low_name:
            focus["113432_expected_corner_tail"] = row["corner_poly_tail"]
            focus["n_cells_ec2_POLY_0"] = len(cells)
        if "113409" in low_name:
            focus["113409_LENGTH_hits"] = length_cells

    summary = {"image": path, "tables": summary_tables}
    summary_path = image_out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    audit["summary_written"] = str(summary_path)

    for row in summary_tables:
        tid = row["table_id"]
        print(f"    {tid} cells={row['n_cells']} LENGTH_hits={row['length_text_hits_in_cells_n']} roles={row['role_counts']} header_groups={row['header_groups_n']}")
        tail = row.get("corner_poly_tail") or []
        if tail:
            print("    poly tail match -0016/-0017:", tail)

    print("    overlay_dir:", image_out)

    audit["tables"] = summary_tables
    return audit, pr, page


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrix role overlays + batch audit.")
    parser.add_argument("--dir", type=str, help="Process all PNG/JPEG in directory")
    parser.add_argument("--recursive", action="store_true", help="Recurse subfolders (with --dir)")
    parser.add_argument(
        "--consolidation-diag",
        action="store_true",
        help="Also run matrix_consolidation_diagnostic.py per image (TABLE trace + EC2 overlays).",
    )
    parser.add_argument("paths", nargs="*", help="Image paths (unless --dir is set)")
    args = parser.parse_args()

    if args.dir:
        data_dir = Path(args.dir).expanduser().resolve()
        if not data_dir.is_dir():
            print("--dir must be an existing directory:", data_dir)
            sys.exit(1)
        image_paths = _gather_image_paths(data_dir, args.recursive)
        if not image_paths:
            print("No PNG/JPEG in", data_dir)
            sys.exit(1)
    elif args.paths:
        image_paths = list(args.paths)
    else:
        image_paths = list(DEFAULT_IMAGES)

    print("MATRIX_ROLE_VIZ: images=", len(image_paths))

    out_root = ROOT / "output" / "unified_diag" / "matrix_role_viz"
    out_root.mkdir(parents=True, exist_ok=True)

    batch_audit: Dict[str, Any] = {"output_root": str(out_root.resolve()), "n_images": len(image_paths), "audits": []}

    diag_script = ROOT / "tools" / "matrix_consolidation_diagnostic.py"

    for path in image_paths:
        print("\n" + "=" * 72)
        print("FILE:", path)
        audit, _, _ = _run_one_page(path, out_root)
        batch_audit["audits"].append(audit)

        if args.consolidation_diag and diag_script.is_file():
            print("    consolidation_diagnostic...")
            subprocess.check_call([sys.executable, str(diag_script), path], cwd=str(ROOT))

    batch_audit["data_note"] = "Fingerprints characterize this run only; compare poly_sha/class counts to an archived baseline to detect regressions."
    batch_audit_path = out_root / "_batch_audit.json"
    batch_audit_path.write_text(json.dumps(batch_audit, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("BATCH_AUDIT:", batch_audit_path)
    print("Per-image fingerprints (classification + EC2 geometry):")

    fp_lines: List[str] = []
    for a in batch_audit["audits"]:
        if a.get("error"):
            fp_lines.append(f"  FAIL {Path(a['image_path']).name}: {a['error']}")
            continue
        cf = a.get("classification_fingerprint") or {}
        con = (a.get("consolidation_fingerprint") or {})
        eco = "|".join(
            f"{e.get('table_id')}:{e.get('n_cells')}(canon {e.get('n_canonical_cells')})"
            for e in con.get("ec2_tables") or []
        ) or "(no EC2)"
        sha_list = [e.get("poly_sha1") for e in con.get("ec2_tables") or []]
        fp_lines.append(
            f"  {a.get('stem')}: TABLEclusters={cf.get('n_TABLE')} INFO={cf.get('n_INFO_CLUSTER')} "
            f"n_tables={con.get('n_tables')} by_type={con.get('by_ec_type')} EC2=[{eco}] poly_sha={sha_list}"
        )
        sfx = a.get("screenshot_matrix_focus") or {}
        if sfx:
            fp_lines.append(f"       focus_trace: {json.dumps(sfx)}")
    print("\n".join(fp_lines))


if __name__ == "__main__":
    main()
