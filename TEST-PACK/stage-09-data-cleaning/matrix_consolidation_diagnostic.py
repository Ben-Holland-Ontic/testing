#!/usr/bin/env python3
"""Matrix consolidation diagnostic: TABLE cell outcomes, EC2 pool stats, color overlay.

Writes under ``output/matrix_diag/<stem>/``:
  - ``cell_trace.json`` — per TABLE-labeled input cell outcome
  - ``summary.json`` — counts + EC2 probe on rejected pre-EC pool
  - ``overlay_table_cells.png`` — consolidation outcome for TABLE cells (legend on image)
  - ``overlay_classification_clusters.png`` — classifier labels TABLE vs INFO_CLUSTER (thin boxes)

Run:
  py -3 tools/matrix_consolidation_diagnostic.py [image.png ...]
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(level=logging.WARNING)

from cleaners.data_cleaningv2 import (  # noqa: E402
    EC2_MIN_EVIDENCE_SIGNALS,
    EC_COHESIVENESS_MIN_EC2,
    JUNCTION_TOLERANCE_MAX,
    JUNCTION_TOLERANCE_RATIO,
    PageMetadata,
    TableConsolidator,
    _compute_outer_footprint_shape_scores,
    _ec_junction_tol,
    _debug_mode_log,
    _ec2_matrix_pool_components,
    _junction_connected_components,
    _point_in_bbox,
    build_data_cleaning_inputs_v2,
    classify_document_structure,
    consolidate_tables,
)
from input.process_input import DocumentData, _extract_image_page  # noqa: E402
from pipeline import process_page_extraction  # noqa: E402

DEFAULT_IMAGES = [
    r"C:\Users\benja\Downloads\Screenshot 2026-04-08 113432.png",
    r"C:\Users\benja\Downloads\Screenshot 2026-04-08 113409.png",
]


def _trace_color_bgr(row: Dict[str, Any]) -> tuple:
    if row.get("outcome") == "non_table_cluster":
        return (0, 0, 255)  # red — TABLE cell fell to INFO_CLUSTER path
    if row.get("outcome") != "in_consolidated_table":
        return (0, 255, 255)  # yellow — orphan
    ec = str(row.get("ec_type") or "STANDARD").upper()
    if ec == "EC2":
        return (0, 140, 255)  # orange
    if ec == "EC1":
        return (255, 0, 255)  # magenta
    if ec == "EC4":
        return (0, 200, 255)  # light blue
    return (0, 200, 0)  # green — STANDARD


def _draw_overlay(
    img_bgr: np.ndarray,
    cell_trace: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    vis = img_bgr.copy()
    for row in cell_trace:
        bb = row.get("bbox") or []
        if len(bb) < 4:
            continue
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        color = _trace_color_bgr(row)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    legend_y = 24
    legend = [
        ("STANDARD / cell_grid", (0, 200, 0)),
        ("EC2 / matrix_nested", (0, 140, 255)),
        ("EC1", (255, 0, 255)),
        ("EC4", (0, 200, 255)),
        ("TABLE -> non_table (INFO path)", (0, 0, 255)),
        ("orphan", (0, 255, 255)),
    ]
    for label, col in legend:
        cv2.putText(
            vis,
            label,
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            col,
            2,
        )
        legend_y += 26
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def _draw_cells_overlay(
    img_bgr: np.ndarray,
    cells: List[Dict[str, Any]],
    out_path: Path,
    *,
    label: str,
    color: tuple[int, int, int],
) -> None:
    vis = img_bgr.copy()
    for c in cells:
        bb = c.get("bbox") or []
        if len(bb) < 4:
            continue
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        vis,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def _ec2_pool_diag(
    consolidator: TableConsolidator,
    pid_to_cell: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    page_diag = consolidator._estimate_page_diag()
    bboxes = {pid: tuple(c["bbox"]) for pid, c in pid_to_cell.items()}
    cf = consolidator.classification_results.get("cluster_features", {}) or {}
    seg_data = {}
    try:
        _, _, seg_data = consolidator._segment_junction_data(page_diag, log_failures=False)
    except Exception as exc:
        return {"error": str(exc), "pool_size": len(pid_to_cell), "component_sizes": []}
    seg_t_pts = list(seg_data.get("t_junction_points") or [])
    comps = (
        _ec2_matrix_pool_components(bboxes, page_diag, cf, seg_t_pts)
        if len(bboxes) >= 2
        else []
    )
    if not comps and len(bboxes) >= 4:
        comps = _junction_connected_components(bboxes, page_diag)
    sizes = sorted(len(c) for c in comps) if comps else []
    n_t = len(seg_data.get("t_junction_points") or [])
    n_4 = len(seg_data.get("four_line_points") or [])
    ec2_try: List[Any] = []
    ec2_skip_reason = None
    if len(pid_to_cell) < 6:
        ec2_skip_reason = "pool_size_lt_6_EC2_hard_minimum"
    try:
        ec2_try = consolidator._detect_ec2_tables(pid_to_cell)
    except Exception as exc:
        return {
            "pool_size": len(pid_to_cell),
            "component_sizes": sizes,
            "n_components": len(comps),
            "segment_t_points": n_t,
            "segment_four_points": n_4,
            "ec2_skip_reason": ec2_skip_reason,
            "ec2_detect_error": str(exc),
        }
    out = {
        "pool_size": len(pid_to_cell),
        "component_count": len(comps),
        "component_sizes_sorted": sizes,
        "largest_component": max(sizes) if sizes else 0,
        "components_ge_6": sum(1 for s in sizes if s >= 6),
        "segment_t_points": n_t,
        "segment_four_points": n_4,
        "ec2_candidate_tables": len(ec2_try),
        "ec2_scores": [float(x[0]) for x in ec2_try[:5]],
    }
    if ec2_skip_reason:
        out["ec2_skip_reason"] = ec2_skip_reason
    return out


def _ec2_gate_probe(
    consolidator: TableConsolidator,
    pid_to_cell: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Replicate EC2 gate calculations and report exact rejection causes per component."""
    out: Dict[str, Any] = {
        "pool_size": len(pid_to_cell),
        "hard_pool_gate_pass": len(pid_to_cell) >= 6,
        "thresholds": {
            "min_union_t_points": EC2_MIN_EVIDENCE_SIGNALS,
            "min_cohesiveness": EC_COHESIVENESS_MIN_EC2,
            "min_rows": 2,
            "min_cols": 2,
            "min_hierarchy_pass": 1,
        },
        "t_to_four_adjacency_not_a_gate": True,
        "components": [],
    }
    if len(pid_to_cell) < 6:
        out["global_fail_reason"] = "pool_size_lt_6"
        return out

    page_diag = consolidator._estimate_page_diag()
    bboxes = {pid: tuple(c["bbox"]) for pid, c in pid_to_cell.items()}
    _, _, seg_data = consolidator._segment_junction_data(page_diag, log_failures=False)
    seg_t_pts = list(seg_data.get("t_junction_points") or [])
    tol_j = _ec_junction_tol(page_diag)
    cf = consolidator.classification_results.get("cluster_features", {}) or {}
    components = _ec2_matrix_pool_components(bboxes, page_diag, cf, seg_t_pts)
    if not components:
        components = _junction_connected_components(bboxes, page_diag)

    for idx, comp_raw in enumerate(components):
        comp_pids = [p for p in comp_raw if p in pid_to_cell]
        comp_entry: Dict[str, Any] = {
            "component_idx": idx,
            "n_cells": len(comp_pids),
            "candidate_eligible_min_cells": len(comp_pids) >= 6,
        }
        if len(comp_pids) < 6:
            comp_entry["final_accept"] = False
            comp_entry["fail_reasons"] = ["component_cells_lt_6"]
            out["components"].append(comp_entry)
            continue

        comp_bboxes = {p: bboxes[p] for p in comp_pids}
        comp_bbox = (
            min(bb[0] for bb in comp_bboxes.values()),
            min(bb[1] for bb in comp_bboxes.values()),
            max(bb[2] for bb in comp_bboxes.values()),
            max(bb[3] for bb in comp_bboxes.values()),
        )
        widths = [max(0.0, comp_bboxes[p][2] - comp_bboxes[p][0]) for p in comp_pids]
        heights = [max(0.0, comp_bboxes[p][3] - comp_bboxes[p][1]) for p in comp_pids]
        med_w = float(np.median(widths)) if widths else 1.0
        med_h = float(np.median(heights)) if heights else 1.0
        rect_t_pts = consolidator._rect_corner_t_junction_points(comp_bboxes, tol_j)
        seg_t_in_pts = [(x, y) for (x, y) in seg_t_pts if _point_in_bbox(comp_bbox, (x, y))]
        rect_t_in_pts = [(x, y) for (x, y) in rect_t_pts if _point_in_bbox(comp_bbox, (x, y))]
        tol_t_dedupe = max(1.5, min(3.0, tol_j * 0.6))
        union_t_pts = consolidator._dedupe_points(rect_t_in_pts + seg_t_in_pts, tol_t_dedupe)
        pid_to_rc, row_clusters, col_clusters = consolidator._pid_row_col_for_component(
            comp_pids, comp_bboxes, pid_to_cell
        )
        outer_shape = _compute_outer_footprint_shape_scores(comp_pids, comp_bboxes)
        cohesiveness = float(outer_shape["outer_cohesion_composite"])

        row0_bbs = [comp_bboxes[p] for p in comp_pids if pid_to_rc.get(p, (-1, -1))[0] == 0]
        row1_bbs = [comp_bboxes[p] for p in comp_pids if pid_to_rc.get(p, (-1, -1))[0] == 1]
        col0_bbs = [comp_bboxes[p] for p in comp_pids if pid_to_rc.get(p, (-1, -1))[1] == 0]
        col1_bbs = [comp_bboxes[p] for p in comp_pids if pid_to_rc.get(p, (-1, -1))[1] == 1]
        use_col_header = (len(col0_bbs) + len(col1_bbs)) > (len(row0_bbs) + len(row1_bbs))
        hier = consolidator._ec2_hierarchy_header_and_tri(
            comp_pids,
            comp_bboxes,
            pid_to_rc,
            union_t_pts,
            [],
            tol_j,
            med_w,
            med_h,
            comp_bbox,
            use_col_header,
        )
        header_groups: List[Dict[str, Any]] = list(hier["header_groups"])
        tri_hierarchy_matches = int(hier["tri_hierarchy_matches"])
        tri_length_matches = int(hier["tri_length_matches"])

        hierarchy_pass = 1 if (tri_hierarchy_matches >= 1 or tri_length_matches >= 1) else 0
        if hierarchy_pass < 1 and len(header_groups) >= 1:
            hierarchy_pass = 1

        gates = {
            "union_t_points_ge_min": len(union_t_pts) >= EC2_MIN_EVIDENCE_SIGNALS,
            "cohesiveness_ge_min": cohesiveness >= EC_COHESIVENESS_MIN_EC2,
            "rows_ge_2": len(row_clusters) >= 2,
            "cols_ge_2": len(col_clusters) >= 2,
            "hierarchy_pass_ge_1": hierarchy_pass >= 1,
        }
        fail_reasons = [k for k, v in gates.items() if not v]
        comp_entry.update(
            {
                "rows": len(row_clusters),
                "cols": len(col_clusters),
                "cohesiveness": float(cohesiveness),
                "outer_convexity": float(outer_shape["outer_convexity"]),
                "outer_envelope_fill": float(outer_shape["outer_envelope_fill"]),
                "outer_rect_composite": float(outer_shape["outer_rect_composite"]),
                "outer_cohesion_composite": float(outer_shape["outer_cohesion_composite"]),
                "union_t_points": len(union_t_pts),
                "rect_t_points_in_bbox": len(rect_t_in_pts),
                "seg_t_points_in_bbox": len(seg_t_in_pts),
                "tol_junction": float(tol_j),
                "tol_t_dedupe": float(tol_t_dedupe),
                "tol_header": float(hier.get("tol_header", 0.0)),
                "tri_near_radius": float(hier.get("tri_near_radius", 0.0)),
                "tri_hierarchy_matches": int(tri_hierarchy_matches),
                "tri_length_matches": int(tri_length_matches),
                "n_header_groups": len(header_groups),
                "hierarchy_pass": int(hierarchy_pass),
                "gates": gates,
                "fail_reasons": fail_reasons,
                "final_accept": len(fail_reasons) == 0,
            }
        )
        out["components"].append(comp_entry)
    out["accepted_components"] = sum(1 for c in out["components"] if c.get("final_accept"))
    return out


def _largest_ec2_component_cells(
    consolidator: TableConsolidator,
    pid_to_cell: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Largest EC2 matrix pool (unified OR vertex + OR edge: 4-line | T | aligning)."""
    if not pid_to_cell:
        return []
    bboxes = {pid: tuple(c["bbox"]) for pid, c in pid_to_cell.items() if c.get("bbox")}
    if not bboxes:
        return []
    page_diag = float(consolidator._estimate_page_diag())
    junction_tol = min(
        JUNCTION_TOLERANCE_MAX, max(2.0, JUNCTION_TOLERANCE_RATIO * page_diag)
    )
    cf = consolidator.classification_results.get("cluster_features", {}) or {}
    seg_t_pts: List[Tuple[float, float]] = []
    try:
        _, _, seg_data = consolidator._segment_junction_data(page_diag, log_failures=False)
        seg_t_pts = list((seg_data or {}).get("t_junction_points") or [])
    except Exception:
        seg_t_pts = []
    tol_j = float(_ec_junction_tol(page_diag))
    comps = _ec2_matrix_pool_components(bboxes, page_diag, cf, seg_t_pts)
    if not comps and len(bboxes) >= 4:
        comps = _junction_connected_components(bboxes, page_diag)
    if not comps:
        return []
    best_list = max(comps, key=lambda xs: len(xs or []))
    pset = {str(p) for p in best_list}
    # region agent log
    try:
        all_ids = {str(k) for k in bboxes.keys()}
        missing = sorted(all_ids - pset)
        focus_raw = os.getenv("MATRIX_DIAG_EC2_FOCUS_PIDS", "POLY_5,POLY_136")
        focus = {s.strip() for s in focus_raw.split(",") if s.strip()}
        focus_rows = {}
        for pid in sorted(focus):
            if pid not in pid_to_cell:
                focus_rows[pid] = {"note": "not_in_pid_to_cell"}
                continue
            bb = tuple(pid_to_cell[pid].get("bbox") or ())
            focus_rows[pid] = {
                "in_ec2_localization_pool": pid in pset,
                "bbox": list(bb) if len(bb) == 4 else bb,
            }
        _debug_mode_log(
            run_id=os.getenv("MATRIX_DIAG_RUN_ID", "ec2-pool"),
            hypothesis_id="H_pool_junction_graph",
            location="matrix_consolidation_diagnostic.py:_largest_ec2_component_cells",
            message="EC2 matrix pool (OR: 4-line|T|aligning vertex; OR edge after locality) vs full pid_to_cell",
            data={
                "n_pid_to_cell": len(pid_to_cell),
                "n_bboxes": len(bboxes),
                "page_diag": page_diag,
                "junction_tol_baseline": float(junction_tol),
                "junction_tol_ec": float(_ec_junction_tol(page_diag)),
                "n_components_ge2": len(comps),
                "best_pool_n": len(pset),
                "n_missing_from_pool": len(missing),
                "missing_sample": missing[:40],
                "focus": focus_rows,
            },
        )
    except Exception:
        pass
    # endregion agent log
    return [c for pid, c in pid_to_cell.items() if str(pid) in pset]


def _bbox_for_cluster_poly(
    inputs: Any, classification: Dict[str, Any], poly_id: str
) -> tuple | None:
    for rect in getattr(inputs, "rectangles", None) or []:
        if getattr(rect, "poly_id", None) == poly_id:
            return tuple(rect.bbox)
    for c in getattr(inputs, "contours", None) or []:
        if getattr(c, "poly_id", None) == poly_id:
            return tuple(c.bbox)
    feats = (inputs.cluster_features or {}).get(poly_id, {}) or {}
    pts = feats.get("poly_points") or []
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
    return None


def _draw_classification_clusters_overlay(
    img_bgr: np.ndarray,
    inputs: Any,
    classification: Dict[str, Any],
    out_path: Path,
) -> None:
    vis = img_bgr.copy()
    clusters = classification.get("clusters") or {}
    for pid, row in clusters.items():
        if not isinstance(row, dict):
            continue
        lab = str(row.get("label") or "")
        bb = _bbox_for_cluster_poly(inputs, classification, str(pid))
        if not bb or len(bb) < 4:
            continue
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        if lab == "TABLE":
            color = (180, 80, 80)  # BGR dusty blue-red for TABLE label
        elif lab == "INFO_CLUSTER":
            color = (180, 180, 180)
        else:
            color = (100, 160, 100)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
    # Dark text only — never draw white fills over the raster (confuses line review).
    cv2.putText(
        vis,
        "Classifier: TABLE=dark tint  INFO_CLUSTER=gray",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def run_one(image_path: str, out_root: Path) -> None:
    stem = Path(image_path).stem.replace(" ", "_")
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = DocumentData(
        document_id=f"DIAG_{stem[:20]}",
        batch_id="MATRIX_DIAG_BATCH",
        source_path=image_path,
        file_size=os.path.getsize(image_path),
    )
    page = _extract_image_page(doc, image_path)[0]
    pr = process_page_extraction(page)
    if getattr(pr, "status", None) == "SKIPPED":
        print(image_path, "EXTRACTION SKIPPED", getattr(pr, "errors", None))
        return

    page_meta = PageMetadata(
        page_id=page.page_id,
        width=float(page.width),
        height=float(page.height),
        batch_id=page.batch_id,
        document_id=page.document_id,
    )
    inputs = build_data_cleaning_inputs_v2(
        rect_data=pr.rect_data,
        cluster_results=pr.cluster_results,
        region_results=pr.region_results,
        page_meta=page_meta,
    )
    classification = classify_document_structure(inputs)
    consolidator = TableConsolidator(inputs, classification)
    cons = consolidate_tables(inputs, classification, return_cell_trace=True)

    cell_trace = cons.get("cell_trace") or []
    trace_summary = cons.get("trace_summary") or {}
    rejected_pre: Set[str] = set(cons.get("rejected_poly_ids_pre_ec") or [])

    table_cells = consolidator._extract_table_cells()
    pid_to_cell_all = {c["poly_id"]: c for c in table_cells if c.get("poly_id")}
    pid_to_cell = {c["poly_id"]: c for c in table_cells if c.get("poly_id") in rejected_pre}

    summary: Dict[str, Any] = {
        "image": image_path,
        "trace_summary": trace_summary,
        "tables": [
            {
                "table_id": t.get("table_id"),
                "ec_type": t.get("ec_type"),
                "reading_path": t.get("reading_path"),
                "grid_shape": t.get("grid_shape"),
                "n_cells": len(t.get("cells") or []),
            }
            for t in (cons.get("tables") or [])
        ],
        "n_non_table_clusters": len(cons.get("non_table_clusters") or []),
        "ec2_pool_on_rejected_pre_ec": _ec2_pool_diag(consolidator, pid_to_cell),
        "ec2_gate_probe_on_rejected_pre_ec": _ec2_gate_probe(consolidator, pid_to_cell),
        "ec2_pool_on_all_table_cells": _ec2_pool_diag(consolidator, pid_to_cell_all),
        "ec2_gate_probe_on_all_table_cells": _ec2_gate_probe(consolidator, pid_to_cell_all),
        "classification_table_clusters": sum(
            1
            for c in (classification.get("clusters") or {}).values()
            if isinstance(c, dict) and c.get("label") == "TABLE"
        ),
        "consolidation_arbitration": cons.get("consolidation_arbitration") or {},
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "cell_trace.json").write_text(
        json.dumps(cell_trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Page raster from ingest is BGR (OpenCV).
    img = np.asarray(page.image)
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img_bgr = img.copy()

    _draw_overlay(img_bgr, cell_trace, out_dir / "overlay_table_cells.png")
    # Standard-only overlay for side-by-side visual comparison.
    std_cells = [
        {
            "bbox": row.get("bbox"),
        }
        for row in cell_trace
        if row.get("outcome") == "in_consolidated_table"
        and str(row.get("ec_type") or "STANDARD").upper() == "STANDARD"
    ]
    _draw_cells_overlay(
        img_bgr,
        std_cells,
        out_dir / "overlay_standard_cells.png",
        label="STANDARD retained cells",
        color=(0, 200, 0),
    )

    # Best EC2 candidate overlay from all TABLE-labeled cells.
    ec2_candidates_all = consolidator._detect_ec2_tables(pid_to_cell_all)
    best_ec2_cells: List[Dict[str, Any]] = []
    best_ec2_score = None
    ec2_best_label = "EC2 best accepted candidate cells (none — gates failed)"
    if ec2_candidates_all:
        best_score, best_table, _used = max(ec2_candidates_all, key=lambda t: float(t[0]))
        best_ec2_score = float(best_score)
        best_ec2_cells = [{"bbox": list(c.bbox)} for c in (best_table.cells or [])]
        ec2_best_label = f"EC2 best accepted candidate (n={len(best_ec2_cells)}, score={best_ec2_score:.3f})"
    else:
        # When EC2 fails internal gates, `_detect_ec2_tables` returns nothing — still show the
        # same largest pre-gate component so overlays are not a blank "best" view.
        best_ec2_cells = _largest_ec2_component_cells(consolidator, pid_to_cell_all)
        ec2_best_label = (
            f"EC2 no accepted table — showing largest pre-gate pool (n={len(best_ec2_cells)})"
        )
    _draw_cells_overlay(
        img_bgr,
        best_ec2_cells,
        out_dir / "overlay_ec2_best_candidate.png",
        label=ec2_best_label,
        color=(0, 140, 255),
    )
    # EC2 pre-gate pool overlay (largest component before shape/hierarchy gating).
    ec2_pool_cells = _largest_ec2_component_cells(consolidator, pid_to_cell_all)
    _draw_cells_overlay(
        img_bgr,
        ec2_pool_cells,
        out_dir / "overlay_ec2_pregate_pool.png",
        label=f"EC2 pre-gate pool cells (n={len(ec2_pool_cells)})",
        color=(0, 100, 255),
    )

    _draw_classification_clusters_overlay(
        img_bgr, inputs, classification, out_dir / "overlay_classification_clusters.png"
    )

    # Console digest
    print("\n" + "=" * 72)
    print("IMAGE:", image_path)
    print("OUT:", out_dir)
    print("tables:", len(cons.get("tables") or []))
    for t in cons.get("tables") or []:
        print(
            f"  {t.get('table_id')} ec_type={t.get('ec_type')} reading_path={t.get('reading_path')} "
            f"grid={t.get('grid_shape')} cells={len(t.get('cells') or [])}"
        )
    print("trace_summary:", json.dumps(trace_summary, indent=2))
    print(
        "consolidation_arbitration:",
        json.dumps(summary.get("consolidation_arbitration") or {}, indent=2),
    )
    print("EC2 pool diag:", json.dumps(summary["ec2_pool_on_rejected_pre_ec"], indent=2))
    print("EC2 gate probe:", json.dumps(summary["ec2_gate_probe_on_rejected_pre_ec"], indent=2))
    print("EC2 pool diag (all TABLE):", json.dumps(summary["ec2_pool_on_all_table_cells"], indent=2))
    print("EC2 gate probe (all TABLE):", json.dumps(summary["ec2_gate_probe_on_all_table_cells"], indent=2))
    cnt = Counter()
    for r in cell_trace:
        if r.get("outcome") == "in_consolidated_table":
            cnt[str(r.get("ec_type") or "?")] += 1
        else:
            cnt[r.get("outcome") or "?"] += 1
    print("cell_trace rollup:", dict(cnt))


def main() -> None:
    paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_IMAGES
    out_root = ROOT / "output" / "matrix_diag"
    for p in paths:
        if not os.path.isfile(p):
            print("missing:", p)
            continue
        run_one(p, out_root)


if __name__ == "__main__":
    main()
