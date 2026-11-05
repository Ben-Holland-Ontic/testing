"""Human-verified regions must reach the COCO detector dataset.

The review chain is: review_data -> annotations/<doc>.json -> the frontend edits and saves
it -> `finalize` marks every region human_verified and sets graph["verified"].

Before this, `assemble_session_to_coco` read nodes.parquet, which the graph exporter writes
from the MACHINE's own output with "human_verified": False hardcoded -- it never looked at
the review graph. A reviewer could move a table box or correct its class, verify the
document, and the detector would still be trained on the original machine box. The edits
reached the synthetic pages and stopped there.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY TESTING\\tests\\test_verified_coco.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, check_eq, main  # noqa: E402

SIDECAR = r"C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\LayoutML sidecar"
if SIDECAR not in sys.path:
    sys.path.insert(0, SIDECAR)


def _png(path: Path, w=400, h=300):
    import cv2
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), 255, np.uint8))
    return w, h


def _session(tmp: Path, verified=True, regions=None, doc="DOC1", page="DOC1_PG001"):
    """A session laid out the way `finalize` leaves one."""
    img = tmp / "review_data" / "rasters" / (page + ".png")
    w, h = _png(img)
    ann = tmp / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    graph = {
        "document_id": doc,
        "verified": verified,
        "pages": [{
            "page_id": page,
            "width": w,
            "height": h,
            "image_path": str(img),
            "regions": regions if regions is not None else [
                {"id": "R1", "class": "TITLE_BLOCK", "bbox": [10, 10, 110, 60],
                 "confidence": 0.8, "human_verified": True},
            ],
        }],
    }
    (ann / (doc + ".json")).write_text(json.dumps(graph), encoding="utf-8")
    return tmp


def test_verified_graph_becomes_coco():
    import dataset_assembler as DA

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sess = _session(tmp / "sessions" / "run1")
        out = DA.assemble_verified_to_coco(sess, tmp / "out")
        if not check("a verified session produces COCO", out is not None):
            return
        coco = json.loads(Path(out).read_text(encoding="utf-8"))

        check_eq("one image", len(coco["images"]), 1)
        check_eq("one annotation", len(coco["annotations"]), 1)
        check("the image is flagged human_verified", coco["images"][0]["human_verified"])
        ann = coco["annotations"][0]
        check_eq("bbox is COCO xywh, not xyxy", ann["bbox"], [10.0, 10.0, 100.0, 50.0])
        check_eq("area matches", ann["area"], 5000.0)
        check_eq("class maps to the detector category",
                 ann["category_id"], DA.CLASS_TO_ID["TITLE_BLOCK"])
        check_eq("a verified region is ground truth, not hedged", ann["confidence"], 1.0)
        check("the annotation records human_verified", ann["human_verified"])
        check("the image file was copied next to the labels",
              (Path(tmp / "out") / coco["images"][0]["file_name"]).exists())


def test_an_unverified_graph_is_ignored():
    """Half-finished edits are worse to train on than the machine's own guess."""
    import dataset_assembler as DA

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sess = _session(tmp / "sessions" / "run1", verified=False)
        check_eq("no pages are loaded from an unverified graph",
                 DA.load_verified_pages(sess), {})
        check("...and no COCO is produced",
              DA.assemble_verified_to_coco(sess, tmp / "out") is None)


def test_the_reviewers_edits_are_what_lands():
    """The exact case: a reviewer moves a box and corrects its class."""
    import dataset_assembler as DA

    edited = [
        # moved from where the machine put it, AND relabelled
        {"id": "R1", "class": "REVISION_TABLE", "bbox": [200, 150, 380, 260],
         "confidence": 0.2, "human_verified": True},
    ]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sess = _session(tmp / "sessions" / "run1", regions=edited)
        out = DA.assemble_verified_to_coco(sess, tmp / "out")
        coco = json.loads(Path(out).read_text(encoding="utf-8"))
        ann = coco["annotations"][0]
        check_eq("the reviewer's BOX is what lands", ann["bbox"],
                 [200.0, 150.0, 180.0, 110.0])
        check_eq("the reviewer's CLASS is what lands",
                 ann["category_id"], DA.CLASS_TO_ID["REVISION_TABLE"])
        check("the machine's low confidence does not follow it",
              ann["confidence"] == 1.0)


def test_verified_replaces_machine_never_duplicates():
    """A moved box must not appear twice -- replacement, not union."""
    import dataset_assembler as DA

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        machine = {
            "images": [
                {"id": 1, "file_name": "images/a.png", "width": 400, "height": 300,
                 "page_id": "DOC1_PG001", "doc_id": "DOC1", "session_id": "run1"},
                {"id": 2, "file_name": "images/b.png", "width": 400, "height": 300,
                 "page_id": "DOC9_PG001", "doc_id": "DOC9", "session_id": "run1"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [10, 10, 100, 50], "area": 5000, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 1,
                 "bbox": [5, 5, 50, 50], "area": 2500, "iscrowd": 0},
            ],
            "categories": [{"id": i, "name": n} for n, i in DA.CLASS_TO_ID.items()],
        }
        mpath = tmp / "machine.json"
        mpath.write_text(json.dumps(machine), encoding="utf-8")

        sess = _session(tmp / "sessions" / "run1", regions=[
            {"id": "R1", "class": "REVISION_TABLE", "bbox": [200, 150, 380, 260],
             "confidence": 0.2, "human_verified": True}])
        vpath = DA.assemble_verified_to_coco(sess, tmp / "out")

        merged_path = DA.merge_verified_over_machine(mpath, vpath, tmp / "merged.json")
        merged = json.loads(Path(merged_path).read_text(encoding="utf-8"))

        pages = [i["page_id"] for i in merged["images"]]
        check_eq("the verified page appears exactly once", pages.count("DOC1_PG001"), 1)
        check_eq("the untouched machine page survives", pages.count("DOC9_PG001"), 1)
        check_eq("two pages total, no duplication", len(merged["images"]), 2)

        by_page = {i["id"]: i["page_id"] for i in merged["images"]}
        verified_anns = [a for a in merged["annotations"]
                         if by_page[a["image_id"]] == "DOC1_PG001"]
        check_eq("the verified page carries the REVIEWER's single box",
                 len(verified_anns), 1)
        check_eq("...at the reviewer's coordinates", verified_anns[0]["bbox"],
                 [200.0, 150.0, 180.0, 110.0])
        check("ids are unique after the merge",
              len({a["id"] for a in merged["annotations"]}) == len(merged["annotations"]))
        check("image ids are unique after the merge",
              len({i["id"] for i in merged["images"]}) == len(merged["images"]))


def test_merge_degrades_sensibly():
    import dataset_assembler as DA

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check("nothing in, nothing out",
              DA.merge_verified_over_machine(None, None, tmp / "x.json") is None)

        payload = {"images": [], "annotations": [], "categories": []}
        p = tmp / "only.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        check("machine only is passed through",
              DA.merge_verified_over_machine(p, None, tmp / "m.json") is not None)
        check("verified only is passed through",
              DA.merge_verified_over_machine(None, p, tmp / "v.json") is not None)


def test_regions_outside_the_detector_classes_are_dropped_not_guessed():
    import dataset_assembler as DA

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sess = _session(tmp / "sessions" / "run1", regions=[
            {"id": "R1", "class": "SOMETHING_ELSE", "bbox": [10, 10, 110, 60]},
            {"id": "R2", "class": "TITLE_BLOCK", "bbox": [10, 80, 110, 130]},
            {"id": "R3", "class": "TITLE_BLOCK", "bbox": [0, 0, 0, 0]},   # degenerate
        ])
        out = DA.assemble_verified_to_coco(sess, tmp / "out")
        coco = json.loads(Path(out).read_text(encoding="utf-8"))
        check_eq("only the valid, in-vocabulary region survives",
                 len(coco["annotations"]), 1)
        check_eq("...and it is the right one", coco["annotations"][0]["node_id"], "R2")


if __name__ == "__main__":
    sys.exit(main(sys.modules[__name__]))
