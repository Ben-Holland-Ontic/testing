#!/usr/bin/env python3
"""OCR baseline capture + differential comparison.

Phase -1 of the Paddle substitution. Captures the OCR-level output of the current
engine over a corpus, so a later engine swap can be diffed against it.

This is a DIFFERENTIAL harness: the captured run is the *reference*, not ground
truth. "Lost strings" is the regression signal that gates ship; "new strings" are
candidate gains that need human review, not diffs to minimise.

Why not reuse tools/generate_desktop_data_artifacts.py directly: that runs the same
pipeline but only persists relationship-resolution results (extraction_results.json),
which is far downstream of the OCR lines. This captures cluster_results /
region_results - the actual engine output.

Usage:
  py tools/ocr_baseline_compare.py capture --data-dir "C:\\...\\data" --label paddle_baseline
  py tools/ocr_baseline_compare.py compare --a paddle_baseline --b parseq_cascade
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "output" / "ocr_baseline"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# Keys under cluster_results / region_results that hold OCR line dicts.
LINE_KEYS = ("Cluster_Line", "Region_Line", "cluster_lines", "region_lines", "lines")


# ---------------------------------------------------------------- capture ----

def _image_paths(data_dir: Path) -> List[Path]:
    return sorted(p for p in data_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _harvest_lines(blob: Any) -> List[Dict[str, Any]]:
    """Pull OCR line dicts out of a cluster/region results blob, tolerantly."""
    out: List[Dict[str, Any]] = []
    if not isinstance(blob, dict):
        return out
    for key in LINE_KEYS:
        v = blob.get(key)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and "text" in item:
                    out.append({
                        "text": item.get("text") or "",
                        "text_norm": item.get("text_norm") or item.get("TEXT") or "",
                        "bbox": item.get("bbox") or item.get("BBOX"),
                        "ocr_label": item.get("ocr_label") or item.get("text_kind"),
                        "poly_id": item.get("poly_id") or item.get("REGION_ID"),
                        "block_id": item.get("block_id"),
                        "confidence": item.get("confidence", item.get("OCR_CONF")),
                        "_src": key,
                    })
    return out


def _env_meta() -> Dict[str, Any]:
    def ver(mod: str) -> str:
        try:
            m = __import__(mod)
            return getattr(m, "__version__", "?")
        except Exception as e:
            return f"<{type(e).__name__}>"

    meta: Dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "versions": {m: ver(m) for m in ("numpy", "cv2", "torch", "doctr", "onnxruntime")},
        "env": {k: os.environ.get(k) for k in (
            "OCULI_GPU_MODE", "OCULI_PADDLE_SUBPROCESS", "OCULI_PADDLE_SHARED",
            "PADDLEOCR_USE_GPU", "PADDLEOCR_CROP_THREADS", "OCULI_OCR_ENGINE",
        )},
    }
    # torch device is the load-bearing detail for timing comparability
    try:
        import torch
        meta["torch_cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        meta["torch_cuda"] = None
    try:
        meta["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        meta["nvidia_smi"] = None
    return meta


def cmd_capture(args: argparse.Namespace) -> int:
    from input.process_input import DocumentData, _extract_image_page
    from pipeline import process_page_extraction

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _image_paths(data_dir)
    if not images:
        print(f"No images in {data_dir}", file=sys.stderr)
        return 2

    print(f"[CAPTURE] label={args.label}  images={len(images)}  -> {out_dir}")
    snapshot: Dict[str, Any] = {"label": args.label, "data_dir": str(data_dir),
                                "meta": _env_meta(), "pages": {}}

    for p in images:
        t0 = time.perf_counter()
        print(f"  [RUN] {p.name}", flush=True)
        try:
            doc = DocumentData(
                document_id=p.name,
                batch_id="OCR_BASELINE",
                source_path=str(p),
                file_size=p.stat().st_size,
            )
            page = _extract_image_page(doc, str(p))[0]
            res = process_page_extraction(page)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            snapshot["pages"][p.name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        dt = time.perf_counter() - t0

        cluster = _harvest_lines(res.cluster_results)
        region = _harvest_lines(res.region_results)
        rects = ((res.rect_data or {}).get("rectangles") or []) if res.rect_data else []
        snapshot["pages"][p.name] = {
            "page_id": res.page_id,
            "status": res.status,
            "seconds": round(dt, 3),
            "extraction_timing": res.extraction_timing,
            "counts": {"cluster_lines": len(cluster),
                       "region_lines": len(region),
                       "rectangles": len(rects)},
            "cluster_lines": cluster,
            "region_lines": region,
        }
        print(f"    ok  {dt:6.2f}s  cluster={len(cluster)} region={len(region)} rects={len(rects)}")

    path = out_dir / "snapshot.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    print(f"[CAPTURE] wrote {path}")
    _print_capture_summary(snapshot)

    # Charset audit (OCULI_OCR_CHARSET_AUDIT=1). Only meaningful with the paddle
    # subprocess disabled, since the counter lives in whichever process filters.
    if os.environ.get("OCULI_OCR_CHARSET_AUDIT", "0").strip() == "1":
        try:
            import workers.cluster_ocr as _co
        except ImportError:
            sys.path.insert(0, str(ROOT / "workers"))
            import cluster_ocr as _co  # type: ignore
        audit = _co.dump_charset_audit(str(out_dir / "charset_audit.json"))
        print(f"\n[CHARSET-AUDIT] distinct dropped chars: {audit.get('distinct_dropped', 0)}  "
              f"total dropped: {audit.get('total_dropped', 0)}")
        for ch, n in list(audit.get("dropped_char_counts", {}).items())[:30]:
            print(f"    {ch!r}  U+{ord(ch):04X}  x{n}")
    return 0


def _print_capture_summary(snap: Dict[str, Any]) -> None:
    tot_c = tot_r = 0
    labels: Dict[str, int] = {}
    for name, pg in snap["pages"].items():
        if "error" in pg:
            continue
        tot_c += pg["counts"]["cluster_lines"]
        tot_r += pg["counts"]["region_lines"]
        for ln in pg["cluster_lines"] + pg["region_lines"]:
            k = ln.get("ocr_label") or "unknown"
            labels[k] = labels.get(k, 0) + 1
    print(f"\n  TOTAL cluster_lines={tot_c}  region_lines={tot_r}")
    print(f"  ocr_label histogram: {dict(sorted(labels.items(), key=lambda x: -x[1]))}")
    confs = [ln.get("confidence") for pg in snap["pages"].values() if "error" not in pg
             for ln in pg["cluster_lines"] + pg["region_lines"]]
    known = [c for c in confs if isinstance(c, (int, float))]
    if known:
        uniq = sorted(set(round(float(c), 4) for c in known))
        print(f"  confidence: n={len(known)}/{len(confs)}  distinct={len(uniq)}  "
              f"min={min(known):.3f} max={max(known):.3f}")
        if len(uniq) <= 3:
            print(f"  ** DEGENERATE confidence (values={uniq}) - Phase 1 plumbing not done **")
    else:
        print(f"  confidence: none carried ({len(confs)} lines)")


# ---------------------------------------------------------------- compare ----

def _norm(s: str) -> str:
    return " ".join((s or "").split()).strip().casefold()


def _page_strings(pg: Dict[str, Any]) -> List[str]:
    return [_norm(l["text"]) for l in pg.get("cluster_lines", []) + pg.get("region_lines", [])
            if _norm(l.get("text", ""))]


def _match(a: List[str], b: List[str], thresh: float = 0.90) -> Tuple[List[str], List[str], List[Tuple[str, str, float]]]:
    """Greedy fuzzy match. Returns (lost, new, matched_triples)."""
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        exact = set(b)
        lost = [s for s in a if s not in exact]
        new = [s for s in b if s not in set(a)]
        return lost, new, [(s, s, 1.0) for s in a if s in exact]

    remaining = list(b)
    lost, matched = [], []
    for s in a:
        if not remaining:
            lost.append(s)
            continue
        hit = process.extractOne(s, remaining, scorer=fuzz.ratio)
        if hit and hit[1] / 100.0 >= thresh:
            matched.append((s, hit[0], hit[1] / 100.0))
            remaining.pop(hit[2])
        else:
            lost.append(s)
    return lost, remaining, matched


def cmd_compare(args: argparse.Namespace) -> int:
    base = Path(args.out).expanduser().resolve()
    sa = json.loads((base / args.a / "snapshot.json").read_text(encoding="utf-8"))
    sb = json.loads((base / args.b / "snapshot.json").read_text(encoding="utf-8"))

    print(f"=== COMPARE  A={args.a}  B={args.b} ===")
    for tag, s in (("A", sa), ("B", sb)):
        m = s.get("meta", {})
        print(f"  {tag}: {m.get('captured_at')}  torch_cuda={m.get('torch_cuda')}  "
              f"gpu={m.get('gpu')}  versions={m.get('versions')}")
    print()

    hdr = f"{'image':<34} {'A':>5} {'B':>5} {'lost':>5} {'new':>5} {'recall':>7} {'A s':>7} {'B s':>7}"
    print(hdr); print("-" * len(hdr))

    tot_lost: List[Tuple[str, str]] = []
    tot_new: List[Tuple[str, str]] = []
    n_a = n_b = 0
    for name in sorted(set(sa["pages"]) | set(sb["pages"])):
        pa, pb = sa["pages"].get(name, {}), sb["pages"].get(name, {})
        A, B = _page_strings(pa), _page_strings(pb)
        lost, new, matched = _match(A, B, args.threshold)
        recall = (len(matched) / len(A)) if A else 1.0
        n_a += len(A); n_b += len(B)
        tot_lost += [(name, s) for s in lost]
        tot_new += [(name, s) for s in new]
        print(f"{name[:34]:<34} {len(A):>5} {len(B):>5} {len(lost):>5} {len(new):>5} "
              f"{recall:>6.1%} {pa.get('seconds', 0):>6.2f}s {pb.get('seconds', 0):>6.2f}s")

    print("-" * len(hdr))
    overall = 1 - (len(tot_lost) / n_a) if n_a else 1.0
    print(f"{'TOTAL':<34} {n_a:>5} {n_b:>5} {len(tot_lost):>5} {len(tot_new):>5} {overall:>6.1%}")
    print()
    print(f"RECALL vs baseline : {overall:.2%}  ({n_a - len(tot_lost)}/{n_a} strings retained)")
    print(f"LOST   (regression): {len(tot_lost)}")
    print(f"NEW    (review)    : {len(tot_new)}")

    if tot_lost:
        print(f"\n--- LOST strings (first {args.show}) ---")
        for name, s in tot_lost[:args.show]:
            print(f"  [{name[:26]:<26}] {s!r}")
    if tot_new and args.show_new:
        print(f"\n--- NEW strings (first {args.show}) ---")
        for name, s in tot_new[:args.show]:
            print(f"  [{name[:26]:<26}] {s!r}")

    print(f"\nGATE: {'PASS - no lost strings' if not tot_lost else 'FAIL - ' + str(len(tot_lost)) + ' lost strings need review'}")
    return 0 if not tot_lost else 1


# ------------------------------------------------------------------- main ----

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="run the pipeline and snapshot OCR output")
    c.add_argument("--data-dir", required=True)
    c.add_argument("--label", required=True)
    c.add_argument("--out", default=str(DEFAULT_OUT))
    c.set_defaults(func=cmd_capture)

    d = sub.add_parser("compare", help="diff two snapshots")
    d.add_argument("--a", required=True, help="baseline label")
    d.add_argument("--b", required=True, help="candidate label")
    d.add_argument("--out", default=str(DEFAULT_OUT))
    d.add_argument("--threshold", type=float, default=0.90)
    d.add_argument("--show", type=int, default=40)
    d.add_argument("--show-new", action="store_true")
    d.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
