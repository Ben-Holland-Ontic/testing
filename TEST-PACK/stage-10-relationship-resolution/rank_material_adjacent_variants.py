import json
from pathlib import Path


REPORT = Path("debug_outputs/rect_crop_metrics/BATCH_08962480_thin_rect_metrics.json")


def main() -> int:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = []
    for doc in data.get("docs", []):
        for c in doc.get("candidates", []):
            if c.get("row_group") != "material_adjacent":
                continue
            metrics = c.get("metrics") or {}
            variants = c.get("variant_metrics") or {}
            base = float((variants.get("baseline") or metrics or {}).get("thin_score", 0.0))
            best = "baseline"
            best_score = base
            for name, vals in variants.items():
                s = float((vals or {}).get("thin_score", 0.0))
                if s > best_score:
                    best_score = s
                    best = name
            rows.append(
                {
                    "doc": doc.get("doc_id", ""),
                    "poly": c.get("poly_id", ""),
                    "cell": c.get("cell_id", ""),
                    "text": (c.get("text") or "").replace("\n", " ")[:40],
                    "base": base,
                    "best": best,
                    "best_score": best_score,
                    "delta": best_score - base,
                }
            )
    rows.sort(key=lambda x: (x["delta"], x["best_score"]), reverse=True)
    print(f"count\t{len(rows)}")
    print("doc\tpoly\tcell\tbase\tbest\tbest_score\tdelta\ttext")
    for r in rows:
        print(
            f"{r['doc']}\t{r['poly']}\t{r['cell']}\t"
            f"{r['base']:.3f}\t{r['best']}\t{r['best_score']:.3f}\t{r['delta']:.3f}\t{r['text']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

