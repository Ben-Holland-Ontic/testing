from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workers import frame_detection as fd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    path = Path(args.image)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    page_w = image.shape[1]
    page_h = image.shape[0]

    cand_a, line_mask, otsu = fd._strategy_a_candidates(gray, scan_branch=None)
    cand_c = fd._strategy_c_candidates(otsu)
    candidates = list(cand_a) + list(cand_c)
    candidates.sort(
        key=lambda c: (
            fd._STRATEGY_PRIORITY.get(c["strategy"], 99),
            c["bbox"][0],
            c["bbox"][1],
            c["bbox"][2],
            c["bbox"][3],
        )
    )

    print(f"Image: {path}")
    print(f"Page size: {page_w}x{page_h}")
    print(f"A candidates: {len(cand_a)}")
    print(f"C candidates: {len(cand_c)}")

    valid = []
    rejected = []
    for candidate in candidates:
        metrics, reason, partial = fd._validate_candidate(candidate, page_w, page_h, line_mask=line_mask)
        if metrics is None:
            rejected.append(
                {
                    "strategy": candidate["strategy"],
                    "bbox": candidate["bbox"],
                    "reason": reason,
                    "metrics": partial or {},
                }
            )
            continue
        c = dict(candidate)
        c["metrics"] = metrics
        valid.append(c)

    if not valid:
        cand_b = fd._strategy_b_candidates(gray)
        cand_b.sort(key=lambda c: (c["bbox"][0], c["bbox"][1], c["bbox"][2], c["bbox"][3]))
        print(f"B fallback candidates: {len(cand_b)}")
        for candidate in cand_b:
            metrics, reason, partial = fd._validate_candidate(candidate, page_w, page_h, line_mask=line_mask)
            if metrics is None:
                rejected.append(
                    {
                        "strategy": candidate["strategy"],
                        "bbox": candidate["bbox"],
                        "reason": reason,
                        "metrics": partial or {},
                    }
                )
            else:
                c = dict(candidate)
                c["metrics"] = metrics
                valid.append(c)

    print(f"Valid candidates: {len(valid)}")
    for i, candidate in enumerate(valid, start=1):
        m = candidate["metrics"]
        print(
            f"  VALID#{i} strategy={candidate['strategy']} bbox={candidate['bbox']} "
            f"line_support={m.get('line_support', 0.0):.4f} total_margin={m.get('total_margin')} symmetry={m.get('symmetry_ratio'):.4f} "
            f"aspect_delta={m.get('aspect_delta', 0.0):.4f} area={m.get('area')} score={m.get('score'):.6f}"
        )

    print(f"Rejected candidates: {len(rejected)}")
    for i, candidate in enumerate(rejected, start=1):
        print(
            f"  REJECT#{i} strategy={candidate['strategy']} bbox={candidate['bbox']} "
            f"reason={candidate['reason']} metrics={candidate['metrics']}"
        )

    if valid:
        best = fd._select_best(valid)
        print(f"Selected candidate: {best['strategy']} {best['bbox']} metrics={best['metrics']}")
    else:
        reason = fd._select_best_rejection(rejected)
        print(f"No valid candidate. Selected reason: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
