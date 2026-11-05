from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workers.frame_detection import detect_frame


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_images(folder: Path):
    for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and "_frame_overlay" not in path.stem.lower()
            and "_frame_inset_overlay" not in path.stem.lower()
        ):
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize detected frame and inset ROI used by rectangle detection."
    )
    parser.add_argument(
        "--input-dir",
        default=r"C:\Users\benja\OneDrive\Desktop\data",
        help="Folder containing images to process.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = list(_iter_images(input_dir))
    if not image_paths:
        print(f"No supported image files found in: {input_dir}")
        return 0

    print(f"Processing {len(image_paths)} image(s) from: {input_dir}")
    for img_path in image_paths:
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[SKIP] Could not read: {img_path.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_result, _ = detect_frame(
            image=image,
            page_id=img_path.stem,
            debug=False,
            basename=None,
            gray=gray,
            scan_branch=None,
        )

        vis = image.copy()
        if "bbox" in frame_result:
            frame_bbox = frame_result["bbox"]
            outer_bbox = frame_result.get("outer_bbox", frame_bbox)
            inset_px = int(frame_result.get("inset_px", 0) or 0)
            inset_bbox = frame_bbox
            fx, fy, fw, fh = outer_bbox
            ix, iy, iw, ih = inset_bbox

            cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)  # outer frame edge
            cv2.rectangle(vis, (ix, iy), (ix + iw, iy + ih), (255, 0, 0), 2)  # inner-border ROI

            cv2.putText(
                vis,
                f"frame={fw}x{fh} strat={frame_result.get('strategy', '?')}",
                (max(10, fx + 5), max(20, fy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                vis,
                f"inset_px={inset_px} roi={iw}x{ih}",
                (max(10, fx + 5), max(40, fy + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
            )
            msg = (
                f"[OK] {img_path.name} -> inner_shift={inset_px}px "
                f"outer_frame={outer_bbox} inner_roi={inset_bbox}"
            )
        else:
            cv2.putText(
                vis,
                f"No frame: {frame_result.get('reason', 'unknown')}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            msg = f"[SKIP] {img_path.name} -> reason={frame_result.get('reason', 'unknown')}"

        out_path = img_path.with_name(f"{img_path.stem}_frame_inset_overlay{img_path.suffix}")
        cv2.imwrite(str(out_path), vis)
        print(msg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
