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
        ):
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frame detection on all images in a folder and save visual overlays beside each image."
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
        frame_result, vis = detect_frame(
            image=image,
            page_id=img_path.stem,
            debug=False,
            basename=None,
            gray=gray,
            scan_branch=None,
        )

        out_path = img_path.with_name(f"{img_path.stem}_frame_overlay{img_path.suffix}")
        cv2.imwrite(str(out_path), vis)

        if "bbox" in frame_result:
            print(
                f"[OK] {img_path.name} -> {out_path.name} "
                f"bbox={frame_result['bbox']} strategy={frame_result.get('strategy')}"
            )
        else:
            print(
                f"[SKIP] {img_path.name} -> {out_path.name} "
                f"reason={frame_result.get('reason', 'unknown')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
