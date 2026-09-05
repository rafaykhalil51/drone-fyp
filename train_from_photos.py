"""
train_from_photos.py
--------------------
One command after you drop training photos:

    python train_from_photos.py

What it does
  1. Reads images (and optional videos) from training_photos/
  2. Deduplicates near-identical frames
  3. Writes dataset/images + draft YOLO labels
  4. Trains yolov8n on cap / mask / glasses / headphones
  5. Installs the result as models/accessory_best.pt

The dashboard then uses that file automatically. Upload a video on the
portal after training finishes.

Draft labels come from YOLO-World. They are a starting point, not perfect
ground truth. More varied photos (especially masks) produce a better model.

Put photos here:
    <project>/training_photos/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
INCOMING = PROJECT_ROOT / "training_photos"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def count_media(folder: Path) -> tuple[int, int]:
    images = videos = 0
    if not folder.is_dir():
        return 0, 0
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXT:
            images += 1
        elif ext in VIDEO_EXT:
            videos += 1
    return images, videos


def main() -> int:
    p = argparse.ArgumentParser(description="Train accessory_best.pt from dropped photos.")
    p.add_argument("--src", default=str(INCOMING),
                   help="folder of training photos/videos")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--check-only", action="store_true",
                   help="survey the folder and stop (no training)")
    args = p.parse_args()

    src = Path(args.src).expanduser()
    src.mkdir(parents=True, exist_ok=True)
    n_img, n_vid = count_media(src)

    print(f"Training photos folder: {src}")
    print(f"  images: {n_img}")
    print(f"  videos: {n_vid}")

    if n_img + n_vid == 0:
        print("\nNo photos yet. Copy your training images into:")
        print(f"  {src}")
        print("\nInclude people wearing:")
        print("  - caps / hats / beanies")
        print("  - face masks")
        print("  - glasses")
        print("  - headphones")
        print("Then run this script again.")
        return 1

    if n_img + n_vid < 20:
        print("\nWARNING: fewer than 20 files. The model will overfit and")
        print("will not generalise well to a new video. Add more photos if you can.")

    from prepare_dataset import extract, survey

    if args.check_only:
        return survey(src, min_distance=8)

    print("\n--- extracting and draft-labelling ---")
    rc = extract(src, min_distance=8, val_split=0.2, autolabel=True, force=True)
    if rc != 0:
        return rc

    print("\n--- training ---")
    from train_accessory_model import train as run_train

    class NS:
        data = str(PROJECT_ROOT / "dataset" / "data.yaml")
        base = "yolov8n.pt"
        epochs = args.epochs
        imgsz = args.imgsz
        batch = args.batch
        device = None
        name = "accessory_from_photos"
        patience = 20
        degrees = 5.0
        scale = 0.5
        resume = False
        no_install = False

    return run_train(NS())


if __name__ == "__main__":
    sys.exit(main())
