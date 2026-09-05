"""
prepare_dataset.py
------------------
Turns raw video/photos into a labelling-ready image set for
train_accessory_model.py, and optionally proposes CANDIDATE labels.

It does NOT produce a trained model, and it cannot: YOLO needs bounding-box
annotations, and raw footage has none. What it does is remove the tedious
parts around labelling.

Two stages:

  --survey     Report what is in the source media: how many visually
               DISTINCT frames exist (consecutive frames of a static camera
               are near-duplicates and add nothing), and an estimate of how
               many instances of each accessory class are present.

  --extract    Write deduplicated frames into dataset/images/{train,val}
               and, with --autolabel, candidate YOLO .txt labels beside
               them in dataset/labels/{train,val}.

CANDIDATE LABELS ARE A DRAFT, NOT GROUND TRUTH.
They come from YOLO-World, which is untrained on these classes. Training on
them unreviewed would teach the new model YOLO-World's mistakes, including
its inability to see masks and headphones. Open them in a labelling tool
(Roboflow / CVAT / labelImg), fix them, then train.

Examples
--------
  python prepare_dataset.py --survey  --src "C:/Users/User/Desktop/Fyp Photos"
  python prepare_dataset.py --extract --src "C:/Users/User/Desktop/Fyp Photos" --autolabel
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"

CLASS_INDEX = {"cap": 0, "mask": 1, "glasses": 2, "headphones": 3}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}


# ── perceptual de-duplication ────────────────────────────────────────────
def dhash(image, size: int = 8) -> np.ndarray:
    """Difference hash: cheap perceptual fingerprint of a frame."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def collect_sources(src: Path) -> tuple[list[Path], list[Path]]:
    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    return images, videos


def distinct_video_frames(video: Path, min_distance: int, stride: int = 1):
    """Yield (frame_index, frame) for frames that differ perceptually."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  cannot open {video.name}")
        return

    kept_hashes: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            h = dhash(frame)
            if all(hamming(h, k) >= min_distance for k in kept_hashes):
                kept_hashes.append(h)
                yield idx, frame
        idx += 1
    cap.release()


# ── survey ───────────────────────────────────────────────────────────────
def survey(src: Path, min_distance: int) -> int:
    images, videos = collect_sources(src)
    print(f"Source: {src}")
    print(f"  images: {len(images)}   videos: {len(videos)}\n")

    if not images and not videos:
        print("  Nothing usable found.")
        return 1

    frames: list[tuple[str, np.ndarray]] = []

    for v in videos:
        cap = cv2.VideoCapture(str(v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        cap.release()
        print(f"  VIDEO {v.name}: {total} frames, {fps:.1f} fps, "
              f"{total/fps if fps else 0:.1f}s")
        kept = list(distinct_video_frames(v, min_distance))
        print(f"    visually distinct frames: {len(kept)} of {total}")
        if total and len(kept) / total < 0.25:
            print("    (a static camera makes consecutive frames near-identical; "
                  "only the distinct ones are worth labelling)")
        for i, f in kept:
            frames.append((f"{v.stem}_{i:05d}", f))

    for p in images:
        im = cv2.imread(str(p))
        if im is None:
            print(f"  IMAGE {p.name}: unreadable")
            continue
        print(f"  IMAGE {p.name}: {im.shape[1]}x{im.shape[0]}")
        frames.append((p.stem, im))

    # Cross-source duplicate check: photos are often stills of the same scene.
    print(f"\n  candidate images total: {len(frames)}")
    uniq: list[tuple[str, np.ndarray]] = []
    hashes: list[np.ndarray] = []
    dupes = []
    for name, f in frames:
        h = dhash(f)
        clash = next((uniq[i][0] for i, k in enumerate(hashes)
                      if hamming(h, k) < min_distance), None)
        if clash:
            dupes.append((name, clash))
        else:
            hashes.append(h)
            uniq.append((name, f))
    print(f"  after cross-source de-duplication: {len(uniq)}")
    if dupes:
        print(f"  near-duplicates dropped: {len(dupes)}")
        for a, b in dupes[:8]:
            print(f"    {a}  ~=  {b}")

    # ── accessory presence estimate ──────────────────────────────────────
    print("\n  Estimating accessory presence (YOLO-World, indicative only)...")
    try:
        from zero_shot_accessory_detector import ZeroShotAccessoryDetector
        zs = ZeroShotAccessoryDetector(
            model_path=str(PROJECT_ROOT / "yolov8s-world.pt"), confidence=0.20
        )
    except Exception as exc:
        print(f"    unavailable ({exc})")
        zs = None

    person_counts = []
    try:
        from ultralytics import YOLO
        person = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))
    except Exception:
        person = None

    tally = Counter()
    if zs is not None and zs.available:
        for name, f in uniq:
            for det in zs.detect(f, imgsz=960):
                tally[det["class_name"]] += 1
            if person is not None:
                r = person.predict(f, conf=0.35, classes=[0], imgsz=960,
                                   verbose=False)[0]
                person_counts.append(0 if r.boxes is None else len(r.boxes))

    print(f"\n  persons per image: "
          f"{f'min={min(person_counts)} max={max(person_counts)} mean={np.mean(person_counts):.1f}' if person_counts else 'n/a'}")
    print("\n  estimated accessory instances across the distinct images:")
    for cls in CLASS_INDEX:
        n = tally.get(cls, 0)
        verdict = "NOT PRESENT - cannot train this class" if n == 0 else \
                  "very few - needs far more" if n < 20 else "some"
        print(f"    {cls:<12} ~{n:<5} {verdict}")

    # ── verdict ──────────────────────────────────────────────────────────
    print("\n  " + "-" * 62)
    print("  VERDICT")
    print("  " + "-" * 62)
    print(f"  Labelled images available now : 0")
    print(f"  Distinct images to label      : {len(uniq)}")
    print("  Rough guide for a usable model: 300-500+ images per class,")
    print("  1000+ instances per class, across varied scenes and altitudes.")
    missing = [c for c in CLASS_INDEX if tally.get(c, 0) == 0]
    if missing:
        print(f"\n  Classes with NO visible examples: {', '.join(missing)}")
        print("  No amount of labelling can teach a class the footage")
        print("  does not contain. Extra media is required for these.")
    return 0


# ── extraction ───────────────────────────────────────────────────────────
def extract(src: Path, min_distance: int, val_split: float,
            autolabel: bool, force: bool) -> int:
    images, videos = collect_sources(src)
    if not images and not videos:
        print("Nothing usable found.")
        return 1

    frames: list[tuple[str, np.ndarray]] = []
    for v in videos:
        for i, f in distinct_video_frames(v, min_distance):
            frames.append((f"{v.stem.replace(' ', '_')}_{i:05d}", f))
    for p in images:
        im = cv2.imread(str(p))
        if im is not None:
            frames.append((p.stem.replace(" ", "_"), im))

    # Cross-source de-duplication.
    uniq: list[tuple[str, np.ndarray]] = []
    hashes: list[np.ndarray] = []
    for name, f in frames:
        h = dhash(f)
        if all(hamming(h, k) >= min_distance for k in hashes):
            hashes.append(h)
            uniq.append((name, f))

    if not uniq:
        print("No distinct frames survived de-duplication.")
        return 1

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = DATASET_DIR / kind / split
            if d.exists() and any(d.iterdir()) and not force:
                print(f"{d} is not empty. Re-run with --force to overwrite.")
                return 1
            if d.exists() and force:
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    zs = None
    if autolabel:
        try:
            from zero_shot_accessory_detector import ZeroShotAccessoryDetector
            zs = ZeroShotAccessoryDetector(
                model_path=str(PROJECT_ROOT / "yolov8s-world.pt"),
                confidence=0.20,
            )
            if not zs.available:
                zs = None
        except Exception as exc:
            print(f"Auto-label unavailable ({exc}); writing empty labels.")
        if zs is None:
            print("Proceeding without candidate labels.")

    # Deterministic split: every 5th image to val by default.
    step = max(int(round(1 / val_split)), 2) if val_split > 0 else 0
    tally = Counter()
    written = {"train": 0, "val": 0}

    for i, (name, frame) in enumerate(uniq):
        split = "val" if step and i % step == 0 else "train"
        img_path = DATASET_DIR / "images" / split / f"{name}.jpg"
        cv2.imwrite(str(img_path), frame)

        lines = []
        if zs is not None:
            h, w = frame.shape[:2]
            for det in zs.detect(frame, imgsz=960):
                cls = CLASS_INDEX.get(det["class_name"])
                if cls is None:
                    continue
                x1, y1, x2, y2 = det["xyxy"]
                # Clamp, then convert to normalised centre/size.
                x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
                y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append(
                    f"{cls} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} "
                    f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}"
                )
                tally[det["class_name"]] += 1

        # Always write the .txt, even when empty: an empty label file is a
        # valid negative example, a missing one is silently skipped.
        (DATASET_DIR / "labels" / split / f"{name}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        written[split] += 1

    print(f"\nWrote {written['train']} train / {written['val']} val images to "
          f"{DATASET_DIR / 'images'}")
    if zs is not None:
        print("\nCANDIDATE label instances (DRAFT - must be reviewed):")
        for cls in CLASS_INDEX:
            print(f"  {cls:<12} {tally.get(cls, 0)}")
        print("\nThese came from an untrained open-vocabulary model. Review and")
        print("correct them before training, or the new model will inherit its")
        print("errors. Pay particular attention to mask and headphones, which")
        print("this model detects poorly.")
    else:
        print("\nEmpty label files written. Annotate them before training.")

    print("\nNext:")
    print("  1. Label/correct in Roboflow, CVAT or labelImg")
    print("  2. python train_accessory_model.py --check")
    print("  3. python train_accessory_model.py")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Prepare raw video/photos for accessory-model training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", required=True, help="folder containing video/photos")
    p.add_argument("--survey", action="store_true",
                   help="report what is present; write nothing")
    p.add_argument("--extract", action="store_true",
                   help="write deduplicated frames into dataset/")
    p.add_argument("--autolabel", action="store_true",
                   help="also write CANDIDATE labels (must be reviewed)")
    p.add_argument("--min-distance", type=int, default=8,
                   help="dHash distance below which frames count as duplicates")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="fraction of images held back for validation")
    p.add_argument("--force", action="store_true",
                   help="overwrite a non-empty dataset/ directory")
    args = p.parse_args()

    src = Path(args.src).expanduser()
    if not src.is_dir():
        print(f"Not a directory: {src}")
        return 1

    if args.extract:
        return extract(src, args.min_distance, args.val_split,
                       args.autolabel, args.force)
    return survey(src, args.min_distance)


if __name__ == "__main__":
    sys.exit(main())
