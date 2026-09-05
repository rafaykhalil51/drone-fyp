"""
train_accessory_model.py
------------------------
Trains the custom accessory model this project needs and installs it at the
path the application already looks for: models/accessory_best.pt

The four classes are fixed by the pipeline: cap, mask, glasses, headphones.

Quick start
-----------
  1. Put your labelled data under dataset/ (layout documented in
     dataset/data.yaml).
  2. Check it:      python train_accessory_model.py --check
  3. Train:         python train_accessory_model.py
  4. Restart the dashboard. It picks the model up automatically.

Useful options
--------------
  --epochs 150            longer training (default 100)
  --base yolov8s.pt       larger backbone; more accurate, slower
  --imgsz 960             must match accessories.imgsz in config.yaml
  --batch 8               lower this first if you run out of memory
  --device cpu            force CPU (auto-detected by default)
  --no-install            train but do not copy the weights into models/
  --resume                continue an interrupted run

Notes
-----
Small accessories in drone footage need a large inference size, which is why
imgsz defaults to 960 here and in config.yaml. Keep the two in step: training
at 960 and inferring at 320 will lose most of the small detections.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
INSTALL_PATH = PROJECT_ROOT / "models" / "accessory_best.pt"

REQUIRED_CLASSES = ["cap", "mask", "glasses", "headphones"]


# ── dataset validation ───────────────────────────────────────────────────
def check_dataset(data_yaml: Path) -> bool:
    """Validate the dataset before spending hours training on it."""
    print(f"Checking dataset definition: {data_yaml}")
    if not data_yaml.exists():
        print(f"  MISSING: {data_yaml}")
        print("  Create it from the template in dataset/data.yaml.")
        return False

    try:
        import yaml
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"  Could not parse: {exc}")
        return False

    ok = True

    # -- classes ----------------------------------------------------------
    names = cfg.get("names")
    if isinstance(names, dict):
        ordered = [str(names[k]) for k in sorted(names)]
    elif isinstance(names, list):
        ordered = [str(n) for n in names]
    else:
        print("  'names' missing or malformed.")
        return False

    print(f"  classes: {ordered}")
    if ordered != REQUIRED_CLASSES:
        # Not necessarily fatal -- the app resolves aliases -- but order
        # mismatches are the most common cause of scrambled predictions.
        print(f"  WARNING: expected exactly {REQUIRED_CLASSES} in that order.")
        missing = [c for c in REQUIRED_CLASSES if c not in ordered]
        if missing:
            print(f"  ERROR: no class resolves to: {missing}")
            ok = False
        else:
            print("  All four are present but the ORDER differs. Predictions "
                  "will be mislabelled unless your label indices match.")
            ok = False

    # -- splits -----------------------------------------------------------
    # Ultralytics resolves `path` against the process cwd (project root).
    # This checker also tries the YAML folder so either layout works.
    raw_path = Path(str(cfg.get("path", ".")))
    train_rel = str(cfg.get("train") or "images/train")
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([
            Path.cwd() / raw_path,
            data_yaml.parent / raw_path,
            data_yaml.parent,
        ])
    root = next(
        (c.resolve() for c in candidates if (c / train_rel).is_dir()),
        (Path.cwd() / raw_path).resolve(),
    )
    for split in ("train", "val"):
        rel = cfg.get(split)
        if not rel:
            print(f"  ERROR: '{split}' not defined.")
            ok = False
            continue

        img_dir = (root / str(rel)).resolve()
        if not img_dir.is_dir():
            print(f"  ERROR: {split} images directory not found: {img_dir}")
            ok = False
            continue

        images = [p for p in img_dir.rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        lbl_dir = Path(str(img_dir).replace("images", "labels"))
        labels = list(lbl_dir.rglob("*.txt")) if lbl_dir.is_dir() else []

        print(f"  {split}: {len(images)} images, {len(labels)} label files "
              f"({lbl_dir})")

        if not images:
            print(f"  ERROR: {split} split has no images.")
            ok = False
        elif not labels:
            print(f"  ERROR: no labels found. Expected them in {lbl_dir}")
            ok = False
        else:
            # Report unlabelled images; a few are fine (negatives), but a
            # large number usually means the label tree is in the wrong place.
            stems = {p.stem for p in images}
            lstems = {p.stem for p in labels}
            orphans = stems - lstems
            if orphans:
                print(f"  note: {len(orphans)} image(s) have no .txt label. "
                      "Empty label files are correct for negatives; missing "
                      "ones are skipped by Ultralytics.")

            # Validate a sample of label files for format and class range.
            bad = 0
            for lp in labels[:200]:
                for ln, line in enumerate(
                    lp.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
                ):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"  BAD FORMAT {lp.name}:{ln} -> {len(parts)} "
                              "fields, expected 5")
                        bad += 1
                        break
                    try:
                        cid = int(parts[0])
                        coords = [float(v) for v in parts[1:]]
                    except ValueError:
                        print(f"  BAD NUMBER {lp.name}:{ln}")
                        bad += 1
                        break
                    if not 0 <= cid < len(ordered):
                        print(f"  BAD CLASS {lp.name}:{ln} -> index {cid} "
                              f"outside 0..{len(ordered)-1}")
                        bad += 1
                        break
                    if any(not 0.0 <= c <= 1.0 for c in coords):
                        print(f"  NOT NORMALISED {lp.name}:{ln} -> {coords}. "
                              "Coordinates must be 0-1, not pixels.")
                        bad += 1
                        break
                if bad >= 5:
                    print("  (stopping after 5 bad files)")
                    break
            if bad:
                ok = False

    print("  DATASET OK" if ok else "  DATASET HAS PROBLEMS (see above)")
    return ok


# ── training ─────────────────────────────────────────────────────────────
def train(args) -> int:
    if not check_dataset(Path(args.data)):
        print("\nFix the dataset problems above before training.")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Run: pip install ultralytics")
        return 1

    device = args.device
    if device is None:
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    if device == "cpu":
        print("\nNOTE: training on CPU. This is slow -- expect many hours. "
              "A CUDA GPU is strongly recommended.")

    print(f"\nBase model : {args.base}")
    print(f"Data       : {args.data}")
    print(f"Epochs     : {args.epochs}")
    print(f"Image size : {args.imgsz}")
    print(f"Batch      : {args.batch}")
    print(f"Device     : {device}")
    print(f"Run name   : {args.name}\n")

    model = YOLO(args.base)
    model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        project=str(PROJECT_ROOT / "runs" / "accessory"),
        patience=args.patience,
        seed=0,
        # Mild augmentation suited to drone footage: accessories are small and
        # roughly upright, so heavy vertical flipping would hurt.
        fliplr=0.5,
        flipud=0.0,
        degrees=args.degrees,
        scale=args.scale,
        exist_ok=True,
        resume=args.resume,
    )

    best = (PROJECT_ROOT / "runs" / "accessory" / args.name / "weights" / "best.pt")
    if not best.exists():
        print(f"\nTraining finished but {best} was not produced.")
        return 1

    print(f"\nBest weights: {best}  ({best.stat().st_size / 1e6:.1f} MB)")

    metrics = model.val(data=str(Path(args.data).resolve()), imgsz=args.imgsz,
                        device=device)
    try:
        print(f"mAP50    : {metrics.box.map50:.4f}")
        print(f"mAP50-95 : {metrics.box.map:.4f}")
    except Exception:
        pass

    if args.no_install:
        print(f"\n--no-install set. Copy it yourself when ready:\n  {best}\n  -> {INSTALL_PATH}")
        return 0

    INSTALL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if INSTALL_PATH.exists():
        backup = INSTALL_PATH.with_suffix(".pt.bak")
        shutil.copy2(INSTALL_PATH, backup)
        print(f"\nExisting model backed up to {backup}")
    shutil.copy2(best, INSTALL_PATH)
    print(f"Installed -> {INSTALL_PATH}")

    # Confirm the app will accept it.
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from model_loader import format_class_validation
        validated = YOLO(str(INSTALL_PATH))
        names = validated.names
        ordered = ([str(names[k]) for k in sorted(names)]
                   if isinstance(names, dict) else [str(n) for n in names])
        print("\nmodel.names:", names)
        print(format_class_validation(ordered))
    except Exception as exc:
        print(f"\nCould not self-validate ({exc}); check it in the dashboard.")

    print("\nDone. Restart the dashboard -- it loads models/accessory_best.pt "
          "automatically and the status should read ACCESSORY AI ACTIVE.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Train the A.E.G.I.S. accessory model "
                    "(cap / mask / glasses / headphones).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default=str(DATA_YAML), help="dataset YAML")
    p.add_argument("--base", default="yolov8n.pt",
                   help="base checkpoint (yolov8n/s/m.pt)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=960,
                   help="must match accessories.imgsz in config.yaml")
    p.add_argument("--batch", type=int, default=16,
                   help="reduce if you hit out-of-memory")
    p.add_argument("--device", default=None, help="'0' for GPU, 'cpu'; auto by default")
    p.add_argument("--name", default="accessory_v1", help="run name")
    p.add_argument("--patience", type=int, default=30,
                   help="early-stopping patience in epochs")
    p.add_argument("--degrees", type=float, default=5.0,
                   help="rotation augmentation, degrees")
    p.add_argument("--scale", type=float, default=0.5,
                   help="scale augmentation; helps with varying drone altitude")
    p.add_argument("--resume", action="store_true", help="resume an interrupted run")
    p.add_argument("--no-install", action="store_true",
                   help="do not copy weights to models/accessory_best.pt")
    p.add_argument("--check", action="store_true",
                   help="validate the dataset and exit without training")
    args = p.parse_args()

    if args.check:
        return 0 if check_dataset(Path(args.data)) else 1
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
