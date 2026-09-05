"""
diag_legacy_accessory.py
------------------------
DIAGNOSTIC ONLY -- not part of the pipeline, not imported by the app.

Reproduces the accessory detector as it existed at commit 165754e (HEAD) to
establish what the previous "working" version actually produced. That version
fell back to yolov8s-world.pt (YOLO-World, open-vocabulary) with a prompt list
of accessory words whenever accessory_best.pt was absent -- which was always,
because no .pt file was ever tracked in this repository.

Reports every raw label, its confidence, and which canonical slot the old
substring mapper assigned it, so the zero-shot output can be judged on
evidence rather than on the dashboard looking populated.

Run: python diag_legacy_accessory.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

# The exact prompt vocabulary from commit 165754e.
LEGACY_VOCAB = [
    "cap", "hat", "baseball cap", "beanie", "beret",
    "mask", "face mask", "surgical mask",
    "glasses", "eyeglasses", "sunglasses", "spectacles",
    "headphones", "over-ear headphones", "headset", "earphones",
    "wireless headphones", "headphones on head", "headphones on neck",
    "black headphones", "earmuffs",
]

# The exact confidence the legacy code forced in fallback mode:
#   effective_conf = min(self.confidence, 0.18) if not self.is_custom
LEGACY_CONF = 0.18
LEGACY_IMGSZ = 384


def legacy_map_class_name(raw_name: str):
    """Verbatim copy of the legacy substring mapper from commit 165754e."""
    raw_name = raw_name.strip().lower()
    if any(w in raw_name for w in ["cap", "hat", "helmet", "hood", "beanie", "beret"]):
        return "cap"
    elif any(w in raw_name for w in ["mask", "facemask", "respirator"]):
        return "mask"
    elif any(w in raw_name for w in ["glass", "spectacle", "goggle"]):
        return "glasses"
    elif any(w in raw_name for w in ["headphone", "earphone", "headset",
                                     "earmuff", "earbud", "audio"]):
        return "headphones"
    return None


def main():
    weights = Path("yolov8s-world.pt")
    if not weights.exists():
        print("yolov8s-world.pt not present; cannot reproduce legacy behaviour.")
        return 1

    from ultralytics import YOLO

    print(f"Loading {weights.name} (YOLO-World, open-vocabulary)...")
    model = YOLO(str(weights))
    model.set_classes(LEGACY_VOCAB)
    print(f"Prompted with {len(LEGACY_VOCAB)} text labels, conf={LEGACY_CONF}\n")

    targets = []
    for img in ("uploads/classroom_test.jpg", "uploads/classroom_sample.jpg"):
        if Path(img).exists():
            targets.append(("image", img))
    if Path("input.mp4").exists():
        targets.append(("video", "input.mp4"))

    for kind, path in targets:
        print("=" * 68)
        print(f"{kind.upper()}: {path}")
        print("=" * 68)

        frames = []
        if kind == "image":
            frame = cv2.imread(path)
            if frame is None:
                print("  could not read\n")
                continue
            frames = [frame]
        else:
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            # Sample 12 frames spread across the clip.
            picks = [int(total * i / 12) for i in range(12)] if total else list(range(12))
            for idx in picks:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, f = cap.read()
                if ok:
                    frames.append(f)
            cap.release()
            print(f"  sampled {len(frames)} frames of {total}")

        raw_counter = Counter()
        slot_counter = Counter()
        conf_by_slot = defaultdict(list)
        total_dets = 0

        for f in frames:
            res = model.predict(f, conf=LEGACY_CONF, iou=0.50,
                                imgsz=LEGACY_IMGSZ, verbose=False)
            for r in res:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cid = int(b.cls[0])
                    names = r.names
                    raw = (names.get(cid, str(cid)) if isinstance(names, dict)
                           else str(names[cid])).lower()
                    cf = float(b.conf[0])
                    total_dets += 1
                    raw_counter[raw] += 1
                    slot = legacy_map_class_name(raw)
                    if slot:
                        slot_counter[slot] += 1
                        conf_by_slot[slot].append(cf)

        print(f"\n  raw detections: {total_dets}")
        if raw_counter:
            print("  raw labels emitted:")
            for raw, n in raw_counter.most_common():
                print(f"    {raw:<26} x{n}")
        print("\n  mapped to canonical slots:")
        if slot_counter:
            for slot in ("cap", "mask", "glasses", "headphones"):
                n = slot_counter.get(slot, 0)
                confs = conf_by_slot.get(slot, [])
                if confs:
                    print(f"    {slot:<12} x{n:<4} conf min={min(confs):.2f} "
                          f"mean={sum(confs)/len(confs):.2f} max={max(confs):.2f}")
                else:
                    print(f"    {slot:<12} x0")
        else:
            print("    (nothing mapped)")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
