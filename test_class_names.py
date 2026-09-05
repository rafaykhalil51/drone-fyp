"""
Class-label mapping tests for match_hud_slot.

The dashboard reads class names straight from the model, so a trained
accessory model can use any labels its dataset happened to ship. Two failure
modes matter: a NEGATIVE label being counted as the accessory (public
face-mask datasets ship 'without_mask' next to 'with_mask'), and a common
synonym being silently dropped so real detections never reach the HUD.
"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model_loader import match_hud_slot  # noqa: E402

FAILS = []


def check(label, expected):
    actual = match_hud_slot(label)
    ok = actual == expected
    if not ok:
        FAILS.append(f"{label} -> {actual} (want {expected})")
    print(f"[{'PASS' if ok else 'FAIL'}] {label!r:32} -> {actual}")


print("--- person ---")
for lbl in ("person", "Person", "PEOPLE", " person ", "pedestrian", "human"):
    check(lbl, "person")

print("\n--- positive accessories ---")
for lbl in ("mask", "Mask", "face_mask", "face-mask", "facemask", "with_mask",
            "surgical_mask", "N95", "respirator"):
    check(lbl, "mask")
for lbl in ("glasses", "eyeglasses", "eye_glasses", "spectacles", "specs",
            "sunglasses", "Sunglasses", "sun-glasses", "goggles", "shades"):
    check(lbl, "glasses")
for lbl in ("headphones", "headphone", "headset", "earphones", "earbuds",
            "airpods", "earmuffs"):
    check(lbl, "headphones")
for lbl in ("cap", "Cap", "hat", "beanie", "helmet", "baseball_cap",
            "hard_hat", "hardhat", "headwear"):
    check(lbl, "cap")

print("\n--- NEGATIVE labels must NOT count as wearing ---")
for lbl in ("without_mask", "no_mask", "no-mask", "not_wearing_mask",
            "mask_weared_incorrect", "incorrect_mask", "improper_mask",
            "without_glasses", "no_glasses", "no_helmet", "without_cap",
            "no_headphones", "mask_absent", "mask_missing"):
    check(lbl, None)

print("\n--- unrelated COCO classes must not map ---")
for lbl in ("bicycle", "car", "toothbrush", "handbag", "tie", "backpack",
            "umbrella", "suitcase", "bottle", "chair"):
    check(lbl, None)

# 'wine glass' is a real COCO class and must never register as eyewear,
# or every COCO model would look like an accessory model.
print("\n--- COCO collisions ---")
for lbl in ("wine glass", "wine_glass", "baseball glove", "baseball bat",
            "sports ball", "cell phone", "hair drier", "teddy bear"):
    check(lbl, None)

# Guard the whole COCO label set in one shot.
COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
_coco_acc = [c for c in COCO if match_hud_slot(c) in
             ("cap", "mask", "glasses", "headphones")]
check_name = "no COCO class maps to an accessory slot"
ok = _coco_acc == []
if not ok:
    FAILS.append(f"{check_name}: {_coco_acc}")
print(f"[{'PASS' if ok else 'FAIL'}] {check_name}: {_coco_acc or 'none'}")

_coco_person = [c for c in COCO if match_hud_slot(c) == "person"]
ok2 = _coco_person == ["person"]
if not ok2:
    FAILS.append(f"only 'person' maps to person: {_coco_person}")
print(f"[{'PASS' if ok2 else 'FAIL'}] only 'person' maps to person: {_coco_person}")

print("\n--- malformed input must not raise ---")
for lbl in ("", "   ", "___", "?!"):
    check(lbl, None)
check(None, None)

print("\n--- compound labels resolve via tokens ---")
check("man_wearing_cap", "cap")
check("person_with_headphones", "headphones")

# AccessoryDetector has its own _map_class_name entry point; it must agree
# with match_hud_slot, or the dual-model path would use different rules than
# the dashboard and reintroduce the negative-label bug.
print("\n--- AccessoryDetector._map_class_name agrees ---")
from accessory_detector import CANONICAL_CLASSES, AccessoryDetector  # noqa: E402

_det = AccessoryDetector.__new__(AccessoryDetector)   # no weights needed
for lbl, want in [
    ("cap", "cap"), ("hat", "cap"), ("hood", "cap"), ("beret", "cap"),
    ("with_mask", "mask"), ("face_mask", "mask"),
    ("sunglasses", "glasses"), ("goggles", "glasses"),
    ("earbuds", "headphones"), ("headset", "headphones"),
    ("without_mask", None), ("no_mask", None), ("no_helmet", None),
    ("mask_weared_incorrect", None), ("person", None), ("wine_glass", None),
]:
    got = _det._map_class_name(lbl)
    ok = got == want
    if not ok:
        FAILS.append(f"detector {lbl} -> {got} (want {want})")
    print(f"[{'PASS' if ok else 'FAIL'}] detector {lbl!r:24} -> {got}")

_bad = [c for c in ("cap", "mask", "glasses", "headphones")
        if c not in CANONICAL_CLASSES]
ok = _bad == []
if not ok:
    FAILS.append(f"CANONICAL_CLASSES missing {_bad}")
print(f"[{'PASS' if ok else 'FAIL'}] CANONICAL_CLASSES covers all four slots")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED:")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("ALL CLASS NAME CHECKS PASSED")
