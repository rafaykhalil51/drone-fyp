"""
test_model_architecture.py
--------------------------
Guards the person/accessory model separation:

  * person detection never depends on a custom model
  * accessory AI is reported active ONLY for a real custom model that
    resolves all four required classes
  * a COCO model can never masquerade as an accessory detector
  * a missing / corrupt / incomplete model degrades instead of crashing
  * mock accessory detection stays off in real mode

Run: python test_model_architecture.py
"""

import os
import sys
import tempfile
from pathlib import Path

import app_config
from accessory_detector import CANONICAL_CLASSES, AccessoryDetector
from model_loader import (
    ACCESSORY_OFFLINE_MESSAGE,
    DEFAULT_CUSTOM_PATH,
    PERSON_FALLBACKS,
    REQUIRED_ACCESSORY_CLASSES,
    ModelResolution,
    format_status_banner,
    is_coco_model,
    is_open_vocab_weight,
    resolve_models,
    validate_accessory_classes,
)

failures = []


def check(label, got, expected):
    ok = got == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: got={got!r} expected={expected!r}")
    if not ok:
        failures.append(label)


def check_true(label, got):
    check(label, bool(got), True)


# ── 1. required classes and aliases ──────────────────────────────────────
print("\n--- required classes / normalisation dictionary ---")
check("four required classes", list(REQUIRED_ACCESSORY_CLASSES),
      ["cap", "mask", "glasses", "headphones"])
check("canonical classes agree", sorted(CANONICAL_CLASSES),
      sorted(REQUIRED_ACCESSORY_CLASSES))

# The aliases named in the brief must all resolve through one dictionary.
alias_model = ["hat", "face_mask", "eyeglasses", "headphone"]
check("brief aliases resolve", validate_accessory_classes(alias_model)["valid"], True)
check("alias mapping is reported",
      validate_accessory_classes(alias_model)["resolved"],
      {"cap": "hat", "mask": "face_mask", "glasses": "eyeglasses",
       "headphones": "headphone"})

for raw, slot in (("facemask", "mask"), ("spectacles", "glasses"),
                  ("headset", "headphones")):
    res = validate_accessory_classes([raw])["resolved"]
    check(f"alias {raw!r} -> {slot}", res.get(slot), raw)

# ── 2. exact model.names dict from the brief ─────────────────────────────
print("\n--- brief's example model.names ---")
names_dict = {0: "cap", 1: "mask", 2: "glasses", 3: "headphones"}
ordered = [names_dict[k] for k in sorted(names_dict)]
check("example model validates", validate_accessory_classes(ordered)["valid"], True)

# ── 3. incomplete model must be rejected ─────────────────────────────────
print("\n--- incomplete model must NOT be accepted ---")
partial = validate_accessory_classes(["cap", "mask"])
check("partial model invalid", partial["valid"], False)
check("partial reports missing", partial["missing"], ["glasses", "headphones"])

# ── 4. COCO must never fake accessories ──────────────────────────────────
print("\n--- COCO can never supply accessories ---")
coco = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "bottle", "wine glass", "cup", "cell phone",
    "hair drier", "toothbrush",
]
check("coco fails validation", validate_accessory_classes(coco)["valid"], False)
check("coco missing all four", validate_accessory_classes(coco)["missing"],
      list(REQUIRED_ACCESSORY_CLASSES))
check_true("wine glass is not glasses",
           "glasses" not in validate_accessory_classes(["wine glass"])["resolved"])

# ── 5. open-vocabulary weights are not auto-adopted ──────────────────────
print("\n--- open-vocabulary weights excluded ---")
for name in ("yolov8s-world.pt", "yolov8l-worldv2.pt", "yoloe-v8s.pt"):
    check_true(f"{name} flagged open-vocab", is_open_vocab_weight(name))
for name in ("yolov8n.pt", "models/accessory_best.pt", "best.pt"):
    check(f"{name} not open-vocab", is_open_vocab_weight(name), False)

# ── 6. person detection is independent of the accessory model ────────────
print("\n--- person detection independence ---")
check("yolov8n.pt preferred first", PERSON_FALLBACKS[0], "yolov8n.pt")
check("config person model", app_config.person_model_path(), "yolov8n.pt")
check("documented accessory path", DEFAULT_CUSTOM_PATH, "models/accessory_best.pt")

res = resolve_models()
check_true("a person model was resolved", bool(res.person_path))
check_true("person model file exists", os.path.exists(res.person_path))
check_true("person classes include person", "person" in res.person_classes)

# With no trained accessory model present, accessory AI must be offline.
if not app_config.accessory_model_exists():
    check("accessory status is not_found", res.accessory_status, "not_found")
    check("accessory AI offline", res.accessory_ai_active, False)
    check("all four reported missing", res.missing_accessory_classes,
          list(REQUIRED_ACCESSORY_CLASSES))
    check_true("person detection still available", os.path.exists(res.person_path))

# ── 7. a COCO-only resolution can never claim ACCESSORY AI ACTIVE ────────
print("\n--- fabricated resolution cannot claim active ---")
faked = ModelResolution(
    mode="dual",
    person_path="yolov8n.pt",
    accessory_path="yolov8n.pt",          # COCO pretending to be accessories
    person_classes=tuple(coco),
    accessory_classes=tuple(),            # nothing resolved
    accessory_active=True,                # even with the legacy flag set
    requested_path="yolov8n.pt",
)
check("coco-as-accessory is invalid", faked.accessory_ai_active, False)
check("coco-as-accessory status", faked.accessory_status, "invalid")

valid_res = ModelResolution(
    mode="dual",
    person_path="yolov8n.pt",
    accessory_path="models/accessory_best.pt",
    person_classes=tuple(coco),
    accessory_classes=("cap", "mask", "glasses", "headphones"),
    accessory_active=True,
    requested_path="models/accessory_best.pt",
)
check("valid custom model is active", valid_res.accessory_ai_active, True)
check("valid custom model status", valid_res.accessory_status, "active")

# ── 8. detector degrades gracefully ──────────────────────────────────────
print("\n--- graceful degradation ---")
missing = AccessoryDetector(model_path="models/definitely_absent.pt")
check("missing model unavailable", missing.available, False)
check("missing model load_error", missing.load_error, "not_found")
check("missing model yields no detections", missing.detect(None), [])

import numpy as np
blank = np.zeros((64, 64, 3), dtype=np.uint8)
check("missing model no detections on a frame", missing.detect(blank), [])

# A file that is not a real YOLO checkpoint must not raise.
with tempfile.TemporaryDirectory() as td:
    bogus = Path(td) / "corrupt.pt"
    bogus.write_bytes(b"this is not a torch checkpoint")
    try:
        bad = AccessoryDetector(model_path=str(bogus))
        check("corrupt model unavailable", bad.available, False)
        check_true("corrupt model records an error", bad.load_error)
        check("corrupt model yields no detections", bad.detect(blank), [])
    except Exception as exc:
        check(f"corrupt model must not raise ({type(exc).__name__})", True, False)

# ── 9. mock detection is off ─────────────────────────────────────────────
print("\n--- mock accessory detection ---")
check("use_mock_accessories is false", app_config.use_mock_accessories(), False)

import app_config as _c
_orig = _c.load_config
try:
    _c.load_config = lambda: {}                    # simulate a missing config
    check("missing config defaults mock to off", _c.use_mock_accessories(), False)
finally:
    _c.load_config = _orig

# ── 10. device selection ─────────────────────────────────────────────────
print("\n--- device selection ---")
check_true("device is cuda or cpu", app_config.torch_device() in ("cuda", "cpu"))

# ── 11. status banner content ────────────────────────────────────────────
print("\n--- status banner ---")
banner = format_status_banner(res)
for token in ("=== AEGIS AI STATUS ===", "Person model:", "Accessory model:",
              "Tracking:", "Mock accessory detection:", "Accessory AI:"):
    check_true(f"banner contains {token!r}", token in banner)
check_true("banner reports mock OFF", "OFF" in banner)
if not app_config.accessory_model_exists():
    if app_config.zero_shot_enabled():
        check_true("banner reports ZERO-SHOT fallback", "ZERO-SHOT" in banner)
    else:
        check_true("banner reports accessory OFFLINE", "OFFLINE" in banner)
check_true("offline message names all four accessories",
           all(w in ACCESSORY_OFFLINE_MESSAGE
               for w in ("cap", "mask", "glasses", "headphones")))

# ── 12. no giant card text remains ───────────────────────────────────────
print("\n--- dashboard card placeholder ---")
dash = Path("dashboard.py").read_text(encoding="utf-8")
check("'Custom model required' removed", "Custom model required" in dash, False)
check("'COCO MODE' removed", "COCO<br>MODE" in dash, False)
check_true("offline placeholder present", 'OFFLINE_VALUE = "--"' in dash)
check_true("card sublabel present", 'OFFLINE_SUBLABEL = "Model offline"' in dash)
check_true("status pill present", "ACCESSORY AI OFFLINE" in dash)
check_true("active pill present", "ACCESSORY AI ACTIVE" in dash)

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ALL MODEL ARCHITECTURE CHECKS PASSED")
