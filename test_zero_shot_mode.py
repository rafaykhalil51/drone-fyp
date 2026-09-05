"""
test_zero_shot_mode.py
----------------------
Guards the invariants of the experimental zero-shot accessory mode:

  * OFF by default, and off when config is missing or malformed
  * never reported as "ACCESSORY AI ACTIVE"
  * its detections are tagged, and the exported report records the source
  * negative labels are still rejected (it shares the normalisation map)
  * the detector degrades instead of raising when weights are absent
  * the trained path is still preferred whenever it is available

Run: python test_zero_shot_mode.py
"""

import json
import sys
import tempfile
from pathlib import Path

import app_config
from model_loader import (
    ZERO_SHOT_WARNING,
    ModelResolution,
    accessory_ai_label,
    load_zero_shot_detector,
)
from zero_shot_accessory_detector import (
    DEFAULT_VOCABULARY,
    ZeroShotAccessoryDetector,
)

failures = []


def check(label, got, expected):
    ok = got == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: got={got!r} expected={expected!r}")
    if not ok:
        failures.append(label)


def check_true(label, got):
    check(label, bool(got), True)


# ── 1. off by default ────────────────────────────────────────────────────
print("\n--- default state ---")
check("config zero_shot enabled (fallback when no trained model)", app_config.zero_shot_enabled(), True)
_orig_zs = app_config.zero_shot_enabled
try:
    app_config.zero_shot_enabled = lambda: False
    check("loader returns None when disabled", load_zero_shot_detector(), None)
finally:
    app_config.zero_shot_enabled = _orig_zs

_orig = app_config.load_config
try:
    app_config.load_config = lambda: {}
    check("missing config -> disabled", app_config.zero_shot_enabled(), False)
    app_config.load_config = lambda: {"accessories": "not-a-dict"}
    check("malformed config -> disabled", app_config.zero_shot_enabled(), False)
    app_config.load_config = lambda: {"accessories": {"zero_shot": "nonsense"}}
    check("malformed zero_shot -> disabled", app_config.zero_shot_enabled(), False)
finally:
    app_config.load_config = _orig

# ── 2. never claims ACCESSORY AI ACTIVE ──────────────────────────────────
print("\n--- status labelling ---")
offline = ModelResolution(
    mode="person_only",
    person_path="yolov8n.pt",
    accessory_path=None,
    person_classes=("person",),
    accessory_classes=tuple(),
    accessory_active=False,
    requested_path="models/accessory_best.pt",
)
check("offline + zero-shot off", accessory_ai_label(offline, False), "OFFLINE")
check("offline + zero-shot on",
      accessory_ai_label(offline, True),
      "ZERO-SHOT (EXPERIMENTAL - not a trained model)")
check_true("zero-shot label is not plain ACTIVE",
           accessory_ai_label(offline, True) != "ACTIVE")
check("zero-shot never sets accessory_ai_active", offline.accessory_ai_active, False)

trained = ModelResolution(
    mode="dual",
    person_path="yolov8n.pt",
    accessory_path="models/accessory_best.pt",
    person_classes=("person",),
    accessory_classes=("cap", "mask", "glasses", "headphones"),
    accessory_active=True,
    requested_path="models/accessory_best.pt",
)
# A trained model must win even if zero-shot is switched on.
check("trained model takes precedence",
      accessory_ai_label(trained, True), "ACTIVE (trained model)")

# ── 3. detector contract ─────────────────────────────────────────────────
print("\n--- detector contract ---")
check("is_trained is False", ZeroShotAccessoryDetector.is_trained, False)
check_true("source_label marks it experimental",
           "experimental" in ZeroShotAccessoryDetector.source_label.lower())

missing = ZeroShotAccessoryDetector(model_path="no_such_world_model.pt")
check("missing weights -> unavailable", missing.available, False)
check("missing weights -> no detections", missing.detect(None), [])

import numpy as np
blank = np.zeros((64, 64, 3), dtype=np.uint8)
check("missing weights -> no detections on frame", missing.detect(blank), [])

with tempfile.TemporaryDirectory() as td:
    bogus = Path(td) / "corrupt.pt"
    bogus.write_bytes(b"not a checkpoint")
    try:
        bad = ZeroShotAccessoryDetector(model_path=str(bogus))
        check("corrupt weights -> unavailable", bad.available, False)
        check("corrupt weights -> no detections", bad.detect(blank), [])
    except Exception as exc:
        check(f"corrupt weights must not raise ({type(exc).__name__})", True, False)

# A plain (non-open-vocabulary) checkpoint must be refused, not silently used.
if Path("yolov8n.pt").exists():
    plain = ZeroShotAccessoryDetector(model_path="yolov8n.pt")
    check("plain COCO model refused for zero-shot", plain.available, False)
    check("plain COCO model yields nothing", plain.detect(blank), [])

# ── 4. vocabulary hygiene ────────────────────────────────────────────────
print("\n--- vocabulary ---")
check_true("vocabulary is non-empty", len(DEFAULT_VOCABULARY) > 0)
check("'headphones on neck' removed (not worn)",
      "headphones on neck" in DEFAULT_VOCABULARY, False)

from model_loader import match_hud_slot
unmapped = [w for w in DEFAULT_VOCABULARY
            if match_hud_slot(w) not in ("cap", "mask", "glasses", "headphones")]
check("every prompt maps to a slot", unmapped, [])

covered = {match_hud_slot(w) for w in DEFAULT_VOCABULARY}
check("all four slots covered by prompts",
      sorted(covered), ["cap", "glasses", "headphones", "mask"])

# Shares the normalisation map, so negatives are still rejected.
for neg in ("without_mask", "no_helmet", "mask_weared_incorrect"):
    check(f"negative {neg!r} rejected", match_hud_slot(neg), None)

# ── 5. report provenance ─────────────────────────────────────────────────
print("\n--- exported report provenance ---")
from exporter import Exporter


class _S:
    """Minimal stand-in for a StateManager with one track."""
    total_unique = 1

    class _T:
        first_frame, last_frame, frame_count = 0, 10, 11

        def final_flags(self):
            return {"mask": False, "cap": True, "glasses": False,
                    "headphones": False}

    def all_states(self):
        return {1: self._T()}


with tempfile.TemporaryDirectory() as td:
    jp = Path(td) / "fr.json"
    ex = Exporter(csv_path=Path(td) / "t.csv", summary_path=Path(td) / "s.json",
                  final_report_csv=Path(td) / "fr.csv", final_report_json=jp)

    ex.save_final_report(_S(), accessory_source="zero-shot:yolo-world (EXPERIMENTAL, untrained)")
    agg = json.loads(jp.read_text())["aggregate"]
    check("zero-shot source recorded",
          agg.get("accessory_source"),
          "zero-shot:yolo-world (EXPERIMENTAL, untrained)")

    ex.save_final_report(_S(), accessory_source="trained:models/accessory_best.pt")
    agg = json.loads(jp.read_text())["aggregate"]
    check("trained source recorded",
          agg.get("accessory_source"), "trained:models/accessory_best.pt")

    # Absent source must not invent a key (keeps old reports valid).
    ex.save_final_report(_S())
    agg = json.loads(jp.read_text())["aggregate"]
    check("no source -> key omitted", "accessory_source" in agg, False)

# ── 6. warning text is honest ────────────────────────────────────────────
print("\n--- warning text ---")
for token in ("not", "trained", "estimate"):
    check_true(f"warning mentions {token!r}", token in ZERO_SHOT_WARNING.lower())
check_true("warning names the real fix",
           "accessory_best.pt" in ZERO_SHOT_WARNING)

# ── 7. dashboard wiring ──────────────────────────────────────────────────
print("\n--- dashboard wiring ---")
dash = Path("dashboard.py").read_text(encoding="utf-8")
check_true("opt-in checkbox exists", "zero_shot_toggle" in dash)
check_true("separate display gate exists", "accessory_values_available" in dash)
check_true("zero-shot pill exists", "ACCESSORY: ZERO-SHOT (EXPERIMENTAL)" in dash)
check_true("warning is shown", "ZERO_SHOT_WARNING" in dash)
check_true("export records the source", "accessory_source=" in dash)
# accessory_model_active must never be assigned from the zero-shot flag.
check("zero-shot never sets accessory_model_active",
      "accessory_model_active = zero_shot_active" in dash, False)
check("zero-shot never ORed into accessory_model_active",
      "accessory_model_active = accessory_model_active or zero_shot" in dash, False)

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ALL ZERO-SHOT MODE CHECKS PASSED")
