"""
model_loader.py
---------------
Discover YOLO weight files, inspect class names, and resolve person vs
accessory models for the Streamlit dashboard and CLI pipeline.

Person detection falls back to yolo11n.pt (COCO). Accessory detection uses a
custom weight only when cap/mask/glasses/headphones classes are present.
No open-vocabulary or fake accessory inference.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

ACCESSORY_SLOTS = frozenset({"cap", "mask", "glasses", "headphones"})

# Every accessory class the project requires. A custom model must be able to
# resolve all four, or its statistics would be silently incomplete.
REQUIRED_ACCESSORY_CLASSES = ("cap", "mask", "glasses", "headphones")

# Person detection uses a pretrained COCO model. yolov8n.pt is preferred (and
# is what config.yaml requests); yolo11n.pt remains a fallback if it is the
# only weight present.
PERSON_FALLBACKS = ("yolov8n.pt", "yolo11n.pt")

# The documented home for the trained accessory model.
DEFAULT_CUSTOM_PATH = "models/accessory_best.pt"
NAMED_CUSTOM_CANDIDATES = (
    "models/accessory_best.pt",     # documented location
    "models/best.pt",
    "accessory_best.pt",            # legacy locations, still honoured
    "best.pt",
    "weights/accessory_best.pt",
    "weights/best.pt",
)
CLIP_NAME_FRAGMENT = "clip"

# Open-vocabulary weights can be prompted to emit any label, which would let a
# generic model masquerade as a trained accessory detector. Never auto-select
# them for accessory duty.
OPEN_VOCAB_FRAGMENTS = ("world", "yoloe", "owlvit", "grounding")


# Tokens that describe the ABSENCE (or incorrect wearing) of an accessory.
# Public datasets ship these alongside the positive class -- the widely used
# face-mask sets label 'with_mask' / 'without_mask' / 'mask_weared_incorrect'.
# They must never be counted as someone wearing the accessory.
#
# Matched as whole underscore-separated tokens, never as substrings: a
# substring test would read the "un" of "sunglasses" as a negation.
NEGATIVE_TOKENS = frozenset({
    "no", "not", "non", "none", "without", "wo",
    "absent", "missing", "off", "never", "lack", "lacking",
    "incorrect", "incorrectly", "improper", "improperly", "wrong",
})

# Compact negations that arrive as a single token, e.g. 'nomask', 'unmasked'.
_NEGATIVE_PREFIXES = ("without", "no", "un", "non")

# Positive synonyms per HUD slot, matched on whole tokens.
_SLOT_SYNONYMS: dict[str, frozenset[str]] = {
    "mask": frozenset({
        "mask", "masks", "face_mask", "facemask", "with_mask", "mask_on",
        "surgical_mask", "medical_mask", "cloth_mask", "n95", "respirator",
    }),
    "glasses": frozenset({
        "glasses", "glass", "eyeglasses", "eyeglass", "eye_glasses",
        "spectacles", "specs", "sunglasses", "sun_glasses", "shades",
        "goggles", "with_glasses", "glasses_on",
    }),
    # NOTE: bare "glass" is deliberately matched only as a complete label.
    # COCO ships "wine glass", which must never register as eyewear.
    "headphones": frozenset({
        "headphones", "headphone", "headset", "headsets", "earphones",
        "earphone", "earbuds", "earbud", "earpods", "airpods", "earmuffs",
        "with_headphones",
    }),
    "cap": frozenset({
        "cap", "caps", "hat", "hats", "beanie", "helmet", "helmets",
        "baseball_cap", "baseball_hat", "hard_hat", "hardhat", "cowboy_hat",
        "sun_hat", "headwear", "headgear", "with_cap", "with_hat", "cap_on",
        "hood", "hoodie", "beret", "turban", "bandana",
    }),
}


# Words specific enough to identify an accessory when they appear as one token
# inside a longer label. Ambiguous words ("glass", "cap" as in bottle cap) are
# excluded and only ever matched as a complete label.
_COMPOUND_SAFE_TOKENS: dict[str, frozenset[str]] = {
    "mask": frozenset({"mask", "masks", "facemask", "respirator"}),
    "glasses": frozenset({
        "glasses", "eyeglasses", "spectacles", "sunglasses", "goggles",
    }),
    "headphones": frozenset({
        "headphones", "headphone", "headset", "headsets", "earphones",
        "earphone", "earbuds", "airpods", "earmuffs",
    }),
    "cap": frozenset({
        "cap", "caps", "hat", "hats", "beanie", "helmet", "hardhat",
        "hood", "beret", "turban",
    }),
}


@lru_cache(maxsize=1)
def _all_synonyms() -> frozenset[str]:
    out: set[str] = set()
    for names in _SLOT_SYNONYMS.values():
        out |= names
    return frozenset(out)


def _is_negative_label(key: str) -> bool:
    """
    True when the label denotes absence rather than presence.

    Works on whole tokens, plus compact single-token forms such as 'nomask'
    or 'unmasked' where the negation is glued to the accessory word.
    """
    tokens = key.split("_")
    if any(tok in NEGATIVE_TOKENS for tok in tokens):
        return True

    synonyms = _all_synonyms()
    for tok in tokens:
        for prefix in _NEGATIVE_PREFIXES:
            if not tok.startswith(prefix) or len(tok) <= len(prefix):
                continue
            rest = tok[len(prefix):].lstrip("_")
            # 'unmasked' -> 'masked' -> 'mask'
            for candidate in (rest, rest.rstrip("d").rstrip("e"), f"{rest}s"):
                if candidate in synonyms:
                    return True
    return False


def match_hud_slot(class_name: str) -> str | None:
    """
    Map a model class label to a person or accessory HUD slot.

    Returns None for unknown labels and for labels describing the absence of
    an accessory, so a 'without_mask' detection is never counted as a mask.
    """
    key = (class_name or "").strip().lower()
    for ch in (" ", "-", ".", "/"):
        key = key.replace(ch, "_")
    key = "_".join(part for part in key.split("_") if part)

    if not key:
        return None

    if key in ("person", "people", "pedestrian", "human"):
        return "person"

    if _is_negative_label(key):
        return None

    for slot, synonyms in _SLOT_SYNONYMS.items():
        if key in synonyms:
            return slot

    # Fall back to token matching so compound labels such as 'man_wearing_cap'
    # still resolve. Only unambiguous words take part: matching bare "glass"
    # here would turn COCO's "wine glass" into eyewear.
    tokens = set(key.split("_"))
    for slot, safe_tokens in _COMPOUND_SAFE_TOKENS.items():
        if tokens & safe_tokens:
            return slot

    return None


def model_class_names(model) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)


def is_coco_model(class_names: list[str]) -> bool:
    normalized = {n.strip().lower() for n in class_names}
    return (
        len(normalized) >= 70
        and "person" in normalized
        and "bicycle" in normalized
        and "toothbrush" in normalized
    )


def accessory_slots(class_names: list[str]) -> set[str]:
    slots: set[str] = set()
    for name in class_names:
        slot = match_hud_slot(name)
        if slot in ACCESSORY_SLOTS:
            slots.add(slot)
    return slots


def has_person(class_names: list[str]) -> bool:
    return any(match_hud_slot(name) == "person" for name in class_names)


def _rel_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


@lru_cache(maxsize=64)
def inspect_model_classes(path: str) -> tuple[str, ...]:
    from ultralytics import YOLO

    model = YOLO(path)
    return tuple(model_class_names(model))


def discover_weight_files(root: Path | None = None) -> list[str]:
    """Return project-relative paths to YOLO .pt files (CLIP weights excluded)."""
    root = root or PROJECT_ROOT
    found: set[str] = set()

    for pattern in ("*.pt", "weights/*.pt"):
        for path in root.glob(pattern):
            if CLIP_NAME_FRAGMENT in path.name.lower():
                continue
            if path.is_file():
                found.add(_rel_path(path))

    return sorted(found)


def inspect_weight_inventory(root: Path | None = None) -> list[dict]:
    """Inspect every discovered weight file and return class metadata."""
    inventory = []
    for rel_path in discover_weight_files(root):
        full_path = str((root or PROJECT_ROOT) / rel_path)
        if not os.path.exists(full_path):
            continue
        try:
            classes = list(inspect_model_classes(full_path))
            inventory.append({
                "path": rel_path,
                "classes": classes,
                "accessory_slots": sorted(accessory_slots(classes)),
                "has_person": has_person(classes),
                "is_coco": is_coco_model(classes),
            })
        except Exception as exc:
            inventory.append({
                "path": rel_path,
                "classes": [],
                "accessory_slots": [],
                "has_person": False,
                "is_coco": False,
                "error": str(exc),
            })
    return inventory


@dataclass(frozen=True)
class ModelResolution:
    mode: str  # unified | dual | person_only
    person_path: str
    accessory_path: str | None
    person_classes: tuple[str, ...]
    accessory_classes: tuple[str, ...]
    accessory_active: bool
    requested_path: str

    @property
    def person_label(self) -> str:
        return self.person_path

    @property
    def accessory_label(self) -> str:
        if not self.accessory_active:
            return "not found"
        if self.mode == "unified":
            return f"{self.accessory_path} (unified)"
        return self.accessory_path or "not found"

    # ── required-class validation ─────────────────────────────────────────
    @property
    def class_validation(self) -> dict:
        """Validation of accessory_classes against the four required classes."""
        return validate_accessory_classes(self.accessory_classes)

    @property
    def accessory_classes_valid(self) -> bool:
        """True only when a model is present AND covers all four classes."""
        return bool(self.accessory_path) and self.class_validation["valid"]

    @property
    def missing_accessory_classes(self) -> list[str]:
        if not self.accessory_path:
            return list(REQUIRED_ACCESSORY_CLASSES)
        return self.class_validation["missing"]

    @property
    def accessory_status(self) -> str:
        """One of: active | invalid | not_found | disabled."""
        try:
            from app_config import accessory_enabled
            if not accessory_enabled():
                return "disabled"
        except Exception:
            pass
        if not self.accessory_path:
            return "not_found"
        return "active" if self.class_validation["valid"] else "invalid"

    @property
    def accessory_ai_active(self) -> bool:
        """
        The single source of truth for 'is accessory detection real'.

        Requires a genuine TRAINED custom model that resolved every required
        class. Never true for the zero-shot fallback, and never true merely
        because the interface loaded.
        """
        return self.accessory_status == "active" and self.accessory_active


def _classes_for_path(rel_path: str, cache: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    if rel_path in cache:
        return cache[rel_path]
    full = str(PROJECT_ROOT / rel_path)
    if os.path.exists(full):
        classes = inspect_model_classes(full)
        cache[rel_path] = classes
        return classes
    return tuple()


def _pick_person_fallback(cache: dict[str, tuple[str, ...]]) -> tuple[str, tuple[str, ...]]:
    """
    Choose the person detector: the model named in config.yaml first, then the
    built-in fallbacks. Person detection never depends on a custom model.
    """
    candidates: list[str] = []
    try:
        from app_config import person_model_path
        configured = person_model_path()
        if configured:
            candidates.append(configured)
    except Exception:
        pass                          # config unreadable; use built-in order
    candidates += [c for c in PERSON_FALLBACKS if c not in candidates]

    for candidate in candidates:
        if (PROJECT_ROOT / candidate).exists():
            return candidate, _classes_for_path(candidate, cache)
    return candidates[0] if candidates else PERSON_FALLBACKS[0], tuple()


def _accessory_class_names(class_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in class_names if match_hud_slot(name) in ACCESSORY_SLOTS)


def is_open_vocab_weight(path: str) -> bool:
    """True for prompt-driven models that could fake any label on request."""
    name = Path(path).name.lower()
    return any(frag in name for frag in OPEN_VOCAB_FRAGMENTS)


def validate_accessory_classes(class_names) -> dict:
    """
    Check that a candidate accessory model covers all four required classes.

    Returns
    -------
    dict with:
      valid    : bool  -- every required class resolved
      resolved : dict  -- required class -> the model label that supplied it
      missing  : list  -- required classes with no matching label
      classes  : list  -- the model's raw labels, for display
    """
    names = [str(n) for n in (class_names or [])]
    resolved: dict[str, str] = {}

    for raw in names:
        slot = match_hud_slot(raw)
        if slot in ACCESSORY_SLOTS and slot not in resolved:
            resolved[slot] = raw

    missing = [c for c in REQUIRED_ACCESSORY_CLASSES if c not in resolved]
    return {
        "valid": not missing,
        "resolved": resolved,
        "missing": missing,
        "classes": names,
    }


def format_class_validation(class_names) -> str:
    """Human-readable validation report for the dashboard and logs."""
    report = validate_accessory_classes(class_names)
    lines = [
        "Model classes:",
        "  " + (", ".join(report["classes"]) if report["classes"] else "(none)"),
        "",
        "Required:",
        "  " + ", ".join(REQUIRED_ACCESSORY_CLASSES),
    ]
    if report["missing"]:
        lines += ["", "Missing:", "  " + ", ".join(report["missing"])]
    else:
        mapping = ", ".join(
            f"{slot}<-{raw}" for slot, raw in sorted(report["resolved"].items())
        )
        lines += ["", "All required classes resolved:", "  " + mapping]
    return "\n".join(lines)


def _find_accessory_candidate(
    cache: dict[str, tuple[str, ...]],
    preferred_paths: tuple[str, ...] = NAMED_CUSTOM_CANDIDATES,
) -> tuple[str | None, tuple[str, ...]]:
    for rel_path in preferred_paths:
        full = PROJECT_ROOT / rel_path
        if not full.exists():
            continue
        classes = _classes_for_path(rel_path, cache)
        if accessory_slots(classes) and not is_coco_model(classes):
            return rel_path, classes

    best_path: str | None = None
    best_classes: tuple[str, ...] = tuple()
    best_score = (-1, -1)

    for rel_path in discover_weight_files():
        # An open-vocabulary model can be prompted into reporting any label,
        # so it must never be auto-adopted as the trained accessory detector.
        if is_open_vocab_weight(rel_path):
            continue
        classes = _classes_for_path(rel_path, cache)
        if not classes or is_coco_model(classes):
            continue
        slots = accessory_slots(classes)
        if not slots:
            continue
        score = (len(slots), -len(classes))
        if score > best_score:
            best_score = score
            best_path = rel_path
            best_classes = classes

    return best_path, best_classes


def resolve_models(custom_path: str | None = None) -> ModelResolution:
    """
    Resolve person and accessory YOLO weights.

    Priority:
      1. Unified custom model (person + cap/mask/glasses/headphones)
      2. Dual: yolo11n person + dedicated accessory weights
      3. Person-only fallback when no compatible accessory model exists
    """
    if custom_path is None:
        # Fall back to config.yaml's accessories.model_path so there is one
        # place that decides where the trained model lives.
        try:
            from app_config import accessory_model_path
            custom_path = accessory_model_path()
        except Exception:
            custom_path = DEFAULT_CUSTOM_PATH

    requested = (custom_path or DEFAULT_CUSTOM_PATH).strip() or DEFAULT_CUSTOM_PATH
    cache: dict[str, tuple[str, ...]] = {}

    unified_candidates: list[tuple[str, tuple[str, ...]]] = []

    if os.path.exists(requested):
        classes = _classes_for_path(requested, cache)
        if has_person(classes) and accessory_slots(classes):
            unified_candidates.append((requested, classes))

    for rel_path in NAMED_CUSTOM_CANDIDATES:
        if not (PROJECT_ROOT / rel_path).exists():
            continue
        classes = _classes_for_path(rel_path, cache)
        if has_person(classes) and accessory_slots(classes):
            unified_candidates.append((rel_path, classes))

    if unified_candidates:
        path, classes = unified_candidates[0]
        return ModelResolution(
            mode="unified",
            person_path=path,
            accessory_path=path,
            person_classes=classes,
            accessory_classes=_accessory_class_names(classes),
            accessory_active=True,
            requested_path=requested,
        )

    accessory_path, accessory_classes = _find_accessory_candidate(cache)

    if os.path.exists(requested):
        req_classes = _classes_for_path(requested, cache)
        req_slots = accessory_slots(req_classes)
        if req_slots and not is_coco_model(req_classes):
            accessory_path = requested
            accessory_classes = req_classes
        elif has_person(req_classes) and not is_coco_model(req_classes):
            person_path = requested
            person_classes = req_classes
            if accessory_path:
                return ModelResolution(
                    mode="dual",
                    person_path=person_path,
                    accessory_path=accessory_path,
                    person_classes=person_classes,
                    accessory_classes=_accessory_class_names(accessory_classes),
                    accessory_active=True,
                    requested_path=requested,
                )
            return ModelResolution(
                mode="person_only",
                person_path=person_path,
                accessory_path=None,
                person_classes=person_classes,
                accessory_classes=tuple(),
                accessory_active=False,
                requested_path=requested,
            )

    person_path, person_classes = _pick_person_fallback(cache)

    if accessory_path:
        return ModelResolution(
            mode="dual",
            person_path=person_path,
            accessory_path=accessory_path,
            person_classes=person_classes,
            accessory_classes=_accessory_class_names(accessory_classes),
            accessory_active=True,
            requested_path=requested,
        )

    return ModelResolution(
        mode="person_only",
        person_path=person_path,
        accessory_path=None,
        person_classes=person_classes,
        accessory_classes=tuple(),
        accessory_active=False,
        requested_path=requested,
    )


def person_class_filter(model) -> list[int]:
    """Return YOLO class filter list for person-only inference."""
    for idx, name in enumerate(model_class_names(model)):
        if match_hud_slot(name) == "person":
            return [idx]
    return [0]


def format_model_report(resolution: ModelResolution) -> str:
    acc_classes = ", ".join(resolution.accessory_classes) if resolution.accessory_classes else "—"
    status_text = {
        "active": "ACTIVE",
        "invalid": "LOADED BUT REQUIRED CLASSES MISSING",
        "not_found": "NOT FOUND",
        "disabled": "DISABLED IN CONFIG",
    }[resolution.accessory_status]

    lines = [
        f"Person model: {resolution.person_label}",
        f"Accessory model: {resolution.accessory_label}",
        f"Accessory AI: {status_text}",
        f"Person classes: {', '.join(resolution.person_classes)}",
        f"Accessory classes: {acc_classes}",
    ]
    if resolution.accessory_path and not resolution.class_validation["valid"]:
        lines += ["", format_class_validation(resolution.accessory_classes)]
    return "\n".join(lines)


ZERO_SHOT_WARNING = (
    "Zero-shot accessory mode is ON. These figures come from YOLO-World "
    "prompted with text labels, not from a model trained on cap / mask / "
    "glasses / headphones. Treat them as an unvalidated estimate: on this "
    "project's own sample media this mode found some caps and glasses but no "
    "masks and no headphones at all. For results you intend to defend, train "
    f"and install {DEFAULT_CUSTOM_PATH}."
)


def load_zero_shot_detector():
    """
    Build the experimental open-vocabulary detector, or return None.

    Returns None whenever the mode is disabled in config, the weights are
    absent, or loading fails -- so no caller can accidentally end up with
    zero-shot output it did not explicitly ask for.
    """
    try:
        from app_config import (
            zero_shot_confidence, zero_shot_enabled, zero_shot_imgsz,
            zero_shot_model_path,
        )
    except Exception:
        return None

    if not zero_shot_enabled():
        return None

    try:
        from zero_shot_accessory_detector import ZeroShotAccessoryDetector
        detector = ZeroShotAccessoryDetector(
            model_path=_absolute(zero_shot_model_path()),
            confidence=zero_shot_confidence(),
            imgsz=zero_shot_imgsz(),
        )
    except Exception as exc:
        logger.error("Zero-shot detector could not be created (%s).", exc)
        return None

    return detector if detector.available else None


ACCESSORY_OFFLINE_MESSAGE = (
    "Accessory AI model not loaded. Person detection and tracking are "
    "available, but real cap, mask, glasses and headphones detection "
    "requires a trained accessory model."
)


def format_status_banner(
    resolution: ModelResolution,
    person_loaded: bool = True,
    tracking_active: bool = True,
) -> str:
    """The startup diagnostic described in the project brief."""
    try:
        from app_config import (
            torch_device, use_mock_accessories, zero_shot_enabled,
        )
        device = torch_device()
        mock_on = use_mock_accessories()
        zs_on = zero_shot_enabled()
    except Exception:
        device, mock_on, zs_on = "cpu", False, False

    acc_state = {
        "active": "LOADED",
        "invalid": "INVALID",
        "not_found": "NOT FOUND",
        "disabled": "DISABLED",
    }[resolution.accessory_status]

    acc_classes = (
        ", ".join(resolution.accessory_classes)
        if resolution.accessory_classes else "—"
    )

    return "\n".join([
        "=== AEGIS AI STATUS ===",
        "",
        "Person model:",
        f"  {'LOADED' if person_loaded else 'FAILED'}",
        f"  {resolution.person_label}",
        f"  device: {device}",
        "",
        "Accessory model:",
        f"  {acc_state}",
        f"  {resolution.accessory_path or resolution.requested_path}",
        f"  classes: {acc_classes}",
        "",
        "Tracking:",
        f"  {'ACTIVE' if tracking_active else 'FAILED'}",
        "",
        "Mock accessory detection:",
        f"  {'ON' if mock_on else 'OFF'}",
        "",
        "Zero-shot accessory mode:",
        f"  {'ON (EXPERIMENTAL, UNTRAINED)' if zs_on else 'OFF'}",
        "",
        "Accessory AI:",
        f"  {accessory_ai_label(resolution, zs_on)}",
        "",
        "=======================",
    ])


def accessory_ai_label(resolution: ModelResolution, zero_shot_on: bool) -> str:
    """
    Status text for the accessory pipeline.

    'ACTIVE' is reserved for a validated trained model. Zero-shot output gets
    its own label so it can never be mistaken for a trained result.
    """
    if resolution.accessory_ai_active:
        return "ACTIVE (trained model)"
    if zero_shot_on:
        return "ZERO-SHOT (EXPERIMENTAL - not a trained model)"
    return "OFFLINE"


def log_status_banner(
    resolution: ModelResolution,
    person_loaded: bool = True,
    tracking_active: bool = True,
) -> str:
    banner = format_status_banner(resolution, person_loaded, tracking_active)
    for line in banner.splitlines():
        logger.info("%s", line)
    return banner


def _absolute(model_path: str) -> str:
    return str(PROJECT_ROOT / model_path) if not os.path.isabs(model_path) else model_path


def load_person_model(model_path: str):
    """
    Load the person detector. Raises on a genuinely unusable weight file so
    the caller can surface it, since person detection is not optional.
    """
    from ultralytics import YOLO

    model = YOLO(_absolute(model_path))

    # Pin the device explicitly so a machine without CUDA falls back to CPU
    # rather than failing mid-inference.
    try:
        from app_config import torch_device
        device = torch_device()
        if device == "cpu":
            model.to("cpu")
    except Exception as exc:
        logger.debug("Could not pin person model device (%s).", exc)

    return model


def load_accessory_detector(
    model_path: str,
    confidence: float | None = None,
    iou_threshold: float | None = None,
    imgsz: int | None = None,
):
    """
    Build the accessory detector. Never raises: a missing or corrupt weight
    file yields a detector with available=False so the dashboard can report
    the accessory AI as offline and keep running.
    """
    from accessory_detector import AccessoryDetector

    try:
        from app_config import (
            accessory_confidence, accessory_imgsz, accessory_iou,
        )
        confidence = accessory_confidence() if confidence is None else confidence
        iou_threshold = accessory_iou() if iou_threshold is None else iou_threshold
        imgsz = accessory_imgsz() if imgsz is None else imgsz
    except Exception:
        confidence = 0.35 if confidence is None else confidence
        iou_threshold = 0.50 if iou_threshold is None else iou_threshold
        imgsz = 960 if imgsz is None else imgsz

    return AccessoryDetector(
        model_path=_absolute(model_path),
        confidence=confidence,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )
