"""
app_config.py
-------------
One reader for config.yaml, shared by the Streamlit dashboard, the CLI
pipeline and the Flask app so there is a single configuration system.

Every value has a fallback, so a missing or malformed config.yaml degrades to
sane defaults instead of taking the application down.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Defaults mirror config.yaml and are used when a key or the file is missing.
DEFAULT_PERSON_MODEL = "yolov8n.pt"
DEFAULT_ACCESSORY_MODEL = "models/accessory_best.pt"
DEFAULT_ACCESSORY_CONF = 0.35
DEFAULT_ACCESSORY_IMGSZ = 960


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Parse config.yaml once. Returns {} when unreadable."""
    try:
        import yaml
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            logger.warning("config.yaml did not parse to a mapping; using defaults.")
            return {}
        return cfg
    except FileNotFoundError:
        logger.warning("config.yaml not found at %s; using defaults.", CONFIG_PATH)
        return {}
    except Exception as exc:                       # malformed YAML, bad perms
        logger.warning("Could not read config.yaml (%s); using defaults.", exc)
        return {}


def _section(name: str) -> dict:
    value = load_config().get(name, {})
    return value if isinstance(value, dict) else {}


# ── person detection ─────────────────────────────────────────────────────
def person_model_path() -> str:
    """
    Configured person model. This is a pretrained COCO detector and is kept
    entirely separate from accessory detection.
    """
    return str(_section("detection").get("model_path") or DEFAULT_PERSON_MODEL)


def person_confidence() -> float:
    try:
        return float(_section("detection").get("confidence", 0.40))
    except (TypeError, ValueError):
        return 0.40


# ── accessory detection ──────────────────────────────────────────────────
def accessory_enabled() -> bool:
    return bool(_section("accessories").get("enabled", True))


def accessory_model_path() -> str:
    return str(_section("accessories").get("model_path") or DEFAULT_ACCESSORY_MODEL)


def accessory_model_abspath() -> Path:
    """Absolute path, resolved against the project root for relative values."""
    raw = accessory_model_path()
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def accessory_model_exists() -> bool:
    return accessory_model_abspath().is_file()


def accessory_confidence() -> float:
    try:
        return float(_section("accessories").get("confidence", DEFAULT_ACCESSORY_CONF))
    except (TypeError, ValueError):
        return DEFAULT_ACCESSORY_CONF


def accessory_imgsz() -> int:
    try:
        return int(_section("accessories").get("imgsz", DEFAULT_ACCESSORY_IMGSZ))
    except (TypeError, ValueError):
        return DEFAULT_ACCESSORY_IMGSZ


def accessory_iou() -> float:
    try:
        return float(_section("accessories").get("iou_threshold", 0.50))
    except (TypeError, ValueError):
        return 0.50


# ── zero-shot fallback (experimental, opt-in) ────────────────────────────
def _zero_shot_section() -> dict:
    value = _section("accessories").get("zero_shot", {})
    return value if isinstance(value, dict) else {}


def zero_shot_enabled() -> bool:
    """
    Opt-in only. Defaults to False so a missing or malformed config can never
    silently turn untrained guessing back on.
    """
    return bool(_zero_shot_section().get("enabled", False))


def zero_shot_model_path() -> str:
    return str(_zero_shot_section().get("model_path") or "yolov8s-world.pt")


def zero_shot_confidence() -> float:
    try:
        return float(_zero_shot_section().get("confidence", 0.25))
    except (TypeError, ValueError):
        return 0.25


def zero_shot_imgsz() -> int:
    try:
        return int(_zero_shot_section().get("imgsz", 640))
    except (TypeError, ValueError):
        return 640


def head_fraction() -> float:
    try:
        return float(_section("accessories").get("head_fraction", 0.40))
    except (TypeError, ValueError):
        return 0.40


# ── mock / sample detections ─────────────────────────────────────────────
def use_mock_accessories() -> bool:
    """
    False means no synthetic accessory detection may be produced anywhere.
    Defaults to False so a missing config can never silently enable fakes.
    """
    return bool(load_config().get("use_mock_accessories", False))


# ── device ───────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def torch_device() -> str:
    """'cuda' when a GPU is genuinely usable, otherwise 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
    except Exception as exc:
        logger.debug("CUDA probe failed (%s); using CPU.", exc)
    return "cpu"
