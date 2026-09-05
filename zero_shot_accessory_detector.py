"""
zero_shot_accessory_detector.py
-------------------------------
EXPERIMENTAL, OPT-IN accessory detection using YOLO-World.

This is NOT a trained accessory model. YOLO-World is an open-vocabulary
detector: it is handed a list of text prompts at runtime and reports whatever
it thinks matches them. Nothing here was trained on this project's dataset, so
its output is a zero-shot guess, not a measured detection.

Why it exists
-------------
Commit 165754e and earlier silently fell back to this whenever
`accessory_best.pt` was missing -- which was always, because no .pt file was
ever tracked in the repository. Its output was then presented as real
accessory detection. Measured on this project's own sample media, that
fallback found caps and glasses with some success but detected ZERO masks and
ZERO headphones (see diag_legacy_accessory.py).

The mode is therefore kept available for demonstration purposes but is:

  * disabled by default,
  * never reported as "ACCESSORY AI ACTIVE",
  * always labelled ZERO-SHOT / EXPERIMENTAL in the interface,
  * recorded as the detection source in the exported report.

Use a trained models/accessory_best.pt for any result you intend to defend.
"""

from __future__ import annotations

import logging

import numpy as np

from model_loader import match_hud_slot

logger = logging.getLogger(__name__)

# Text prompts handed to YOLO-World. Carried over from commit 165754e, whose
# author had already tuned them, minus "headphones on neck": headphones around
# the neck mean the person is NOT wearing them, so that prompt manufactured
# false positives for the very thing it was meant to measure.
DEFAULT_VOCABULARY = [
    "cap", "hat", "baseball cap", "beanie", "knit cap", "beret",
    "mask", "face mask", "surgical mask",
    "glasses", "eyeglasses", "sunglasses", "spectacles",
    "headphones", "over-ear headphones", "headset", "earphones",
    "wireless headphones", "headphones on head", "black headphones",
    "earmuffs",
]

# Weights that support open-vocabulary prompting.
DEFAULT_MODEL = "yolov8s-world.pt"


class ZeroShotAccessoryDetector:
    """
    Open-vocabulary accessory detector.

    Exposes the same surface as AccessoryDetector (``available``,
    ``detect(frame, person_boxes)`` -> list of dicts with ``class_name`` /
    ``confidence`` / ``xyxy``) so it can drop into the existing association,
    voting and counting stages without changes to them.

    ``is_trained`` is always False. Callers must use it to decide what the
    interface is allowed to claim.
    """

    is_trained = False
    source_label = "zero-shot (YOLO-World, experimental)"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence: float = 0.25,
        iou_threshold: float = 0.50,
        imgsz: int = 640,
        vocabulary: list[str] | None = None,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou_threshold
        self.imgsz = imgsz
        self.vocabulary = list(vocabulary or DEFAULT_VOCABULARY)

        self.model = None
        self.available = False
        self.load_error: str | None = None
        self.class_names: list[str] = []

        import os
        if not os.path.exists(model_path):
            self.load_error = "not_found"
            logger.warning(
                "Zero-shot weights '%s' not found; zero-shot mode unavailable.",
                model_path,
            )
            return

        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            # set_classes builds CLIP text embeddings for the prompts. This is
            # what makes the model "detect" arbitrary words, and what makes the
            # result a guess rather than a trained prediction.
            if not hasattr(model, "set_classes"):
                raise TypeError(
                    f"{model_path} is not an open-vocabulary model "
                    "(no set_classes); use a YOLO-World checkpoint."
                )
            model.set_classes(self.vocabulary)
        except Exception as exc:
            self.load_error = f"invalid: {type(exc).__name__}: {exc}"
            logger.error(
                "Zero-shot model '%s' failed to load (%s); mode unavailable.",
                model_path, exc,
            )
            return

        self.model = model
        self.class_names = list(self.vocabulary)
        self.available = True

        try:
            from app_config import torch_device
            if torch_device() == "cpu":
                self.model.to("cpu")
        except Exception as exc:
            logger.debug("Could not pin zero-shot model device (%s).", exc)

        logger.warning(
            "ZERO-SHOT accessory mode ENABLED (%s, conf=%.2f, imgsz=%d, "
            "%d prompts). Results are UNTRAINED estimates and must not be "
            "reported as trained-model accuracy.",
            model_path, confidence, imgsz, len(self.vocabulary),
        )

    def detect(
        self,
        frame: np.ndarray,
        person_boxes: list | None = None,
        imgsz: int | None = None,
    ) -> list[dict]:
        """
        Run open-vocabulary inference and map prompts to canonical slots.

        Returns the same dict shape the trained detector returns, plus
        ``zero_shot: True`` so downstream code and the exported report can
        tell where a detection came from.
        """
        if frame is None or not self.available or self.model is None:
            return []

        try:
            results = self.model.predict(
                frame,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.imgsz if imgsz is None else imgsz,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("Zero-shot inference failed on a frame (%s); skipping.", exc)
            return []

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            names = result.names
            for box in boxes:
                cls_id = int(box.cls[0])
                if isinstance(names, dict):
                    raw = str(names.get(cls_id, cls_id))
                else:
                    raw = str(names[cls_id]) if cls_id < len(names) else str(cls_id)

                # Same normalisation the trained path uses, so "without_mask"
                # style negatives can never count as wearing an accessory.
                slot = match_hud_slot(raw)
                if slot not in ("cap", "mask", "glasses", "headphones"):
                    continue

                detections.append({
                    "class_name": slot,
                    "raw_class": raw.lower(),
                    "confidence": round(float(box.conf[0]), 3),
                    "xyxy": box.xyxy[0].cpu().numpy().astype(int).tolist(),
                    "zero_shot": True,
                })

        return detections
