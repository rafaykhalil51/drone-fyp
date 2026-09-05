"""
accessory_detector.py
---------------------
Loads a dedicated YOLO model (accessory_best.pt) trained on:
  - cap (class 0)
  - mask (class 1)
  - glasses (class 2)
  - headphones (class 3)

Runs inference each frame alongside the person detector and outputs
standardized accessory detections for association.py.
"""

import os
import logging
import numpy as np

from model_loader import match_hud_slot

logger = logging.getLogger(__name__)


def _class_name_list(model_or_result) -> list[str]:
    """Ordered class labels from a YOLO model or result (dict or list names)."""
    names = getattr(model_or_result, "names", None)
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names.keys())]
    if names:
        return [str(n) for n in names]
    return []

# Standard target classes
CANONICAL_CLASSES = ["cap", "mask", "glasses", "headphones"]


class AccessoryDetector:
    """
    YOLO-based accessory detector using trained accessory_best.pt weights.
    """

    # Distinguishes this from ZeroShotAccessoryDetector. Only a detector with
    # is_trained=True may be reported as "ACCESSORY AI ACTIVE".
    is_trained = True
    source_label = "trained custom model"

    def __init__(
        self,
        model_path: str = "models/accessory_best.pt",
        confidence: float = 0.35,
        iou_threshold: float = 0.50,
        imgsz: int = 960,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou_threshold
        self.imgsz = imgsz

        self.model = None
        self.is_custom = False
        self.available = False
        # True only when the weights emit at least one box on a real image.
        # A 6-image overfit checkpoint can load and list the right classes
        # while detecting nothing — that must not be reported as working AI.
        self.usable = False
        self.class_names: list[str] = []
        self.load_error: str | None = None
        self.validation: dict | None = None

        if not os.path.exists(model_path):
            self.load_error = "not_found"
            logger.warning(
                "Accessory model weights '%s' not found — accessory AI OFFLINE. "
                "Person detection and tracking are unaffected.",
                model_path,
            )
            return

        # A corrupt or non-YOLO .pt must not take the application down.
        try:
            from ultralytics import YOLO
            logger.info("Loading trained accessory model from: '%s'", model_path)
            model = YOLO(model_path)
        except Exception as exc:
            self.load_error = f"invalid: {type(exc).__name__}: {exc}"
            logger.error(
                "Accessory model '%s' could not be loaded (%s) — accessory AI OFFLINE.",
                model_path, exc,
            )
            return

        from model_loader import validate_accessory_classes

        self.class_names = _class_name_list(model)
        self.validation = validate_accessory_classes(self.class_names)

        logger.info("Accessory model loaded:\n%s", model_path)
        logger.info("Classes:\n%s", model.names)

        if not self.validation["valid"]:
            # Loaded, but cannot supply every required class. Refuse to serve
            # detections rather than report partial, misleading statistics.
            self.load_error = "missing_classes"
            logger.error(
                "Accessory model loaded but required classes are missing: %s. "
                "Accessory AI OFFLINE.",
                ", ".join(self.validation["missing"]),
            )
            return

        self.model = model
        self.is_custom = True
        self.available = True

        try:
            from app_config import torch_device
            if torch_device() == "cpu":
                self.model.to("cpu")
        except Exception as exc:
            logger.debug("Could not pin accessory model device (%s).", exc)

        if not self._sanity_emits_detections():
            self.usable = False
            self.load_error = "no_detections"
            logger.error(
                "Accessory model '%s' loaded and has the required classes, "
                "but it emits no detections even at conf=0.01. The weights "
                "are not usable for inference (usually too little training "
                "data). Accessory AI will fall back rather than show empty "
                "counts as if the model were working.",
                model_path,
            )
        else:
            self.usable = True

        logger.info(
            "AccessoryDetector ready (model='%s', conf=%.2f, iou=%.2f, imgsz=%d, usable=%s)",
            model_path, confidence, iou_threshold, imgsz, self.usable,
        )

    def _sanity_emits_detections(self) -> bool:
        """True if the model can produce at least one box on a known image."""
        if self.model is None:
            return False
        import cv2
        from pathlib import Path

        root = Path(__file__).resolve().parent
        candidates = [
            root / "dataset" / "images" / "val" / "Classroom3.jpg",
            root / "dataset" / "images" / "train" / "Classroom2.jpg",
            root / "uploads" / "classroom_sample.jpg",
            root / "training_photos" / "Classroom3_orig.jpeg",
        ]
        for path in candidates:
            if not path.is_file():
                continue
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            try:
                result = self.model.predict(
                    frame, conf=0.01, imgsz=640, verbose=False,
                )[0]
            except Exception as exc:
                logger.debug("Sanity predict failed on %s (%s).", path.name, exc)
                continue
            if result.boxes is not None and len(result.boxes) > 0:
                return True
        return False

    def detect(self, frame: np.ndarray, person_boxes: list = None, imgsz: int = None) -> list[dict]:
        """
        Run accessory detection on frame.

        Parameters
        ----------
        frame        : np.ndarray image in BGR format
        person_boxes : optional list of person bounding boxes
        imgsz        : input image size for inference

        Returns
        -------
        list of dicts: [{"class_name": str, "confidence": float, "xyxy": list[int]}]
        """
        # No model, an invalid model, or a model missing required classes all
        # yield zero detections. Accessory counts stay unavailable rather than
        # being invented.
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
            # A single bad frame must not abort the whole video run.
            logger.warning("Accessory inference failed on a frame (%s); skipping.", exc)
            return []

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                raw_cls_id = int(box.cls[0])
                # result.names is normally a dict, but tolerate a list so a
                # differently-exported model cannot raise AttributeError here.
                names = result.names
                if isinstance(names, dict):
                    raw_name = str(names.get(raw_cls_id, raw_cls_id))
                elif 0 <= raw_cls_id < len(names):
                    raw_name = str(names[raw_cls_id])
                else:
                    raw_name = str(raw_cls_id)
                raw_name = raw_name.lower()
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

                canonical_name = self._map_class_name(raw_name)
                if canonical_name:
                    detections.append({
                        "class_name": canonical_name,
                        "raw_class": raw_name,
                        "confidence": round(conf, 3),
                        "xyxy": xyxy,
                    })

        return detections

    def _map_class_name(self, raw_name: str) -> str | None:
        """
        Map a trained model's class label to one of CANONICAL_CLASSES.

        Delegates to model_loader.match_hud_slot so the dashboard, the CLI and
        this detector all agree on what a label means. The previous local
        implementation used substring tests, which mapped the negative classes
        that public datasets ship ('without_mask', 'no_helmet') onto the
        positive accessory -- counting people who were NOT wearing one.
        """
        slot = match_hud_slot(raw_name)
        return slot if slot in CANONICAL_CLASSES else None
