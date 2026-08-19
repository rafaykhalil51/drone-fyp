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

logger = logging.getLogger(__name__)

# Standard target classes
CANONICAL_CLASSES = ["cap", "mask", "glasses", "headphones"]


class AccessoryDetector:
    """
    YOLO-based accessory detector using trained accessory_best.pt weights.
    """

    def __init__(
        self,
        model_path: str = "accessory_best.pt",
        confidence: float = 0.30,
        iou_threshold: float = 0.50,
    ):
        from ultralytics import YOLO
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou_threshold

        if not os.path.exists(model_path):
            logger.warning(
                "Accessory model weights '%s' not found! Falling back to 'yolov8s-world.pt' with open-vocabulary classes.",
                model_path,
            )
            fallback_path = "yolov8s-world.pt" if os.path.exists("yolov8s-world.pt") else "yolov8n.pt"
            self.model = YOLO(fallback_path)
            if "world" in fallback_path:
                self.model.set_classes([
                    "cap", "hat", "baseball cap",
                    "mask", "face mask",
                    "glasses", "eyeglasses", "sunglasses",
                    "headphones", "over-ear headphones", "headset"
                ])
            self.is_custom = False
        else:
            logger.info("Loading trained accessory model from: '%s'", model_path)
            self.model = YOLO(model_path)
            self.is_custom = True

        logger.info(
            "AccessoryDetector initialized (model='%s', conf=%.2f, iou=%.2f)",
            model_path, confidence, iou_threshold
        )

    def detect(self, frame: np.ndarray, person_boxes: list = None) -> list[dict]:
        """
        Run accessory detection on frame.

        Parameters
        ----------
        frame        : np.ndarray image in BGR format
        person_boxes : optional list of person bounding boxes

        Returns
        -------
        list of dicts: [{"class_name": str, "confidence": float, "xyxy": list[int]}]
        """
        if frame is None:
            return []

        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                raw_cls_id = int(box.cls[0])
                raw_name = result.names.get(raw_cls_id, str(raw_cls_id)).lower()
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

    def _map_class_name(self, raw_name: str) -> str:
        """Map detected class label to canonical class name."""
        raw_name = raw_name.strip().lower()
        if any(w in raw_name for w in ["cap", "hat", "helmet", "hood"]):
            return "cap"
        elif any(w in raw_name for w in ["mask", "facemask", "respirator"]):
            return "mask"
        elif any(w in raw_name for w in ["glass", "spectacle", "goggle"]):
            return "glasses"
        elif any(w in raw_name for w in ["headphone", "earphone", "headset", "earmuff"]):
            return "headphones"
        elif raw_name in CANONICAL_CLASSES:
            return raw_name
        return None
