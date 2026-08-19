"""
real_accessory_detector.py
--------------------------
Uses YOLO-World (open-vocabulary object detector) to detect real accessories
(cap, hat, mask, glasses, headphones) in images and video frames.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Canonical accessory classes we map detected labels into
ACCESSORY_MAPPING = {
    "cap": "cap",
    "hat": "cap",
    "baseball cap": "cap",
    "mask": "mask",
    "face mask": "mask",
    "surgical mask": "mask",
    "glasses": "glasses",
    "eyeglasses": "glasses",
    "sunglasses": "glasses",
    "spectacles": "glasses",
    "headphones": "headphones",
    "over-ear headphones": "headphones",
    "headset": "headphones",
    "earphones": "headphones",
}

# The class vocabulary we instruct YOLO-World to search for
VOCABULARY = [
    "cap", "hat", "baseball cap",
    "face mask", "mask",
    "glasses", "eyeglasses", "sunglasses",
    "headphones", "over-ear headphones", "headset"
]


class RealAccessoryDetector:
    """
    Detects actual accessories using YOLO-World.
    """

    def __init__(self, model_path: str = "yolov8s-world.pt", confidence: float = 0.25):
        from ultralytics import YOLO
        logger.info("Loading RealAccessoryDetector (%s)...", model_path)
        self.model = YOLO(model_path)
        self.model.set_classes(VOCABULARY)
        self.confidence = confidence
        logger.info("RealAccessoryDetector ready with vocabulary: %s", VOCABULARY)

    def detect(self, person_boxes: list[list[int]], frame: np.ndarray = None) -> list[dict]:
        """
        Detect accessories in the frame and map them to standard classes.

        Parameters
        ----------
        person_boxes : list of [x1, y1, x2, y2]
        frame        : np.ndarray image (BGR)

        Returns
        -------
        list of dicts: [{"class_name": str, "confidence": float, "xyxy": list[int]}]
        """
        if frame is None:
            return []

        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        accessories = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                raw_name = r.names[int(box.cls[0])].lower()
                canonical = ACCESSORY_MAPPING.get(raw_name)
                if not canonical:
                    continue

                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

                accessories.append({
                    "class_name": canonical,
                    "raw_class": raw_name,
                    "confidence": round(conf, 3),
                    "xyxy": xyxy,
                })

        return accessories
