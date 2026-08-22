import logging
logger = logging.getLogger(__name__)

class PersonDetector:
    def __init__(self, model_path, confidence, iou_threshold,
                 person_class_id=0, accessory_detector=None):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou_threshold
        self.person_class_id = person_class_id
        self.accessory_detector = accessory_detector
        logger.info("PersonDetector loaded '%s'  mock_accessories=%s",
                    model_path, accessory_detector is not None)

    def detect(self, frame, imgsz: int = 480):
        results = self.model.predict(frame, conf=self.confidence, iou=self.iou,
                                     imgsz=imgsz,
                                     classes=[self.person_class_id], verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0])
                if cls != self.person_class_id:
                    continue
                det = {
                    "xyxy": box.xyxy[0].cpu().numpy().astype(int).tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": cls,
                    "accessories": [],
                }
                detections.append(det)

        # Attach mock (or real) accessory detections to each person
        if self.accessory_detector is not None and detections:
            person_boxes = [d["xyxy"] for d in detections]
            all_accs = self.accessory_detector.detect(person_boxes)
            # Associate each accessory with the person whose bbox it belongs to
            for acc in all_accs:
                ax1, ay1, ax2, ay2 = acc["xyxy"]
                acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
                for det in detections:
                    dx1, dy1, dx2, dy2 = det["xyxy"]
                    if dx1 <= acx <= dx2 and dy1 <= acy <= dy2:
                        det["accessories"].append(acc)
                        break

        return detections
