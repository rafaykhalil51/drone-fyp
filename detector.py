import logging
logger = logging.getLogger(__name__)

class PersonDetector:
    def __init__(self, model_path, confidence, iou_threshold, person_class_id=0):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou_threshold
        self.person_class_id = person_class_id
        logger.info("PersonDetector loaded '%s'", model_path)

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.confidence, iou=self.iou,
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
                detections.append({
                    "xyxy": box.xyxy[0].cpu().numpy().astype(int).tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": cls,
                })
        return detections
