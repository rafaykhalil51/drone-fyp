import logging
logger = logging.getLogger(__name__)

class PersonTracker:
    def __init__(self, model_path, confidence, iou_threshold,
                 tracker_config, persist, person_class_id=0):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou_threshold
        self.tracker_config = tracker_config
        self.persist = persist
        self.person_class_id = person_class_id
        logger.info("PersonTracker ready model='%s' tracker='%s'",
                    model_path, tracker_config)

    def track(self, frame, imgsz: int = 480):
        results = self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=imgsz,
            classes=[self.person_class_id],
            tracker=self.tracker_config,
            persist=self.persist,
            verbose=False,
        )
        tracks = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            if boxes.id is None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls != self.person_class_id:
                        continue
                    tracks.append({
                        "track_id": -1,
                        "xyxy": box.xyxy[0].cpu().numpy().astype(int).tolist(),
                        "confidence": float(box.conf[0]),
                        "class_id": cls,
                    })
                continue
            for box in boxes:
                cls = int(box.cls[0])
                if cls != self.person_class_id:
                    continue
                track_id = int(box.id[0]) if box.id is not None else -1
                tracks.append({
                    "track_id": track_id,
                    "xyxy": box.xyxy[0].cpu().numpy().astype(int).tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": cls,
                })
        return tracks
