"""
accessory_crops.py
------------------
Accessory detection on per-person head crops.

Running the accessory model on the whole frame wastes most of its input on
background and leaves glasses, masks and earbuds only a few pixels wide. Each
person's head-and-shoulders region is cropped and upscaled instead, which is
both faster and far more sensitive to small accessories. Boxes are returned in
full-frame coordinates so association and voting work unchanged.
"""

from __future__ import annotations

CROP_IMGSZ = 320           # inference size for each head crop
MAX_CROPS_PER_FRAME = 8    # largest persons are processed first
MIN_PERSON_HEIGHT = 40     # persons shorter than this (px) are too small to judge


def head_crop_box(xyxy, frame_w: int, frame_h: int, min_height: int = MIN_PERSON_HEIGHT):
    """Padded head-and-shoulders crop for one person box, or None if too small."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    if w <= 4 or h < min_height:
        return None
    cx1 = max(0, int(x1 - 0.15 * w))
    cx2 = min(frame_w, int(x2 + 0.15 * w))
    cy1 = max(0, int(y1 - 0.10 * h))
    cy2 = min(frame_h, int(y1 + 0.55 * h))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    return cx1, cy1, cx2, cy2


def detect_on_head_crops(
    detector,
    frame,
    tracks,
    imgsz: int = CROP_IMGSZ,
    max_crops: int = MAX_CROPS_PER_FRAME,
    include_full_frame: bool = False,
) -> list[dict]:
    """
    Run ``detector.detect`` on each person's head crop.

    *detector* is an AccessoryDetector or ZeroShotAccessoryDetector (anything
    exposing ``available`` and ``detect(image, imgsz=...)``).
    """
    if detector is None or not getattr(detector, "available", False) or not tracks:
        return []

    frame_h, frame_w = frame.shape[:2]
    by_size = sorted(
        tracks,
        key=lambda t: (t["xyxy"][2] - t["xyxy"][0]) * (t["xyxy"][3] - t["xyxy"][1]),
        reverse=True,
    )[:max_crops]

    detections = []
    for t in by_size:
        box = head_crop_box(t["xyxy"], frame_w, frame_h)
        if box is None:
            continue
        cx1, cy1, cx2, cy2 = box
        for d in detector.detect(frame[cy1:cy2, cx1:cx2], imgsz=imgsz):
            ax1, ay1, ax2, ay2 = d["xyxy"]
            mapped = dict(d)
            mapped["xyxy"] = [ax1 + cx1, ay1 + cy1, ax2 + cx1, ay2 + cy1]
            detections.append(mapped)

    if include_full_frame:
        detections.extend(detector.detect(frame))
    return detections


def translate_accessories(accs: list, old_box, new_box) -> list:
    """Shift cached accessory boxes from a person's old position to the new one."""
    dx = new_box[0] - old_box[0]
    dy = new_box[1] - old_box[1]
    moved = []
    for a in accs:
        x1, y1, x2, y2 = a["xyxy"]
        b = dict(a)
        b["xyxy"] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        moved.append(b)
    return moved
