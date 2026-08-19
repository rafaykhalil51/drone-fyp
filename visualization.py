import time, cv2
import numpy as np
from mock_accessory_detector import ACCESSORY_COLORS

_PALETTE = [
    (255,56,56),(255,157,151),(255,112,31),(255,178,29),(207,210,49),
    (72,249,10),(146,204,23),(61,219,134),(26,147,52),(0,212,187),
    (44,153,168),(0,194,255),(52,69,147),(100,115,255),(0,24,236),
    (132,56,255),(82,0,133),(203,56,255),(255,27,108),(255,81,102),
]

def _track_color(tid):
    return _PALETTE[tid % len(_PALETTE)]

class Visualizer:
    def __init__(self, vis_cfg, counting_cfg=None):
        self.box_color    = tuple(vis_cfg.get("box_color", [0,255,0]))
        self.id_color     = tuple(vis_cfg.get("id_color",  [0,0,255]))
        self.text_scale   = float(vis_cfg.get("text_scale", 0.7))
        self.text_thickness = int(vis_cfg.get("text_thickness", 2))
        self.box_thickness  = int(vis_cfg.get("box_thickness",  2))
        self.show_conf    = bool(vis_cfg.get("show_confidence", True))
        self.show_fps     = bool(vis_cfg.get("show_fps", True))
        self.counting_cfg = counting_cfg
        self._fps_ts      = time.time()
        self._fps_val     = 0.0
        self._frame_count = 0

    def draw(self, frame, tracks, counter=None):
        out = frame.copy()
        if self.counting_cfg and self.counting_cfg.get("enabled"):
            lc = self.counting_cfg["line"]
            cv2.line(out, (lc["x1"],lc["y1"]), (lc["x2"],lc["y2"]), (0,255,255), 2)

        for t in tracks:
            tid = t["track_id"]
            x1,y1,x2,y2 = t["xyxy"]
            color = _track_color(tid) if tid >= 0 else self.box_color

            # Person bounding box
            cv2.rectangle(out, (x1,y1), (x2,y2), color, self.box_thickness)
            label = f"ID:{tid}" if tid >= 0 else "?"
            if self.show_conf:
                label += f"  {t['confidence']:.2f}"
            (tw,th),bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          self.text_scale, self.text_thickness)
            cv2.rectangle(out, (x1, y1-th-bl-4), (x1+tw, y1), color, -1)
            cv2.putText(out, label, (x1, y1-bl-2),
                        cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
                        (255,255,255), self.text_thickness, cv2.LINE_AA)

            # Accessory detections (dashed thin box + label)
            for acc in t.get("accessories", []):
                self._draw_accessory(out, acc)

        if counter is not None:
            cv2.putText(out, f"In:{counter.count_in}  Out:{counter.count_out}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)

        if self.show_fps:
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._fps_ts
            if elapsed >= 1.0:
                self._fps_val = self._frame_count / elapsed
                self._fps_ts  = now
                self._frame_count = 0
            cv2.putText(out, f"FPS:{self._fps_val:.1f}",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2, cv2.LINE_AA)
        return out

    def _draw_accessory(self, frame, acc):
        ax1, ay1, ax2, ay2 = acc["xyxy"]
        cls_name   = acc["class_name"]
        confidence = acc["confidence"]
        color = ACCESSORY_COLORS.get(cls_name, (255, 255, 0))

        # Dashed rectangle (simulate with 4 short line segments)
        pts = [(ax1,ay1,ax2,ay1), (ax2,ay1,ax2,ay2),
               (ax2,ay2,ax1,ay2), (ax1,ay2,ax1,ay1)]
        for x1,y1,x2,y2 in pts:
            # Draw dashes along each edge
            steps = max(abs(x2-x1), abs(y2-y1))
            if steps == 0:
                continue
            dash_len = max(4, steps // 8)
            for s in range(0, steps, dash_len * 2):
                t1 = s / steps
                t2 = min((s + dash_len) / steps, 1.0)
                p1 = (int(x1 + (x2-x1)*t1), int(y1 + (y2-y1)*t1))
                p2 = (int(x1 + (x2-x1)*t2), int(y1 + (y2-y1)*t2))
                cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

        # Small label below the box
        acc_label = f"{cls_name} {confidence:.2f}"
        (tw, th), bl = cv2.getTextSize(acc_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lx, ly = ax1, ay2 + th + bl + 2
        cv2.rectangle(frame, (lx-1, ly-th-bl-2), (lx+tw+1, ly+1), color, -1)
        cv2.putText(frame, acc_label, (lx, ly-bl),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
