"""
visualization.py
----------------
Draws bounding boxes, persistent track IDs, confidence scores,
accessory detection bounding boxes with labels, and a live HUD overlay
displaying Total Persons / Masks / Caps / Glasses / Headphones.
"""

import time
import cv2
import numpy as np
from mock_accessory_detector import ACCESSORY_COLORS

_PALETTE = [
    (255, 56, 56),  (255, 157, 151), (255, 112, 31),
    (255, 178, 29), (207, 210, 49),  (72, 249, 10),
    (146, 204, 23), (61, 219, 134),  (26, 147, 52),
    (0, 212, 187),  (44, 153, 168),  (0, 194, 255),
    (52, 69, 147),  (100, 115, 255), (0, 24, 236),
    (132, 56, 255), (82, 0, 133),    (203, 56, 255),
    (255, 27, 108), (255, 81, 102),
]


def _track_color(tid: int):
    return _PALETTE[tid % len(_PALETTE)]


class Visualizer:
    def __init__(self, vis_cfg: dict = None, counting_cfg: dict = None):
        vis_cfg = vis_cfg or {}
        self.box_color = tuple(vis_cfg.get("box_color", [0, 255, 0]))
        self.id_color = tuple(vis_cfg.get("id_color", [0, 0, 255]))
        self.text_scale = float(vis_cfg.get("text_scale", 0.65))
        self.text_thickness = int(vis_cfg.get("text_thickness", 2))
        self.box_thickness = int(vis_cfg.get("box_thickness", 2))
        self.show_conf = bool(vis_cfg.get("show_confidence", True))
        self.show_fps = bool(vis_cfg.get("show_fps", True))
        self.counting_cfg = counting_cfg
        self._fps_ts = time.time()
        self._fps_val = 0.0
        self._frame_count = 0

    def draw(
        self,
        frame: np.ndarray,
        tracks: list,
        counter=None,
        live_totals: dict = None,
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        if self.counting_cfg and self.counting_cfg.get("enabled"):
            lc = self.counting_cfg["line"]
            cv2.line(out, (lc["x1"], lc["y1"]), (lc["x2"], lc["y2"]), (0, 255, 255), 2)

        for t in tracks:
            tid = t["track_id"]
            x1, y1, x2, y2 = t["xyxy"]
            color = _track_color(tid) if tid >= 0 else self.box_color

            cv2.rectangle(out, (x1, y1), (x2, y2), color, self.box_thickness)

            label = f"ID:{tid}" if tid >= 0 else "?"
            if self.show_conf and "confidence" in t:
                label += f" {t['confidence']:.2f}"

            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness)
            cv2.rectangle(out, (x1, max(0, y1 - th - bl - 4)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                out, label, (x1 + 2, max(th + 2, y1 - bl - 2)),
                cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
                (255, 255, 255), self.text_thickness, cv2.LINE_AA
            )

            for acc in t.get("accessories", []):
                self._draw_accessory(out, acc)

        self._draw_hud(out, live_totals, counter)
        return out

    def _draw_hud(self, frame: np.ndarray, totals: dict, counter=None):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Scale banner height and font size proportionally to video resolution
        scale = max(min(w / 1280.0, 1.6), 0.55)
        bar_height = int(50 * scale)
        font_scale = 0.55 * scale
        font_thick = max(int(2 * scale), 1)
        y_text = int(32 * scale)

        # Translucent dark cyberpunk HUD header
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (6, 9, 19), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        # Neon cyan accent divider line
        cv2.line(frame, (0, bar_height), (w, bar_height), (0, 242, 254), max(int(2 * scale), 1))

        if self.show_fps:
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._fps_ts
            if elapsed >= 1.0:
                self._fps_val = self._frame_count / elapsed
                self._fps_ts = now
                self._frame_count = 0

        hud_items = []
        if totals:
            hud_items.append((f"TARGETS: {totals.get('persons', 0)}", (0, 242, 254)))      # Cyan
            hud_items.append((f"CAPS: {totals.get('caps', 0)}", (3, 183, 255)))            # Amber
            hud_items.append((f"MASKS: {totals.get('masks', 0)}", (0, 133, 251)))          # Orange
            hud_items.append((f"GLASSES: {totals.get('glasses', 0)}", (255, 125, 199)))    # Purple
            hud_items.append((f"HEADPHONES: {totals.get('headphones', 0)}", (135, 255, 0))) # Green
        else:
            hud_items.append(("A.E.G.I.S. VISION HUD // ACTIVE", (0, 242, 254)))

        curr_x = int(14 * scale)
        for text, color in hud_items:
            cv2.putText(frame, text, (curr_x, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            curr_x += tw + int(16 * scale)
            if text != hud_items[-1][0]:
                cv2.circle(frame, (curr_x - int(8 * scale), y_text - int(6 * scale)), int(3 * scale), (100, 116, 139), -1)

        right_str = ""
        if counter is not None:
            right_str += f"IN:{counter.count_in} OUT:{counter.count_out} | "
        if self.show_fps:
            right_str += f"FPS:{self._fps_val:.1f}"

        if right_str:
            (rw, _), _ = cv2.getTextSize(right_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, 1)
            cv2.putText(frame, right_str, (w - rw - int(14 * scale), y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (203, 213, 225), 1, cv2.LINE_AA)

    def _draw_accessory(self, frame: np.ndarray, acc: dict):
        ax1, ay1, ax2, ay2 = acc["xyxy"]
        cls_name = acc["class_name"]
        confidence = acc.get("confidence", 0.0)
        color = ACCESSORY_COLORS.get(cls_name, (255, 255, 0))

        pts = [(ax1, ay1, ax2, ay1), (ax2, ay1, ax2, ay2),
               (ax2, ay2, ax1, ay2), (ax1, ay2, ax1, ay1)]
        for x1, y1, x2, y2 in pts:
            steps = max(abs(x2 - x1), abs(y2 - y1))
            if steps == 0:
                continue
            dash_len = max(4, steps // 8)
            for s in range(0, steps, dash_len * 2):
                t1 = s / steps
                t2 = min((s + dash_len) / steps, 1.0)
                p1 = (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1))
                p2 = (int(x1 + (x2 - x1) * t2), int(y1 + (y2 - y1) * t2))
                cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

        acc_label = f"{cls_name} {confidence:.2f}"
        (tw, th), bl = cv2.getTextSize(acc_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        lx, ly = ax1, ay2 + th + bl + 2
        cv2.rectangle(frame, (lx - 1, ly - th - bl - 2), (lx + tw + 1, ly + 1), color, -1)
        cv2.putText(frame, acc_label, (lx, ly - bl),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
