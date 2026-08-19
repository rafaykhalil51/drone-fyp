import logging
from association import box_bottom_center, side_of_line

logger = logging.getLogger(__name__)

class LineCounter:
    def __init__(self, line_cfg, state_manager):
        self.p1 = (float(line_cfg["x1"]), float(line_cfg["y1"]))
        self.p2 = (float(line_cfg["x2"]), float(line_cfg["y2"]))
        self.state_mgr = state_manager
        self._prev_side = {}
        self.count_in  = 0
        self.count_out = 0

    def update(self, tracks):
        for t in tracks:
            tid = t["track_id"]
            if tid < 0:
                continue
            pt = box_bottom_center(t["xyxy"])
            current_side = side_of_line(pt, self.p1, self.p2)
            if tid in self._prev_side:
                prev = self._prev_side[tid]
                if prev != current_side:
                    if current_side == 1:
                        self.count_in += 1
                        self.state_mgr.mark_crossed(tid, "in")
                    else:
                        self.count_out += 1
                        self.state_mgr.mark_crossed(tid, "out")
            self._prev_side[tid] = current_side

    @property
    def total(self):
        return self.count_in + self.count_out
