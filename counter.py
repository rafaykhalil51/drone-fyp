import logging
from association import box_bottom_center, side_of_line
from state_manager import ACCESSORY_KEYS

logger = logging.getLogger(__name__)


class LineCounter:
    """Virtual-line IN/OUT people counter (unchanged)."""

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


class AccessoryCounter:
    """
    Derives summary counts from StateManager's temporal-voting final flags.

    Counts are computed on demand from the stable final_* booleans, so
    this class holds no mutable state itself -- call compute() whenever
    you need fresh numbers (e.g. once at run end, or periodically).

    Attributes set after compute()
    --------------------------------
    total_unique_persons : int
        Number of distinct track IDs ever seen.
    totals : dict[str, int]
        Per-accessory count of tracks where final_<key> is True.
        Keys: mask, cap, glasses, headphones.
    """

    def __init__(self):
        self.total_unique_persons: int = 0
        self.totals: dict = {key: 0 for key in ACCESSORY_KEYS}
        self._computed = False

    def compute(self, state_manager) -> "AccessoryCounter":
        """
        Walk all TrackState objects in *state_manager* and tally:
          - total unique persons (distinct track IDs ever tracked)
          - per-accessory count of tracks whose final_* flag is True

        Parameters
        ----------
        state_manager : StateManager
            Must have had apply_temporal_voting() called on each track
            before compute() is invoked, otherwise final_* flags will
            all be False.

        Returns
        -------
        self  (fluent, so you can do counter.compute(sm).totals)
        """
        self.total_unique_persons = state_manager.total_unique
        self.totals = {key: 0 for key in ACCESSORY_KEYS}

        for state in state_manager.all_states().values():
            flags = state.final_flags()
            for key in ACCESSORY_KEYS:
                if flags.get(key, False):
                    self.totals[key] += 1

        self._computed = True
        logger.info(
            "AccessoryCounter.compute()  unique_persons=%d  %s",
            self.total_unique_persons,
            "  ".join(f"{k}={v}" for k, v in self.totals.items()),
        )
        return self

    def as_dict(self) -> dict:
        """Return totals as a flat dict for JSON serialisation."""
        return {
            "total_unique_persons": self.total_unique_persons,
            **{f"total_{k}": v for k, v in self.totals.items()},
        }
