import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class TrackState:
    track_id: int
    first_frame: int
    last_frame: int
    last_xyxy: list
    confidence: float
    crossed_line: bool = False
    cross_direction: Optional[str] = None
    frame_count: int = 1

class StateManager:
    def __init__(self):
        self._states = {}
        self._total_seen = set()

    def update(self, frame_idx, tracks):
        for t in tracks:
            tid = t["track_id"]
            if tid < 0:
                continue
            self._total_seen.add(tid)
            if tid not in self._states:
                self._states[tid] = TrackState(
                    track_id=tid, first_frame=frame_idx, last_frame=frame_idx,
                    last_xyxy=t["xyxy"], confidence=t["confidence"])
            else:
                s = self._states[tid]
                s.last_frame = frame_idx
                s.last_xyxy  = t["xyxy"]
                s.confidence = t["confidence"]
                s.frame_count += 1

    def mark_crossed(self, track_id, direction):
        if track_id in self._states:
            self._states[track_id].crossed_line = True
            self._states[track_id].cross_direction = direction

    def get(self, track_id):
        return self._states.get(track_id)

    @property
    def active_count(self):
        return len(self._states)

    @property
    def total_unique(self):
        return len(self._total_seen)

    def all_states(self):
        return dict(self._states)
