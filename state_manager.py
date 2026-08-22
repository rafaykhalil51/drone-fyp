import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Any

logger = logging.getLogger(__name__)

ACCESSORY_KEYS = ("mask", "cap", "glasses", "headphones")


@dataclass
class TrackState:
    """
    Complete per-track state, including temporal voting for accessories.

    Vote lists
    ----------
    Each list stores one bool per observed frame: True if that accessory
    was detected on this track in that frame, False otherwise.
    A deque capped at *window* keeps only the most recent observations so
    memory stays bounded even for very long videos.

    Final flags
    -----------
    final_mask, final_cap, final_glasses, final_headphones are set by
    apply_temporal_voting() and reflect the *stable* accessory decision
    (appeared in >= threshold of the last *window* observations).
    """
    # ── tracking basics ───────────────────────────────────────────────────
    track_id:    int
    first_frame: int
    last_frame:  int
    last_xyxy:   list
    confidence:  float
    crossed_line:     bool          = False
    cross_direction:  Optional[str] = None
    frame_count: int                = 1

    # ── accessory vote lists (one bool per frame observation) ─────────────
    mask_votes:       list = field(default_factory=list)
    cap_votes:        list = field(default_factory=list)
    glasses_votes:    list = field(default_factory=list)
    headphones_votes: list = field(default_factory=list)

    # ── stable accessory decisions (set by apply_temporal_voting) ─────────
    final_mask:       bool = False
    final_cap:        bool = False
    final_glasses:    bool = False
    final_headphones: bool = False

    def vote_lists(self) -> dict[str, list]:
        """Convenience accessor keyed by accessory name."""
        return {
            "mask":       self.mask_votes,
            "cap":        self.cap_votes,
            "glasses":    self.glasses_votes,
            "headphones": self.headphones_votes,
        }

    def final_flags(self) -> dict[str, bool]:
        """Return the current stable accessory decisions as a dict."""
        return {
            "mask":       self.final_mask,
            "cap":        self.final_cap,
            "glasses":    self.final_glasses,
            "headphones": self.final_headphones,
        }

    def active_finals(self) -> list[str]:
        """Return names of accessories whose final flag is True."""
        return [k for k, v in self.final_flags().items() if v]


class StateManager:
    """
    Maintains per-track state across the lifetime of a pipeline run.

    Key additions over the previous version:
      - update_state(track_id, frame_associations, frame_index)
            Appends one evidence observation per accessory class.
      - apply_temporal_voting(track_id, window, threshold)
            Sets final_* flags using a sliding-window majority vote.
    """

    def __init__(self):
        self._states: dict[int, TrackState] = {}
        self._total_seen: set[int] = set()

    # ── existing public API (unchanged) ────────────────────────────────────
    def update(self, frame_idx: int, tracks: list[dict]) -> None:
        """Update basic tracking state from a list of tracker results."""
        for t in tracks:
            tid = t["track_id"]
            if tid < 0:
                continue
            self._total_seen.add(tid)
            if tid not in self._states:
                self._states[tid] = TrackState(
                    track_id=tid,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    last_xyxy=t["xyxy"],
                    confidence=t["confidence"],
                )
                logger.debug("New track ID=%d at frame %d", tid, frame_idx)
            else:
                s = self._states[tid]
                s.last_frame  = frame_idx
                s.last_xyxy   = t["xyxy"]
                s.confidence  = t["confidence"]
                s.frame_count += 1

    def mark_crossed(self, track_id: int, direction: str) -> None:
        if track_id in self._states:
            self._states[track_id].crossed_line     = True
            self._states[track_id].cross_direction  = direction

    def get(self, track_id: int) -> Optional[TrackState]:
        return self._states.get(track_id)

    @property
    def active_count(self) -> int:
        return len(self._states)

    @property
    def total_unique(self) -> int:
        return len(self._total_seen)

    def all_states(self) -> dict[int, TrackState]:
        return dict(self._states)

    # ── new accessory-voting API ───────────────────────────────────────────
    def update_state(
        self,
        track_id: int,
        frame_associations: list[dict[str, Any]],
        frame_index: int,
    ) -> None:
        """
        Record accessory evidence for *track_id* from a single frame.

        Parameters
        ----------
        track_id           : persistent track ID from the tracker
        frame_associations : list of accessory dicts associated to this
                             track in this frame  (may be empty).
                             Each dict must have a 'class_name' key.
        frame_index        : current frame number (stored as last_seen_frame)
        """
        state = self._states.get(track_id)
        if state is None:
            logger.debug(
                "update_state: track_id=%d not yet in states (will be created "
                "on next update() call)", track_id
            )
            return

        # Track the frame this state was last updated via accessory evidence
        state.last_frame = frame_index

        # Collect which accessory classes were seen this frame
        seen_this_frame: set[str] = {
            acc["class_name"]
            for acc in frame_associations
            if "class_name" in acc
        }

        # Append one bool observation per accessory class
        for key in ACCESSORY_KEYS:
            state.vote_lists()[key].append(key in seen_this_frame)

        logger.debug(
            "update_state  track=%d  frame=%d  seen=%s",
            track_id, frame_index, sorted(seen_this_frame) or "none",
        )

    def apply_temporal_voting(
        self,
        track_id: int,
        window: int   = 30,
        threshold: float = 0.35,
        latch: bool = True,
    ) -> dict[str, bool]:
        """
        Evaluate sliding-window and cumulative evidence for each accessory class
        and update the corresponding final_* flag on the track's state.

        With latch=True, once an accessory is confirmed on a track, it remains
        confirmed even during momentary occlusions or head turns.
        """
        state = self._states.get(track_id)
        if state is None:
            logger.warning("apply_temporal_voting: unknown track_id=%d", track_id)
            return {}

        results: dict[str, bool] = {}

        for key in ACCESSORY_KEYS:
            votes = state.vote_lists()[key]
            current_flag = getattr(state, f"final_{key}", False)

            if latch and current_flag:
                # Already confirmed for this track
                decision = True
            else:
                recent = votes[-window:] if len(votes) >= window else votes
                if not recent:
                    decision = False
                else:
                    positive_rate = sum(recent) / len(recent)
                    # Confirmed if rate in window >= threshold OR at least 5 cumulative detections
                    decision = (positive_rate >= threshold) or (sum(votes) >= 5)

            setattr(state, f"final_{key}", decision)
            results[key] = decision

        return results

    def apply_voting_all(
        self,
        window: int = 30,
        threshold: float = 0.6,
    ) -> dict[int, dict[str, bool]]:
        """
        Convenience: call apply_temporal_voting for every known track.

        Returns
        -------
        dict mapping track_id -> {accessory -> bool}
        """
        return {
            tid: self.apply_temporal_voting(tid, window, threshold)
            for tid in self._states
        }

    def accessory_summary(self) -> dict[int, list[str]]:
        """
        Return a summary of confirmed accessories per track
        (only tracks with at least one final_* flag set).
        """
        return {
            tid: s.active_finals()
            for tid, s in self._states.items()
            if s.active_finals()
        }
