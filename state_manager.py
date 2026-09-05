import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Any

logger = logging.getLogger(__name__)

from voting_config import ACCESSORY_KEYS, PHOTO_VOTING, VOTING, VotingConfig

# Backwards-compatible aliases; tune values in voting_config.py.
MIN_OBSERVATIONS = VOTING.min_frames_seen
MIN_POSITIVE_VOTES = VOTING.min_hits


def _vote_decision(
    votes: list,
    accessory: str,
    config: VotingConfig,
    window: int | None,
) -> bool:
    """
    Shared voting rule for sliding-window and end-of-video evaluation.

    True only when the track has enough frames, enough hits, and a hit ratio
    at or above the configured threshold for *accessory*.
    """
    frames_seen = len(votes)
    if frames_seen == 0:
        return False

    hits = sum(votes)
    if hits < config.min_hits:
        return False

    if frames_seen < config.min_frames_seen:
        # Track too short to trust a ratio; demand unanimous evidence.
        return hits == frames_seen

    scope = votes[-window:] if window else votes
    if not scope:
        return False

    ratio = sum(scope) / len(scope)
    return ratio >= config.threshold_for(accessory)


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

    def observation_counts(self) -> dict[str, int]:
        """
        Number of frames each accessory was actually observed on this track.

        e.g. {"cap": 18, "mask": 0, "glasses": 14, "headphones": 0}
        """
        return {key: sum(votes) for key, votes in self.vote_lists().items()}

    # ── temporal voting inputs ────────────────────────────────────────────
    @property
    def frames_seen(self) -> int:
        """Frames in which this track was observed and evidence recorded."""
        return len(self.cap_votes)

    @property
    def cap_hits(self) -> int:
        return sum(self.cap_votes)

    @property
    def mask_hits(self) -> int:
        return sum(self.mask_votes)

    @property
    def glasses_hits(self) -> int:
        return sum(self.glasses_votes)

    @property
    def headphones_hits(self) -> int:
        return sum(self.headphones_votes)

    def hit_counts(self) -> dict[str, int]:
        """{"mask": n, "cap": n, "glasses": n, "headphones": n}"""
        return self.observation_counts()

    def ratio_for(self, accessory: str) -> float:
        """hits / frames_seen for *accessory* (0.0 when never observed)."""
        frames = self.frames_seen
        if frames == 0:
            return 0.0
        return sum(self.vote_lists()[accessory]) / frames

    @property
    def cap_ratio(self) -> float:
        return self.ratio_for("cap")

    @property
    def mask_ratio(self) -> float:
        return self.ratio_for("mask")

    @property
    def glasses_ratio(self) -> float:
        return self.ratio_for("glasses")

    @property
    def headphones_ratio(self) -> float:
        return self.ratio_for("headphones")

    def ratios(self) -> dict[str, float]:
        """{"mask": r, "cap": r, "glasses": r, "headphones": r}"""
        return {key: self.ratio_for(key) for key in ACCESSORY_KEYS}

    def observed_frames(self) -> int:
        """Deprecated alias for frames_seen."""
        return self.frames_seen


class StateManager:
    """
    Maintains per-track state across the lifetime of a pipeline run.

    Key additions over the previous version:
      - update_state(track_id, frame_associations, frame_index)
            Appends one evidence observation per accessory class.
      - apply_temporal_voting(track_id, config)   thresholds: voting_config.py
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
        config: VotingConfig = VOTING,
        window: int | None = None,
        latch: bool | None = None,
    ) -> dict[str, bool]:
        """
        Provisional voting over a sliding window, used to drive the live HUD
        while a video is still processing.

        A single frame can never confirm an accessory: the track needs at least
        config.min_frames_seen frames and config.min_hits positive frames, and
        the hit ratio must reach the configured threshold for that accessory.

        Thresholds live in voting_config.py.
        """
        state = self._states.get(track_id)
        if state is None:
            logger.warning("apply_temporal_voting: unknown track_id=%d", track_id)
            return {}

        window = config.live_window if window is None else window
        latch = config.live_latch if latch is None else latch

        results: dict[str, bool] = {}

        for key in ACCESSORY_KEYS:
            if latch and getattr(state, f"final_{key}", False):
                decision = True
            else:
                decision = _vote_decision(
                    state.vote_lists()[key], key, config, window
                )
            setattr(state, f"final_{key}", decision)
            results[key] = decision

        return results

    def finalize_accessories(
        self,
        config: VotingConfig = VOTING,
    ) -> dict[int, dict[str, bool]]:
        """
        Authoritative decision for every unique track, using its full frame
        history rather than a sliding window. Call once after the last frame.
        """
        results: dict[int, dict[str, bool]] = {}

        for tid, state in self._states.items():
            flags: dict[str, bool] = {}
            for key in ACCESSORY_KEYS:
                decision = _vote_decision(state.vote_lists()[key], key, config, None)
                setattr(state, f"final_{key}", decision)
                flags[key] = decision
            results[tid] = flags

            logger.debug(
                "track %d frames_seen=%d hits=%s ratios=%s -> %s",
                tid, state.frames_seen, state.hit_counts(),
                {k: round(v, 3) for k, v in state.ratios().items()},
                state.active_finals() or "none",
            )

        confirmed = sum(1 for f in results.values() if any(f.values()))
        logger.info(
            "finalize_accessories: %d tracks evaluated, %d with confirmed gear",
            len(results), confirmed,
        )
        return results

    def apply_voting_all(
        self,
        config: VotingConfig = VOTING,
    ) -> dict[int, dict[str, bool]]:
        """
        Convenience: call apply_temporal_voting for every known track.

        Returns
        -------
        dict mapping track_id -> {accessory -> bool}
        """
        return {
            tid: self.apply_temporal_voting(tid, config)
            for tid in self._states
        }

    def observation_report(self) -> dict[int, dict[str, int]]:
        """
        Per-track accessory observation tallies, keyed by persistent track ID.

        e.g. {12: {"cap": 18, "mask": 0, "glasses": 14, "headphones": 0}}
        """
        return {
            tid: state.observation_counts()
            for tid, state in sorted(self._states.items())
        }

    def format_observation_report(self, config: VotingConfig = VOTING) -> str:
        """
        Per-track voting breakdown for the dashboard diagnostics panel: hits,
        ratio, threshold, and the resulting decision.
        """
        lines = []
        for tid, state in sorted(self._states.items()):
            counts = state.hit_counts()
            ratios = state.ratios()
            lines.append(f"track {tid}:  frames_seen = {state.frames_seen}")
            for key in ACCESSORY_KEYS:
                thr = config.threshold_for(key)
                confirmed = getattr(state, f"final_{key}")
                verdict = "WEARING" if confirmed else "no"
                lines.append(
                    f"  {key + '_hits':<18} = {counts[key]:<5}"
                    f" ratio = {ratios[key]:.2f}  (>= {thr:.2f})  -> {verdict}"
                )
        return "\n".join(lines) if lines else "No tracks recorded."

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
