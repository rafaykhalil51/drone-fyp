"""
person_registry.py
------------------
Turns raw BoT-SORT track IDs into an estimate of unique real people.

A raw track ID is not a person: BoT-SORT issues new IDs for momentary false
positives and re-issues IDs when an identity is lost and re-acquired. Counting
raw IDs (or worse, taking max(track_id)) badly overcounts.

A track is only counted once it has been seen in at least MIN_TRACK_FRAMES
frames, so short-lived spurious tracks are discarded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# A track must appear in at least this many frames to count as a real person.
MIN_TRACK_FRAMES = 8

# Tracks seen fewer than this many frames are reported as discarded noise.
MIN_CONFIDENCE = 0.35


@dataclass
class RawTrack:
    track_id: int
    first_frame: int
    last_frame: int
    frames_seen: int = 1
    best_confidence: float = 0.0


@dataclass
class PersonTrackRegistry:
    """
    Accumulates raw person tracks and promotes them to confirmed identities
    only after they survive several frames.
    """

    min_track_frames: int = MIN_TRACK_FRAMES
    min_confidence: float = MIN_CONFIDENCE
    _raw: dict[int, RawTrack] = field(default_factory=dict)
    _confirmed: set[int] = field(default_factory=set)

    def observe(self, track_id: int, frame_idx: int, confidence: float = 0.0) -> bool:
        """
        Record one sighting of *track_id*. Returns True once the track has
        enough evidence to be treated as a real person.
        """
        if track_id is None or track_id < 0:
            return False

        entry = self._raw.get(track_id)
        if entry is None:
            self._raw[track_id] = RawTrack(
                track_id=track_id,
                first_frame=frame_idx,
                last_frame=frame_idx,
                best_confidence=confidence,
            )
        else:
            entry.last_frame = frame_idx
            entry.frames_seen += 1
            entry.best_confidence = max(entry.best_confidence, confidence)

        entry = self._raw[track_id]
        if (
            entry.frames_seen >= self.min_track_frames
            and entry.best_confidence >= self.min_confidence
        ):
            self._confirmed.add(track_id)
            return True

        return track_id in self._confirmed

    def is_confirmed(self, track_id: int) -> bool:
        return track_id in self._confirmed

    @property
    def raw_ids(self) -> set[int]:
        """Every track ID the tracker created, including discarded noise."""
        return set(self._raw.keys())

    @property
    def confirmed_ids(self) -> set[int]:
        """Track IDs that survived long enough to count as real people."""
        return set(self._confirmed)

    @property
    def unique_person_count(self) -> int:
        """Estimated number of unique real people in the video."""
        return len(self._confirmed)

    @property
    def discarded_ids(self) -> set[int]:
        return self.raw_ids - self._confirmed

    @property
    def max_raw_id(self) -> int:
        return max(self._raw.keys()) if self._raw else 0

    def debug_report(self) -> dict[str, object]:
        """Diagnostics explaining how raw IDs collapsed into person count."""
        discarded = self.discarded_ids
        return {
            "raw_track_ids_created": len(self._raw),
            "confirmed_track_ids": len(self._confirmed),
            "final_unique_person_count": self.unique_person_count,
            "discarded_short_tracks": len(discarded),
            "max_raw_track_id": self.max_raw_id,
            "min_track_frames": self.min_track_frames,
            "min_confidence": self.min_confidence,
            "confirmed_id_list": sorted(self._confirmed),
            "discarded_id_list": sorted(discarded),
        }

    def log_debug_report(self) -> None:
        report = self.debug_report()
        logger.info(
            "PersonTrackRegistry: raw=%d confirmed=%d unique=%d discarded=%d max_raw_id=%d",
            report["raw_track_ids_created"],
            report["confirmed_track_ids"],
            report["final_unique_person_count"],
            report["discarded_short_tracks"],
            report["max_raw_track_id"],
        )


def reset_tracker(model) -> None:
    """
    Clear BoT-SORT state so a new video starts from track ID 1.

    persist=True deliberately keeps tracker state between frames, but a cached
    model reuses that state across separate videos too, so IDs keep climbing
    (1, 2, 3 ... 298) and every run looks like it found new people.
    """
    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        for tracker in getattr(predictor, "trackers", []) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
        if hasattr(predictor, "trackers"):
            del predictor.trackers
        if hasattr(predictor, "vid_path"):
            predictor.vid_path = None

    try:
        from ultralytics.trackers.basetrack import BaseTrack

        BaseTrack.reset_id()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not reset global track ID counter: %s", exc)

    logger.info("Tracker state reset; track IDs restart at 1.")
