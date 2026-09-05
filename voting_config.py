"""
voting_config.py
----------------
Single configuration section for accessory temporal voting.

TUNE EVERYTHING HERE. These are the only knobs that decide whether a tracked
person is classified as wearing a cap, mask, glasses, or headphones.

How the decision works, per confirmed person track and per accessory:

    frames_seen      = frames in which the track was observed
    <acc>_hits       = frames in which <acc> was associated to the track
    <acc>_ratio      = <acc>_hits / frames_seen

A person is classified as wearing <acc> only when ALL of the following hold:

    frames_seen  >= min_frames_seen        (enough evidence to judge at all)
    <acc>_hits   >= min_hits               (never a single-frame decision)
    <acc>_ratio  >= threshold for <acc>    (sustained across the track)

Thresholds sit near 0.20-0.30 rather than 0.5 because accessories are often
hidden: people turn away from the camera, other people occlude them, and small
items like glasses are only resolvable when the face is reasonably frontal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ACCESSORY_KEYS = ("mask", "cap", "glasses", "headphones")


@dataclass(frozen=True)
class VotingConfig:
    """Thresholds controlling accessory classification."""

    # ── Per-accessory ratio thresholds (hits / frames_seen) ───────────────
    # Raise a value to require stronger, more sustained evidence.
    # Lower it if an accessory is being missed on people who clearly wear one.
    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "cap": 0.25,         # usually visible whenever the head is visible
            "mask": 0.25,        # hidden when the person faces away
            "glasses": 0.22,     # small; needs a fairly frontal face
            "headphones": 0.25,  # visible from front and side
        }
    )

    # ── Minimum evidence gates ────────────────────────────────────────────
    # A track must be observed this many frames before any classification.
    min_frames_seen: int = 3
    # An accessory needs this many positive frames. >= 2 makes a single-frame
    # detection incapable of classifying a person.
    min_hits: int = 2

    # ── Live (in-progress) voting ─────────────────────────────────────────
    # During processing the HUD votes over a sliding window of recent frames so
    # counts respond as the video plays. Final counts always use full history.
    live_window: int = 45
    # Keep a confirmed accessory confirmed through later occlusion.
    live_latch: bool = True

    def threshold_for(self, accessory: str) -> float:
        """Ratio threshold for *accessory*, falling back to the cap value."""
        return self.thresholds.get(accessory, self.thresholds.get("cap", 0.25))

    def describe(self) -> str:
        """Readable summary for the dashboard diagnostics panel."""
        lines = [
            f"min_frames_seen = {self.min_frames_seen}",
            f"min_hits        = {self.min_hits}",
            f"live_window     = {self.live_window} frames",
            "",
            "ratio thresholds (hits / frames_seen):",
        ]
        for key in ACCESSORY_KEYS:
            lines.append(f"  {key:<11} >= {self.threshold_for(key):.2f}")
        return "\n".join(lines)


# ── ACTIVE CONFIGURATION ─────────────────────────────────────────────────
# Video / multi-frame voting. Edit the values above, or override here.
VOTING = VotingConfig()

# A still photo has exactly one frame of evidence, so the multi-frame gates
# cannot apply. Ratios still must pass, but a single hit is allowed.
PHOTO_VOTING = VotingConfig(
    min_frames_seen=1,
    min_hits=1,
    live_window=1,
)
