"""Unit checks for accessory temporal voting (ratio-based, config-driven)."""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from state_manager import StateManager
from voting_config import PHOTO_VOTING, VOTING, VotingConfig

FAILURES = []


def check(name, actual, expected):
    ok = actual == expected
    if not ok:
        FAILURES.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={actual} expected={expected}")


def track(sm, tid=1, box=(100, 100, 200, 400)):
    sm.update(0, [{"track_id": tid, "xyxy": list(box), "confidence": 0.9}])


def feed(sm, tid, present_frames, total_frames, accessory="cap"):
    for f in range(total_frames):
        obs = [{"class_name": accessory}] if f < present_frames else []
        sm.update_state(tid, obs, f)


# ── 1. Single frame can never classify ───────────────────────────────────
sm = StateManager(); track(sm)
sm.update_state(1, [{"class_name": "cap"}], 0)
sm.apply_temporal_voting(1, VOTING)
check("single frame does not confirm", sm.get(1).final_cap, False)

sm.finalize_accessories(VOTING)
check("single frame rejected at finalize", sm.get(1).final_cap, False)

# ── 2. frames_seen / hits / ratio bookkeeping ────────────────────────────
sm = StateManager(); track(sm, 12)
for f in range(20):
    obs = []
    if f < 18:
        obs.append({"class_name": "cap"})
    if f < 14:
        obs.append({"class_name": "glasses"})
    sm.update_state(12, obs, f)

s = sm.get(12)
check("frames_seen", s.frames_seen, 20)
check("cap_hits", s.cap_hits, 18)
check("mask_hits", s.mask_hits, 0)
check("glasses_hits", s.glasses_hits, 14)
check("headphones_hits", s.headphones_hits, 0)
check("cap_ratio", round(s.cap_ratio, 2), 0.9)
check("glasses_ratio", round(s.glasses_ratio, 2), 0.7)
check("mask_ratio", s.mask_ratio, 0.0)

sm.finalize_accessories(VOTING)
check("cap confirmed", s.final_cap, True)
check("glasses confirmed", s.final_glasses, True)
check("mask not confirmed", s.final_mask, False)

# ── 3. Ratio threshold boundary ──────────────────────────────────────────
# cap threshold is 0.25 by default
cfg = VotingConfig(thresholds={"cap": 0.25}, min_frames_seen=3, min_hits=2)

sm = StateManager(); track(sm)
feed(sm, 1, present_frames=30, total_frames=100)   # ratio 0.30 >= 0.25
sm.finalize_accessories(cfg)
check("ratio 0.30 passes 0.25", sm.get(1).final_cap, True)

sm = StateManager(); track(sm)
feed(sm, 1, present_frames=20, total_frames=100)   # ratio 0.20 < 0.25
sm.finalize_accessories(cfg)
check("ratio 0.20 fails 0.25", sm.get(1).final_cap, False)

sm = StateManager(); track(sm)
feed(sm, 1, present_frames=25, total_frames=100)   # ratio exactly 0.25
sm.finalize_accessories(cfg)
check("ratio exactly 0.25 passes", sm.get(1).final_cap, True)

# ── 4. Threshold is configurable ─────────────────────────────────────────
strict = VotingConfig(thresholds={"cap": 0.80}, min_frames_seen=3, min_hits=2)
lenient = VotingConfig(thresholds={"cap": 0.05}, min_frames_seen=3, min_hits=2)

sm = StateManager(); track(sm)
feed(sm, 1, present_frames=30, total_frames=100)   # ratio 0.30
sm.finalize_accessories(strict)
check("ratio 0.30 fails strict 0.80", sm.get(1).final_cap, False)
sm.finalize_accessories(lenient)
check("ratio 0.30 passes lenient 0.05", sm.get(1).final_cap, True)

# ── 5. Per-accessory thresholds are independent ──────────────────────────
mixed = VotingConfig(
    thresholds={"cap": 0.60, "glasses": 0.10},
    min_frames_seen=3, min_hits=2,
)
sm = StateManager(); track(sm)
for f in range(100):
    obs = []
    if f < 30:                      # ratio 0.30 for both
        obs.append({"class_name": "cap"})
        obs.append({"class_name": "glasses"})
    sm.update_state(1, obs, f)
sm.finalize_accessories(mixed)
check("cap fails its 0.60 threshold", sm.get(1).final_cap, False)
check("glasses passes its 0.10 threshold", sm.get(1).final_glasses, True)

# ── 6. min_hits blocks sparse noise regardless of ratio ──────────────────
sm = StateManager(); track(sm)
feed(sm, 1, present_frames=1, total_frames=2)      # ratio 0.50 but 1 hit
sm.finalize_accessories(VOTING)
check("1 hit blocked by min_hits", sm.get(1).final_cap, False)

# ── 7. Short unanimous track still confirms ──────────────────────────────
sm = StateManager(); track(sm)
feed(sm, 1, present_frames=2, total_frames=2)      # 2 of 2 frames
sm.finalize_accessories(VOTING)
check("short unanimous track confirms", sm.get(1).final_cap, True)

# ── 8. Sporadic noise across a long track rejected ───────────────────────
sm = StateManager(); track(sm)
for f in range(200):
    obs = [{"class_name": "mask"}] if f % 20 == 0 else []   # ratio 0.05
    sm.update_state(1, obs, f)
sm.finalize_accessories(VOTING)
check("5% noise rejected", sm.get(1).final_mask, False)

# ── 9. Person with nothing resolves to none ──────────────────────────────
sm = StateManager(); track(sm)
feed(sm, 1, present_frames=0, total_frames=40)
sm.finalize_accessories(VOTING)
check("no accessories -> none", sm.get(1).active_finals(), [])

# ── 10. Photo config allows a single frame ───────────────────────────────
sm = StateManager(); track(sm)
sm.update_state(1, [{"class_name": "cap"}], 0)
sm.apply_temporal_voting(1, PHOTO_VOTING)
check("photo config confirms on 1 frame", sm.get(1).final_cap, True)

# ── 11. Defaults sit in the requested 0.20-0.30 band ─────────────────────
for key in ("cap", "mask", "glasses", "headphones"):
    thr = VOTING.threshold_for(key)
    check(f"{key} default threshold in 0.20-0.30", 0.20 <= thr <= 0.30, True)

print()
print("--- voting config ---")
print(VOTING.describe())
print()
print("--- per-track breakdown ---")
sm = StateManager(); track(sm, 12)
for f in range(20):
    obs = []
    if f < 18:
        obs.append({"class_name": "cap"})
    if f < 14:
        obs.append({"class_name": "glasses"})
    sm.update_state(12, obs, f)
sm.finalize_accessories(VOTING)
print(sm.format_observation_report(VOTING))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("ALL VOTING CHECKS PASSED")
