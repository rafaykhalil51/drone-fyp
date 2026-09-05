"""Unit checks for accessory-to-person association and observation tallies."""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from association import (
    ACCESSORY_REGIONS,
    accessory_region,
    associate_accessories_to_tracks,
    is_plausible_size,
)
from state_manager import StateManager
from voting_config import VOTING

FAILURES = []


def check(name, actual, expected):
    ok = actual == expected
    if not ok:
        FAILURES.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={actual} expected={expected}")


def person(tid, x1, y1, x2, y2):
    return {"track_id": tid, "xyxy": [x1, y1, x2, y2], "confidence": 0.9}


def acc(cls, x1, y1, x2, y2, conf=0.8):
    return {"class_name": cls, "confidence": conf, "xyxy": [x1, y1, x2, y2]}


# Person box: x 100-200 (w=100), y 100-500 (h=400)
P = person(1, 100, 100, 200, 500)

# ---- 1. Each accessory lands in its expected body region -------------------
# cap at very top of head (y ~ 100-130)
res = associate_accessories_to_tracks([P], [acc("cap", 130, 100, 170, 130)])
check("cap in head top -> track 1", list(res.keys()), [1])

# glasses at eye level (y ~ 150-170)
res = associate_accessories_to_tracks([P], [acc("glasses", 130, 150, 170, 170)])
check("glasses at eye level -> track 1", list(res.keys()), [1])

# headphones at ear level
res = associate_accessories_to_tracks([P], [acc("headphones", 125, 140, 175, 190)])
check("headphones at ear level -> track 1", list(res.keys()), [1])

# mask over nose/mouth (lower than glasses)
res = associate_accessories_to_tracks([P], [acc("mask", 130, 180, 170, 220)])
check("mask at face level -> track 1", list(res.keys()), [1])

# ---- 2. Implausible placement is rejected ---------------------------------
# cap detected at the person's feet
res = associate_accessories_to_tracks([P], [acc("cap", 130, 450, 170, 490)])
check("cap at feet rejected", list(res.keys()), [])

# glasses detected at waist level
res = associate_accessories_to_tracks([P], [acc("glasses", 130, 300, 170, 330)])
check("glasses at waist rejected", list(res.keys()), [])

# accessory outside the person box entirely
res = associate_accessories_to_tracks([P], [acc("cap", 500, 100, 540, 130)])
check("accessory far away rejected", list(res.keys()), [])

# ---- 3. Oversized accessory rejected -------------------------------------
big = acc("cap", 100, 100, 200, 400)  # height 300 of 400 = 75% -> implausible
res = associate_accessories_to_tracks([P], [big])
check("oversized accessory rejected", list(res.keys()), [])
check("size check direct", is_plausible_size([100, 100, 200, 400], [100, 100, 200, 500]), False)
check("size check normal", is_plausible_size([130, 100, 170, 130], [100, 100, 200, 500]), True)

# ---- 4. One accessory never assigned to two people -----------------------
# Two heavily overlapping people, one cap between them
pa = person(10, 100, 100, 200, 500)
pb = person(11, 140, 100, 240, 500)
one_cap = acc("cap", 150, 100, 190, 130)
res = associate_accessories_to_tracks([pa, pb], [one_cap])
total_assigned = sum(len(v) for v in res.values())
check("overlap: assigned exactly once", total_assigned, 1)
check("overlap: single owner", len(res), 1)

# ---- 5. Closest appropriate person wins ----------------------------------
# cap centred at x=170 -> closer to pb's head centre (190) than pa's (150)
left = person(20, 0, 100, 100, 500)     # head centre x = 50
right = person(21, 120, 100, 220, 500)  # head centre x = 170
cap_right = acc("cap", 160, 100, 180, 130)  # centre x = 170
res = associate_accessories_to_tracks([left, right], [cap_right])
check("closest person wins", list(res.keys()), [21])

# ---- 6. Multiple accessories on one person -------------------------------
res = associate_accessories_to_tracks(
    [P],
    [
        acc("cap", 130, 100, 170, 128),
        acc("glasses", 130, 150, 170, 168),
    ],
)
check("two accessories one person", len(res.get(1, [])), 2)

# ---- 7. Region ordering: cap above mask ----------------------------------
cap_r = accessory_region([100, 100, 200, 500], "cap")
mask_r = accessory_region([100, 100, 200, 500], "mask")
check("cap region ends above mask region", cap_r[3] < mask_r[3], True)
check("all four classes have regions", sorted(ACCESSORY_REGIONS),
      ["cap", "glasses", "headphones", "mask"])

# ---- 8. Unknown class falls back to upper region -------------------------
res = associate_accessories_to_tracks([P], [acc("scarf", 130, 150, 170, 180)], head_fraction=0.4)
check("unknown class uses fallback", list(res.keys()), [1])

# ---- 9. Observation tallies keyed by persistent track ID ----------------
sm = StateManager()
sm.update(0, [person(12, 100, 100, 200, 500)])
for f in range(20):
    obs = []
    if f < 18:
        obs.append({"class_name": "cap"})
    if f < 14:
        obs.append({"class_name": "glasses"})
    sm.update_state(12, obs, f)

counts = sm.get(12).observation_counts()
check("track 12 cap observations", counts["cap"], 18)
check("track 12 mask observations", counts["mask"], 0)
check("track 12 glasses observations", counts["glasses"], 14)
check("track 12 headphones observations", counts["headphones"], 0)

# Temporal voting, not single-frame
sm.finalize_accessories(VOTING)
s = sm.get(12)
check("cap confirmed by voting", s.final_cap, True)
check("glasses confirmed by voting", s.final_glasses, True)
check("mask not confirmed", s.final_mask, False)
check("headphones not confirmed", s.final_headphones, False)

report = sm.observation_report()
check("observation report keyed by track id", list(report.keys()), [12])

print()
print("--- observation report format ---")
print(sm.format_observation_report())

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("ALL ASSOCIATION CHECKS PASSED")
