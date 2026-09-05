"""Unit checks for PersonTrackRegistry confirmation gating."""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from person_registry import PersonTrackRegistry

FAILURES = []


def check(name, actual, expected):
    ok = actual == expected
    if not ok:
        FAILURES.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={actual} expected={expected}")


# 1. A track seen once is not a person
r = PersonTrackRegistry()
r.observe(1, 0, 0.9)
check("1-frame track not confirmed", r.unique_person_count, 0)

# 2. A track seen for 8 frames is confirmed
r = PersonTrackRegistry()
for f in range(8):
    r.observe(1, f, 0.9)
check("8-frame track confirmed", r.unique_person_count, 1)

# 3. Short-lived false tracks are discarded, long ones kept
r = PersonTrackRegistry()
for f in range(30):
    r.observe(100, f, 0.9)          # real person
for tid in range(200, 240):
    r.observe(tid, 5, 0.9)          # 40 one-frame noise tracks
check("40 noise tracks discarded", r.unique_person_count, 1)
check("raw ids include noise", len(r.raw_ids), 41)
check("discarded count", len(r.discarded_ids), 40)

# 4. High track IDs do not inflate the count
r = PersonTrackRegistry()
for tid in (298, 213, 220):
    for f in range(10):
        r.observe(tid, f, 0.9)
check("high IDs -> 3 people", r.unique_person_count, 3)
check("max id not used as count", r.max_raw_id, 298)

# 5. Low-confidence tracks rejected even if long-lived
r = PersonTrackRegistry()
for f in range(50):
    r.observe(1, f, 0.20)
check("low-confidence track rejected", r.unique_person_count, 0)

# 6. Negative / None IDs ignored
r = PersonTrackRegistry()
for f in range(20):
    r.observe(-1, f, 0.9)
    r.observe(None, f, 0.9)
check("invalid ids ignored", r.unique_person_count, 0)

# 7. Debug report shape
r = PersonTrackRegistry()
for f in range(12):
    r.observe(5, f, 0.8)
r.observe(9, 3, 0.8)
rep = r.debug_report()
check("report raw", rep["raw_track_ids_created"], 2)
check("report confirmed", rep["confirmed_track_ids"], 1)
check("report unique", rep["final_unique_person_count"], 1)
check("report discarded", rep["discarded_short_tracks"], 1)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("ALL REGISTRY CHECKS PASSED")
