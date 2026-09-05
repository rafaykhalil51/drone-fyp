"""Verify the dual-model voting path: register rows + diagnostics report."""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from association import associate_accessories_to_tracks
from state_manager import StateManager
from voting_config import VOTING

# Two people. #1 wears a cap most frames + glasses sometimes.
# #2 gets one spurious cap frame only.
sm = StateManager()
P1 = [100, 100, 200, 400]
P2 = [400, 100, 500, 400]

for f in range(60):
    tracks = [
        {"track_id": 1, "xyxy": list(P1), "confidence": 0.9},
        {"track_id": 2, "xyxy": list(P2), "confidence": 0.9},
    ]
    sm.update(f, tracks)

    accs = []
    if f < 40:                                    # cap on #1: ratio 0.67
        accs.append({"class_name": "cap", "xyxy": [130, 105, 170, 130], "confidence": 0.8})
    if f < 15:                                    # glasses on #1: ratio 0.25
        accs.append({"class_name": "glasses", "xyxy": [135, 130, 165, 145], "confidence": 0.7})
    if f == 7:                                    # single spurious cap on #2
        accs.append({"class_name": "cap", "xyxy": [430, 105, 470, 130], "confidence": 0.8})

    acc_map = associate_accessories_to_tracks(tracks, accs, head_fraction=0.52)
    for t in tracks:
        t["accessories"] = acc_map.get(t["track_id"], [])
    for t in tracks:
        sm.update_state(t["track_id"], t.get("accessories", []), f)
        sm.apply_temporal_voting(t["track_id"], VOTING)

sm.finalize_accessories(VOTING)

s1, s2 = sm.get(1), sm.get(2)
fails = []


def check(name, actual, expected):
    if actual != expected:
        fails.append(name)
    print(f"[{'PASS' if actual == expected else 'FAIL'}] {name}: got={actual} expected={expected}")


check("p1 frames_seen", s1.frames_seen, 60)
check("p1 cap_hits", s1.cap_hits, 40)
check("p1 cap_ratio", round(s1.cap_ratio, 2), 0.67)
check("p1 cap confirmed", s1.final_cap, True)
check("p1 glasses_hits", s1.glasses_hits, 15)
check("p1 glasses_ratio", round(s1.glasses_ratio, 2), 0.25)
check("p1 glasses confirmed (0.25 >= 0.22)", s1.final_glasses, True)
check("p1 mask not confirmed", s1.final_mask, False)

check("p2 cap_hits (single frame)", s2.cap_hits, 1)
check("p2 cap not confirmed", s2.final_cap, False)
check("p2 wears nothing", s2.active_finals(), [])

# Register rows carry the new hits/ratio columns. Extract build_table_rows
# from dashboard.py without importing the module (it executes Streamlit calls).
import ast  # noqa: E402

_tree = ast.parse(open("dashboard.py", encoding="utf-8").read())
_fn = next(
    n for n in _tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "build_table_rows"
)
_ns: dict = {}
exec(compile(ast.Module(body=[_fn], type_ignores=[]), "<rows>", "exec"), _ns)
rows = _ns["build_table_rows"](sm)
r1 = next(r for r in rows if r["Person ID"] == "ID-001")
check("row frames seen", r1["Frames Seen"], 60)
check("row cap hits", r1["Cap Hits"], 40)
check("row cap ratio", r1["Cap Ratio"], 0.67)
_gear = r1["Accessories"]
check("row gear lists Cap", "Cap" in _gear, True)
check("row gear lists Glasses", "Glasses" in _gear, True)
check("row gear omits Mask", "Mask" in _gear, False)
check("row 2 gear is None", next(r for r in rows if r["Person ID"] == "ID-002")["Accessories"], "None")

print()
print(sm.format_observation_report(VOTING))
print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL DUAL-PATH VOTING CHECKS PASSED")
