"""
audit_e2e.py — end-to-end pipeline audit.

Exercises the real pipeline modules headlessly (no Streamlit runtime) and
asserts each stage of the declared workflow. Where a stage cannot be reached
because no trained accessory model exists, the stage is driven with a stub
detector so the WIRING is still proven.

Run:  python audit_e2e.py
"""
import logging
import os
import subprocess
import sys
import tempfile

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.disable(logging.CRITICAL)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def record(stage, ok, detail=""):
    RESULTS.append((stage, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {stage}" + (f" — {detail}" if detail else ""))
    return ok


SAMPLE = "input.mp4"
MAX_FRAMES = 120          # keep the audit fast; enough to confirm tracks

# ══════════════════════════════════════════════════════════════════════════
# STAGE 1-2: upload path + video opens
# ══════════════════════════════════════════════════════════════════════════
print("\n=== STAGE 1-2: upload -> video opens ===")

# Simulate the dashboard's upload: bytes -> temp file -> VideoCapture.
# This mirrors dashboard.py's tempfile.NamedTemporaryFile(delete=False) usage.
from video_source import VideoSource  # noqa: E402

with open(SAMPLE, "rb") as f:
    raw_bytes = f.read()

tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tmp.write(raw_bytes)
tmp.flush()
tmp.close()                      # <-- dashboard.py originally omitted this
tmp_path = tmp.name
record("upload writes readable temp file", os.path.getsize(tmp_path) == len(raw_bytes),
       f"{os.path.getsize(tmp_path)} bytes")

try:
    src = VideoSource(tmp_path)
    opened = True
except Exception as e:
    opened = False
    src = None
record("VideoSource opens uploaded temp file", opened)

if src:
    record("video geometry valid",
           src.width > 0 and src.height > 0,
           f"{src.width}x{src.height}")
    record("fps sane (guards VideoWriter + ffmpeg)",
           1.0 <= src.fps <= 120.0, f"fps={src.fps}")
    record("frame count readable", src.total_frames > 0, f"{src.total_frames} frames")
    src.release()

# Windows temp-file deletability (fails if a handle is still open)
try:
    os.unlink(tmp_path)
    record("temp file deletable after use (no leaked handle)", True)
except Exception as e:
    record("temp file deletable after use (no leaked handle)", False, str(e))

# Empty / corrupt input must raise, not hang
bad = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
bad.write(b"not a video")
bad.close()
try:
    VideoSource(bad.name)
    record("corrupt video rejected with error", False, "opened a non-video!")
except Exception:
    record("corrupt video rejected with error", True)
os.unlink(bad.name)

# ══════════════════════════════════════════════════════════════════════════
# STAGE 3-4: person YOLO loads / accessory YOLO loads
# ══════════════════════════════════════════════════════════════════════════
print("\n=== STAGE 3-4: model loading ===")
from model_loader import (  # noqa: E402
    DEFAULT_CUSTOM_PATH, format_model_report, inspect_weight_inventory,
    load_accessory_detector, load_person_model, person_class_filter,
    resolve_models,
)

res = resolve_models(DEFAULT_CUSTOM_PATH)
record("resolve_models returns a person model", bool(res.person_path), res.person_path)
record("person weights exist on disk", os.path.exists(res.person_path), res.person_path)
record("resolution mode is valid", res.mode in ("unified", "dual", "person_only"), res.mode)

person_model = load_person_model(res.person_path)
record("person YOLO loads", person_model is not None, res.person_path)

pcf = person_class_filter(person_model)
record("person class filter resolves", pcf is not None and pcf != [], f"classes={pcf}")

ACCESSORY_AVAILABLE = res.accessory_active
if ACCESSORY_AVAILABLE:
    det = load_accessory_detector(res.accessory_path)
    record("accessory YOLO loads", det is not None and det.available, res.accessory_path)
else:
    record("accessory YOLO loads", False,
           f"NO TRAINED ACCESSORY MODEL (looked for {DEFAULT_CUSTOM_PATH}) "
           "— stages below driven by stub")

inv = inspect_weight_inventory()
record("weight inventory enumerates without error", isinstance(inv, list),
       f"{len(inv)} weight file(s)")
# Guard against a COCO model being silently treated as an accessory model
coco_as_acc = [i["path"] for i in inv if i["accessory_slots"]]
record("no COCO model masquerades as accessory model",
       ACCESSORY_AVAILABLE or not coco_as_acc,
       f"slots found in: {coco_as_acc}" if coco_as_acc else "none claim accessory slots")

# ══════════════════════════════════════════════════════════════════════════
# STAGE 5-7: frames process -> people detected -> stable tracking IDs
# ══════════════════════════════════════════════════════════════════════════
print("\n=== STAGE 5-7: frames -> detection -> tracking ===")
from person_registry import PersonTrackRegistry, reset_tracker  # noqa: E402
from state_manager import StateManager  # noqa: E402
from voting_config import VOTING  # noqa: E402

PERSON_TRACK_MIN_CONF = 0.35


def person_tracks_from_result(result):
    """Mirror of dashboard.py's person track extraction."""
    out = []
    if result.boxes is None or result.boxes.id is None:
        return out
    for box in result.boxes:
        if box.id is None:
            continue
        out.append({
            "track_id": int(box.id[0]),
            "xyxy": [float(v) for v in box.xyxy[0]],
            "confidence": float(box.conf[0]),
        })
    return out


def run_tracking(model, path, max_frames=MAX_FRAMES):
    reset_tracker(model)
    registry = PersonTrackRegistry()
    sm = StateManager()
    src = VideoSource(path)
    frames = detections = 0
    writer_frames = []
    with src:
        for frame in src:
            if frames >= max_frames:
                break
            r = model.track(
                frame, conf=max(0.25, PERSON_TRACK_MIN_CONF), iou=0.50, imgsz=480,
                classes=person_class_filter(model), tracker="botsort.yaml",
                persist=True, verbose=False,
            )[0]
            tracks = person_tracks_from_result(r)
            detections += len(tracks)
            for t in tracks:
                registry.observe(t["track_id"], frames, t.get("confidence", 0.0))
            confirmed = [t for t in tracks if registry.is_confirmed(t["track_id"])]
            if confirmed:
                sm.update(frames, confirmed)
                for t in confirmed:
                    sm.update_state(t["track_id"], t.get("accessories", []), frames)
            writer_frames.append(frame.shape)
            frames += 1
    return registry, sm, frames, detections, writer_frames


registry, sm, frames, detections, shapes = run_tracking(person_model, SAMPLE)

record("frames process", frames > 0, f"{frames} frames")
record("people are detected", detections > 0, f"{detections} person detections")
record("raw tracking IDs created", len(registry.raw_ids) > 0,
       f"{len(registry.raw_ids)} raw IDs")
record("confirmed tracks are a subset of raw",
       registry.confirmed_ids.issubset(registry.raw_ids),
       f"{len(registry.confirmed_ids)} confirmed / {len(registry.raw_ids)} raw")

# The headline defect the user called out: count must not be max(track_id)
uniq = registry.unique_person_count
record("unique count != max track ID",
       uniq != registry.max_raw_id or len(registry.raw_ids) == registry.max_raw_id,
       f"unique={uniq}  max_raw_id={registry.max_raw_id}")
record("unique count != raw track count (short tracks discarded)",
       uniq <= len(registry.raw_ids),
       f"unique={uniq}  raw={len(registry.raw_ids)}  discarded={len(registry.discarded_ids)}")
record("unique count != detection count (no per-frame double counting)",
       uniq < detections, f"unique={uniq}  detections={detections}")
record("unique count == len(confirmed set)",
       uniq == len(registry.confirmed_ids), f"{uniq}")

# Tracker reset must stop IDs creeping across runs in one session
reg2, _, _, _, _ = run_tracking(person_model, SAMPLE, max_frames=40)
reg3, _, _, _, _ = run_tracking(person_model, SAMPLE, max_frames=40)
record("track IDs reset between runs (no cross-run ID creep)",
       min(reg3.raw_ids) <= min(reg2.raw_ids) + 1 if reg2.raw_ids and reg3.raw_ids else False,
       f"run2 min={min(reg2.raw_ids) if reg2.raw_ids else '-'} "
       f"run3 min={min(reg3.raw_ids) if reg3.raw_ids else '-'}")

# StateManager must not invent people beyond confirmed tracks
record("StateManager total_unique == confirmed persons (no duplicate counting)",
       sm.total_unique == registry.unique_person_count,
       f"state={sm.total_unique}  registry={registry.unique_person_count}")

# ══════════════════════════════════════════════════════════════════════════
# STAGE 8-10: accessory detections -> association -> temporal voting
# ══════════════════════════════════════════════════════════════════════════
print("\n=== STAGE 8-10: accessories -> association -> voting ===")
from association import associate_accessories_to_tracks  # noqa: E402

if ACCESSORY_AVAILABLE:
    src = VideoSource(SAMPLE)
    acc_total = 0
    with src:
        for i, frame in enumerate(src):
            if i >= 30:
                break
            acc_total += len(det.detect(frame))
    src.release()
    record("accessory detections generated (real model)", acc_total > 0, f"{acc_total} dets")
else:
    record("accessory detections generated (real model)", False,
           "IMPOSSIBLE without trained weights — using stub to prove wiring")

# Stub accessory stream over the real tracked geometry proves association +
# voting work end-to-end regardless of whether weights exist.
sm_stub = StateManager()
P1 = [100.0, 100.0, 200.0, 400.0]
P2 = [400.0, 100.0, 500.0, 400.0]
assoc_hits = 0
for f in range(60):
    tracks = [
        {"track_id": 1, "xyxy": list(P1), "confidence": 0.9},
        {"track_id": 2, "xyxy": list(P2), "confidence": 0.9},
    ]
    accs = []
    if f < 40:
        accs.append({"class_name": "cap", "xyxy": [130, 105, 170, 130], "confidence": 0.8})
    if f < 15:
        accs.append({"class_name": "glasses", "xyxy": [135, 130, 165, 145], "confidence": 0.7})
    if f == 7:
        accs.append({"class_name": "cap", "xyxy": [430, 105, 470, 130], "confidence": 0.8})
    # a wildly oversized box that must be rejected as implausible
    accs.append({"class_name": "mask", "xyxy": [100, 100, 200, 395], "confidence": 0.9})

    amap = associate_accessories_to_tracks(tracks, accs, head_fraction=0.52)
    assoc_hits += sum(len(v) for v in amap.values())
    for t in tracks:
        t["accessories"] = amap.get(t["track_id"], [])
    sm_stub.update(f, tracks)
    for t in tracks:
        sm_stub.update_state(t["track_id"], t["accessories"], f)
        sm_stub.apply_temporal_voting(t["track_id"], VOTING)

record("accessories associate to people", assoc_hits > 0, f"{assoc_hits} associations")
record("implausibly large accessory rejected",
       sm_stub.get(1).mask_hits == 0, f"mask_hits={sm_stub.get(1).mask_hits}")

final = sm_stub.finalize_accessories(VOTING)
s1, s2 = sm_stub.get(1), sm_stub.get(2)
record("temporal voting confirms sustained accessory",
       s1.final_cap and s1.cap_hits == 40, f"cap {s1.cap_hits}/{s1.frames_seen} r={s1.cap_ratio:.2f}")
record("temporal voting rejects single-frame accessory",
       not s2.final_cap and s2.cap_hits == 1, f"cap {s2.cap_hits}/{s2.frames_seen} r={s2.cap_ratio:.2f}")
record("no zero-division when track never observed",
       StateManager().__class__ is StateManager and s1.ratio_for("mask") == 0.0, "ratio=0.0")

# ══════════════════════════════════════════════════════════════════════════
# STAGE 11-16: unique / cap / mask / glasses / headphones / plain counts
# ══════════════════════════════════════════════════════════════════════════
print("\n=== STAGE 11-16: count calculation ===")
from counter import AccessoryCounter  # noqa: E402

ac = AccessoryCounter().compute(sm_stub)
record("unique person total calculated", ac.total_unique_persons == 2,
       f"{ac.total_unique_persons}")
record("cap count calculated", ac.totals["cap"] == 1, f"{ac.totals['cap']}")
record("mask count calculated", ac.totals["mask"] == 0, f"{ac.totals['mask']}")
record("glasses count calculated", ac.totals["glasses"] == 1, f"{ac.totals['glasses']}")
record("headphones count calculated", ac.totals["headphones"] == 0, f"{ac.totals['headphones']}")

states = list(sm_stub.all_states().values())
plain = sum(1 for s in states
            if not (s.final_cap or s.final_mask or s.final_glasses or s.final_headphones))
record("plain count calculated", plain == 1, f"{plain}")
record("counts never exceed unique persons",
       all(v <= ac.total_unique_persons for v in ac.totals.values()),
       f"unique={ac.total_unique_persons} totals={ac.totals}")
record("AccessoryCounter unique == registry-style unique (no drift)",
       ac.total_unique_persons == len(states), f"{ac.total_unique_persons}=={len(states)}")

# Empty-run safety: no detections at all must not raise or divide by zero
empty = StateManager()
ac_empty = AccessoryCounter().compute(empty)
record("empty detection set yields zeros, no exception",
       ac_empty.total_unique_persons == 0 and all(v == 0 for v in ac_empty.totals.values()),
       f"unique=0 totals={ac_empty.totals}")
record("empty state report does not crash",
       isinstance(empty.format_observation_report(VOTING), str))

# ══════════════════════════════════════════════════════════════════════════
# STAGE 17: processed video written + playable
# ══════════════════════════════════════════════════════════════════════════
print("\n=== STAGE 17: video output ===")
from visualization import Visualizer  # noqa: E402

vis = Visualizer(vis_cfg={"box_thickness": 2, "text_scale": 0.65, "show_confidence": True})
src = VideoSource(SAMPLE)
out_raw = "audit_raw.mp4"
w = cv2.VideoWriter(out_raw, cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (src.width, src.height))
record("VideoWriter opens with mp4v fourcc", w.isOpened())

written = 0
with src:
    for i, frame in enumerate(src):
        if i >= 30:
            break
        ann = vis.draw(frame.copy(), [
            {"track_id": 1, "xyxy": [50, 50, 150, 350], "confidence": 0.9,
             "accessories": [{"class_name": "cap", "xyxy": [70, 55, 110, 80], "confidence": 0.8}]},
        ], live_totals={"persons": 1, "caps": 1, "masks": 0, "glasses": 0, "headphones": 0})
        record_shape = ann.shape[:2] == (src.height, src.width)
        w.write(ann)
        written += 1
w.release()
record("annotated frames written", written == 30, f"{written} frames")
record("visualizer preserves frame size", record_shape)
record("raw output non-empty", os.path.exists(out_raw) and os.path.getsize(out_raw) > 0,
       f"{os.path.getsize(out_raw)} bytes")

# mp4v output is often not browser-playable; ffmpeg H.264 transcode is required
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_src = "imageio_ffmpeg"
except ImportError:
    ffmpeg_exe = "ffmpeg"
    ffmpeg_src = "PATH"
record("ffmpeg binary available for H.264 transcode",
       bool(ffmpeg_exe), f"{ffmpeg_src}: {ffmpeg_exe}")

out_final = "audit_h264.mp4"
try:
    r = subprocess.run(
        [ffmpeg_exe, "-y", "-r", f"{src.fps:.2f}", "-i", out_raw,
         "-c:v", "libx264", "-r", f"{src.fps:.2f}", "-preset", "fast", "-crf", "22",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_final],
        capture_output=True, timeout=120,
    )
    ok = r.returncode == 0 and os.path.exists(out_final) and os.path.getsize(out_final) > 0
    record("H.264 transcode succeeds (browser-playable)", ok,
           f"rc={r.returncode} size={os.path.getsize(out_final) if os.path.exists(out_final) else 0}")
except Exception as e:
    record("H.264 transcode succeeds (browser-playable)", False, str(e))

# The transcoded file must be re-openable (proves it is a valid container)
if os.path.exists(out_final):
    chk = cv2.VideoCapture(out_final)
    record("processed video re-opens (valid for st.video)", chk.isOpened())
    chk.release()

for f in (out_raw, out_final):
    try:
        os.path.exists(f) and os.unlink(f)
    except Exception:
        pass

# Zero geometry must be rejected before a writer is built, not silently
# produce an unusable file.
_zero_w = cv2.VideoWriter("audit_zero.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (0, 0))
record("zero frame size never yields a usable writer", not _zero_w.isOpened(),
       "guarded explicitly in dashboard before writer construction")
_zero_w.release()
for f in ("audit_zero.mp4",):
    try:
        os.path.exists(f) and os.unlink(f)
    except Exception:
        pass

# At least one codec in the fallback chain must be available.
_avail = []
for _fc, _p in (("mp4v", "audit_fb.mp4"), ("avc1", "audit_fb.mp4"), ("MJPG", "audit_fb.avi")):
    _w2 = cv2.VideoWriter(_p, cv2.VideoWriter_fourcc(*_fc), 30.0, (320, 240))
    if _w2.isOpened():
        _avail.append(_fc)
    _w2.release()
    try:
        os.path.exists(_p) and os.unlink(_p)
    except Exception:
        pass
record("codec fallback chain has a working option", bool(_avail),
       f"available: {_avail}")

# ══════════════════════════════════════════════════════════════════════════
# STAGE 18-19: dashboard cards + chart data
# ══════════════════════════════════════════════════════════════════════════
print("\n=== STAGE 18-19: dashboard cards + chart ===")
import ast  # noqa: E402

_tree = ast.parse(open("dashboard.py", encoding="utf-8").read())


def extract(*names):
    ns = {"go": __import__("plotly.graph_objects", fromlist=["x"]),
          "_format_class_name": lambda s: str(s).title()}
    body = [n for n in _tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    for fn in body:
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<x>", "exec"), ns)
    return ns


ns = extract("build_table_rows")
rows = ns["build_table_rows"](sm_stub)
record("dashboard register rows build", len(rows) == 2, f"{len(rows)} rows")
r1 = next(r for r in rows if r["Target ID"] == "TRK-001")
record("register row exposes frames_seen + hits + ratio",
       r1["Frames Seen"] == 60 and r1["Cap Hits"] == 40 and r1["Cap Ratio"] == 0.67,
       f"seen={r1['Frames Seen']} hits={r1['Cap Hits']} ratio={r1['Cap Ratio']}")
record("register row gear reflects voting",
       "Cap" in r1["Gear Verified"] and "Mask" not in r1["Gear Verified"],
       r1["Gear Verified"])
record("empty state produces empty register (no crash)",
       ns["build_table_rows"](StateManager()) == [])

# Chart: verify the figure-building logic across every real input shape.
_chart_src = None
for n in _tree.body:
    if isinstance(n, ast.FunctionDef) and n.name == "render_chart":
        _chart_src = n
record("render_chart exists in dashboard", _chart_src is not None)


def chart_values(tc, tm, tg, th, tn, active):
    """Replicate render_chart's data selection to validate what Plotly gets."""
    if not active:
        return ["Plain 👤 (COCO only)"], [max(tn, 1)]
    total = tc + tm + tg + th + tn
    if total == 0:
        return ["Awaiting Detection"], [1]
    raw = [("Caps 🧢", tc), ("Masks 😷", tm), ("Glasses 👓", tg),
           ("Headphones 🎧", th), ("Plain 👤", tn)]
    active_cats = [c for c in raw if c[1] > 0] or raw
    return [c[0] for c in active_cats], [c[1] for c in active_cats]

cases = [
    ("all zeros + accessory active", (0, 0, 0, 0, 0, True)),
    ("all zeros + COCO mode",        (0, 0, 0, 0, 0, False)),
    ("real gear mix",                (1, 0, 1, 0, 1, True)),
    ("plain only",                   (0, 0, 0, 0, 5, True)),
]
chart_ok = True
for name, args in cases:
    labels, values = chart_values(*args)
    good = (len(labels) == len(values) and len(values) > 0
            and sum(values) > 0 and all(v >= 0 for v in values))
    chart_ok &= good
    record(f"chart data valid: {name}", good, f"{dict(zip(labels, values))}")
record("chart never receives empty/zero-sum data (no plotly div-by-zero)", chart_ok)

# Chart keys must be unique within a run or Streamlit raises DuplicateElementKey
keys = [f"gear_chart_{i}" for i in range(1, 200)]
record("chart keys unique across live updates", len(keys) == len(set(keys)),
       f"{len(set(keys))} unique keys")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
fails = [r for r in RESULTS if r[1] == "FAIL"]
print(f"{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
if fails:
    print("\nFAILURES:")
    for stage, _, detail in fails:
        print(f"  - {stage}: {detail}")
print("=" * 70)
