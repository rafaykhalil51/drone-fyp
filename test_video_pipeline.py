"""Headless smoke test for the video detection pipeline (no Streamlit import)."""
import os
import subprocess
import sys

import cv2

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from ultralytics import YOLO

from association import associate_accessories_to_tracks
from video_source import VideoSource
from visualization import Visualizer

# Mirror dashboard.resolve_model_path fallback logic
def resolve_model_path():
    for candidate in ("best.pt", "yolo11n.pt", "yolov8n.pt"):
        if os.path.exists(candidate):
            return candidate
    return "best.pt"

MODEL = resolve_model_path()
VIDEO = "input.mp4"
MAX_FRAMES = None  # set to int for quick smoke test

if not os.path.exists(VIDEO):
    print(f"ERROR: sample video missing: {VIDEO}")
    sys.exit(1)
if not os.path.exists(MODEL):
    print(f"ERROR: model weights missing: {MODEL}")
    sys.exit(1)

print(f"Model: {MODEL}")
print(f"Video: {VIDEO}")

yolo = YOLO(MODEL)
source = VideoSource(VIDEO)
out_raw = "test_output_raw.mp4"
out_final = "test_output_annotated.mp4"
target_fps = source.fps

writer = cv2.VideoWriter(
    out_raw,
    cv2.VideoWriter_fourcc(*"mp4v"),
    target_fps,
    (source.width, source.height),
)
if not writer.isOpened():
    print("ERROR: VideoWriter failed to open")
    sys.exit(1)

visualizer = Visualizer({"show_confidence": True})
unique_tracks_by_class = {}
unique_person_ids = set()
frame_idx = 0

with source:
    for frame in source:
        results = yolo.track(
            frame,
            conf=0.25,
            iou=0.50,
            imgsz=480,
            tracker="botsort.yaml",
            persist=True,
            verbose=False,
        )
        result = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                track_id = int(box.id[0])
                unique_tracks_by_class.setdefault(cls_name, set()).add(track_id)
                if cls_name.lower() == "person":
                    unique_person_ids.add(track_id)

        annotated = result.plot()
        visualizer._draw_hud(
            annotated,
            {
                "persons": len(unique_person_ids),
                "caps": 0,
                "masks": 0,
                "glasses": 0,
                "headphones": 0,
            },
            None,
        )
        writer.write(annotated)

        frame_idx += 1
        if MAX_FRAMES and frame_idx >= MAX_FRAMES:
            break

writer.release()
print(f"Processed {frame_idx} frames")
print(f"Unique persons: {len(unique_person_ids)}")
print(f"Unique by class: { {k: len(v) for k, v in unique_tracks_by_class.items()} }")
print(f"Raw output: {os.path.exists(out_raw)}, size={os.path.getsize(out_raw) if os.path.exists(out_raw) else 0}")

play_path = out_raw
try:
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    res = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-r",
            f"{target_fps:.2f}",
            "-i",
            out_raw,
            "-c:v",
            "libx264",
            "-r",
            f"{target_fps:.2f}",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_final,
        ],
        capture_output=True,
        timeout=120,
    )
    if res.returncode == 0 and os.path.exists(out_final) and os.path.getsize(out_final) > 0:
        play_path = out_final
        print(f"H264 output: size={os.path.getsize(out_final)}")
    else:
        print(f"ffmpeg warning: {res.stderr.decode(errors='replace')[:500]}")
except Exception as exc:
    print(f"ffmpeg skipped: {exc}")

print(f"Play path: {play_path}")
print("OK")
