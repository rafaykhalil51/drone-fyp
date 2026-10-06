"""
app.py
------
Localhost Web UI and REST API server for Student & Accessory Vision Analytics.
Hosts on http://127.0.0.1:5000

Images are analysed synchronously. Videos run as a background job: the browser
polls /api/jobs/<id> for progress and live totals, and /api/jobs/<id>/frame
for the most recent annotated frame, so the detection feed updates while the
video is being processed.
"""

import os
import csv
import math
import time
import uuid
import json
import logging
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask, Response, render_template, request, jsonify, send_file, send_from_directory,
)
from werkzeug.utils import secure_filename

from detector import PersonDetector
from accessory_detector import AccessoryDetector
from mock_accessory_detector import MockAccessoryDetector
from association import associate_accessories_to_tracks
from visualization import Visualizer
from state_manager import StateManager, ACCESSORY_KEYS
from exporter import Exporter
from tracker import PersonTracker
from person_registry import PersonTrackRegistry
from video_source import VideoSource
from voting_config import VOTING, PHOTO_VOTING
from model_loader import load_zero_shot_detector
from app_config import (
    accessory_confidence, accessory_imgsz, accessory_model_path,
    person_confidence, person_model_path, use_mock_accessories,
    zero_shot_enabled, zero_shot_model_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("web_app")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

UPLOAD_DIR = Path("uploads")
STATIC_DIR = Path("static")
JOBS_DIR = STATIC_DIR / "jobs"
for _d in (UPLOAD_DIR, STATIC_DIR, JOBS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}

# ── video processing budget (CPU friendly) ──────────────────────────────────
TARGET_PROCESS_FPS = 10.0      # source frames are sampled down to roughly this rate
MAX_PROCESSED_FRAMES = 1500    # hard cap on sampled frames per video
MAX_FRAME_WIDTH = 1280         # larger frames are downscaled before inference
TRACK_IMGSZ = 640              # person tracker input size
ACCESSORY_EVERY = 2            # run accessory detection every N sampled frames
CROP_IMGSZ = 320               # accessory inference size for each head crop
MAX_CROPS_PER_FRAME = 8        # largest persons get accessory crops first
MIN_CROP_PERSON_H = 40         # persons shorter than this (px) are too small to judge

# ── models ──────────────────────────────────────────────────────────────────
USE_MOCK_ACCESSORIES = use_mock_accessories()

logger.info("Preloading PersonDetector and AccessoryDetector...")
person_detector = PersonDetector(
    person_model_path(), confidence=person_confidence(), iou_threshold=0.5
)
acc_detector = AccessoryDetector(
    model_path=accessory_model_path(),
    confidence=accessory_confidence(),
    imgsz=accessory_imgsz(),
)

# Only instantiate the mock detector when it is explicitly enabled, so real
# mode cannot fabricate cap/mask/glasses/headphones results.
mock_acc_detector = (
    MockAccessoryDetector(seed=42, max_per_person=2) if USE_MOCK_ACCESSORIES else None
)
if USE_MOCK_ACCESSORIES:
    logger.warning("MOCK accessory detection is ON — results are synthetic.")
else:
    logger.info("Mock accessory detection: OFF (real mode).")

_zero_shot_detector = None
if not acc_detector.available and zero_shot_enabled() and Path(zero_shot_model_path()).exists():
    _zero_shot_detector = load_zero_shot_detector()
    if _zero_shot_detector is not None:
        logger.info("Zero-shot accessory detector ready (YOLO-World, experimental).")

# Ultralytics models are not safe to run from several threads at once.
MODEL_LOCK = threading.Lock()


def _accessory_engine(mode: str):
    """(detector, is_mock) for a request mode."""
    if mode == "mock" and mock_acc_detector is not None:
        return mock_acc_detector, True
    if acc_detector.available:
        return acc_detector, False
    if _zero_shot_detector is not None:
        return _zero_shot_detector, False
    return None, False


def _engine_label(detector, is_mock: bool) -> str:
    if is_mock:
        return "Mock simulator"
    if detector is None:
        return "Accessory AI offline"
    if getattr(detector, "is_trained", False):
        return "Trained accessory model"
    return "YOLO-World zero-shot"


def _engine_source(detector, is_mock: bool) -> str:
    if is_mock:
        return "mock"
    if detector is None:
        return "none (accessory AI offline)"
    if getattr(detector, "is_trained", False):
        return f"trained:{detector.model_path}"
    return "zero-shot:yolo-world (EXPERIMENTAL, untrained)"


# ── accessory detection on head crops ───────────────────────────────────────
def _head_crop_box(xyxy, frame_w: int, frame_h: int):
    """Padded head-and-shoulders crop for one person box, or None if too small."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    if w <= 4 or h < MIN_CROP_PERSON_H:
        return None
    cx1 = max(0, int(x1 - 0.15 * w))
    cx2 = min(frame_w, int(x2 + 0.15 * w))
    cy1 = max(0, int(y1 - 0.10 * h))
    cy2 = min(frame_h, int(y1 + 0.55 * h))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    return cx1, cy1, cx2, cy2


def detect_accessories(detector, is_mock: bool, frame, tracks, include_full_frame=False):
    """
    Run accessory detection for the given person tracks.

    Each person's head region is cropped and upscaled so small accessories
    (glasses, earbuds, masks on distant people) are actually resolvable.
    Returned boxes are in full-frame coordinates.
    """
    if detector is None or not tracks:
        return []
    if is_mock:
        return detector.detect([t["xyxy"] for t in tracks])
    if not getattr(detector, "available", False):
        return []

    frame_h, frame_w = frame.shape[:2]
    by_size = sorted(
        tracks,
        key=lambda t: (t["xyxy"][2] - t["xyxy"][0]) * (t["xyxy"][3] - t["xyxy"][1]),
        reverse=True,
    )[:MAX_CROPS_PER_FRAME]

    detections = []
    for t in by_size:
        box = _head_crop_box(t["xyxy"], frame_w, frame_h)
        if box is None:
            continue
        cx1, cy1, cx2, cy2 = box
        crop = frame[cy1:cy2, cx1:cx2]
        for d in detector.detect(crop, imgsz=CROP_IMGSZ):
            ax1, ay1, ax2, ay2 = d["xyxy"]
            mapped = dict(d)
            mapped["xyxy"] = [ax1 + cx1, ay1 + cy1, ax2 + cx1, ay2 + cy1]
            detections.append(mapped)

    if include_full_frame:
        detections.extend(detector.detect(frame))
    return detections


# ── reports / video helpers ─────────────────────────────────────────────────
def _write_reports(aggregate: dict, track_summaries: list) -> None:
    with open("final_report.json", "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "tracks": track_summaries}, f, indent=2)

    with open("final_report.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["track_id", "first_seen", "last_seen", "frames_observed", "cap", "mask", "glasses", "headphones"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in track_summaries:
            w.writerow({k: r[k] for k in fields})


def _open_raw_writer(stem: str, fps: float, width: int, height: int):
    for fourcc, ext in (("mp4v", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi")):
        path = JOBS_DIR / f"{stem}_raw{ext}"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
        if writer.isOpened():
            return writer, path
        writer.release()
    raise RuntimeError("Could not open a video writer for annotated output.")


def _transcode_h264(src: Path, dst: Path) -> bool:
    """Re-encode to H.264/yuv420p so every browser can play the result inline."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        logger.warning("ffmpeg unavailable (%s); serving the raw OpenCV video.", exc)
        return False
    cmd = [
        exe, "-y", "-loglevel", "error", "-i", str(src),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=900)
    except Exception as exc:
        logger.warning("ffmpeg transcode failed (%s).", exc)
        return False
    if proc.returncode != 0:
        logger.warning("ffmpeg transcode failed: %s", proc.stderr.decode(errors="ignore")[-400:])
        return False
    return dst.exists() and dst.stat().st_size > 0


def _cleanup_old_job_files(max_age_s: float = 3600.0) -> None:
    now = time.time()
    for p in JOBS_DIR.glob("*"):
        try:
            if p.is_file() and now - p.stat().st_mtime > max_age_s:
                p.unlink()
        except OSError:
            pass


def _flags_to_summary(tid, state, flags, confidence=None):
    return {
        "track_id": tid,
        "first_seen": state.first_frame if state else 0,
        "last_seen": state.last_frame if state else 0,
        "frames_observed": state.frame_count if state else 0,
        "confidence": confidence,
        **{k: bool(flags.get(k, False)) for k in ("cap", "mask", "glasses", "headphones")},
        "accessories_list": [k for k in ("cap", "mask", "glasses", "headphones") if flags.get(k)],
    }


def _aggregate(summaries: list, extra: dict | None = None) -> dict:
    agg = {
        "total_unique_persons": len(summaries),
        "total_cap": sum(1 for s in summaries if s["cap"]),
        "total_mask": sum(1 for s in summaries if s["mask"]),
        "total_glasses": sum(1 for s in summaries if s["glasses"]),
        "total_headphones": sum(1 for s in summaries if s["headphones"]),
        "total_none": sum(1 for s in summaries if not s["accessories_list"]),
    }
    if extra:
        agg.update(extra)
    return agg


# ── image analysis ──────────────────────────────────────────────────────────
def analyze_image_file(image_path: str, mode: str = "real"):
    """Run detection and association on a single image and generate results."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not load image from {image_path}")

    detector, is_mock = _accessory_engine(mode)

    with MODEL_LOCK:
        person_dets = person_detector.detect(frame, imgsz=TRACK_IMGSZ)
        tracks = [
            {"track_id": i + 1, "xyxy": p["xyxy"], "confidence": p["confidence"], "accessories": []}
            for i, p in enumerate(person_dets)
        ]
        raw_accs = detect_accessories(detector, is_mock, frame, tracks, include_full_frame=True)

    acc_map = associate_accessories_to_tracks(tracks, raw_accs, head_fraction=0.45)
    for t in tracks:
        t["accessories"] = acc_map.get(t["track_id"], [])

    vis = Visualizer(vis_cfg={"box_thickness": 2, "text_scale": 0.65, "show_confidence": True, "show_fps": False})
    summaries = []
    for t in tracks:
        names = {a["class_name"] for a in t["accessories"]}
        summaries.append(_flags_to_summary(
            t["track_id"], None, {k: k in names for k in ACCESSORY_KEYS}, t["confidence"],
        ))
    aggregate = _aggregate(summaries)
    live = {
        "persons": aggregate["total_unique_persons"],
        "caps": aggregate["total_cap"],
        "masks": aggregate["total_mask"],
        "glasses": aggregate["total_glasses"],
        "headphones": aggregate["total_headphones"],
    }
    annotated = vis.draw(frame, tracks, live_totals=live)

    out_name = f"annotated_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(JOBS_DIR / out_name), annotated)
    _write_reports(aggregate, summaries)

    return {
        "status": "success",
        "kind": "image",
        "mode": mode,
        "engine": _engine_label(detector, is_mock),
        "image_url": f"/static/jobs/{out_name}",
        "aggregate": aggregate,
        "tracks": summaries,
    }


# ── video jobs ──────────────────────────────────────────────────────────────
class VideoJob:
    def __init__(self, video_path: str, filename: str, mode: str):
        self.id = uuid.uuid4().hex[:12]
        self.video_path = video_path
        self.filename = filename
        self.mode = mode
        self.status = "queued"
        self.message = "Waiting for the detector..."
        self.progress = 0.0
        self.frames_done = 0
        self.frames_total = 0
        self.fps = 0.0
        self.live = {"persons": 0, "caps": 0, "masks": 0, "glasses": 0, "headphones": 0}
        self.frame_jpg: bytes | None = None
        self.frame_seq = 0
        self.result = None
        self.error = None
        self.created = time.time()
        self.lock = threading.Lock()

    def snapshot(self) -> dict:
        with self.lock:
            data = {
                "job_id": self.id,
                "filename": self.filename,
                "status": self.status,
                "message": self.message,
                "progress": round(self.progress, 4),
                "frames_done": self.frames_done,
                "frames_total": self.frames_total,
                "fps": round(self.fps, 2),
                "live": dict(self.live),
                "frame_seq": self.frame_seq,
            }
            if self.result is not None:
                data["result"] = self.result
            if self.error is not None:
                data["error"] = self.error
            return data


JOBS: dict[str, VideoJob] = {}
JOBS_LOCK = threading.Lock()


def _translate_cached(accs: list, old_box, new_box) -> list:
    """Shift accessory boxes from where a person was to where they are now."""
    dx = new_box[0] - old_box[0]
    dy = new_box[1] - old_box[1]
    out = []
    for a in accs:
        x1, y1, x2, y2 = a["xyxy"]
        b = dict(a)
        b["xyxy"] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        out.append(b)
    return out


def run_video_job(job: VideoJob) -> None:
    try:
        with MODEL_LOCK:
            with job.lock:
                job.status = "running"
                job.message = "Opening video..."
            _process_video(job)
    except Exception as exc:
        logger.exception("Video job %s failed", job.id)
        with job.lock:
            job.status = "error"
            job.error = str(exc)
            job.message = f"Failed: {exc}"


def _process_video(job: VideoJob) -> None:
    source = VideoSource(job.video_path)
    if source.width <= 0 or source.height <= 0:
        source.release()
        raise ValueError("Video reports an invalid frame size. The file may be corrupt.")

    src_fps = float(source.fps) if source.fps else 30.0
    total_frames = int(source.total_frames or 0)
    stride = max(1, int(round(src_fps / TARGET_PROCESS_FPS)))
    if total_frames > 0:
        stride = max(stride, int(math.ceil(total_frames / MAX_PROCESSED_FRAMES)))
    out_fps = max(1.0, src_fps / stride)
    expected = int(math.ceil(total_frames / stride)) if total_frames > 0 else 0

    scale = min(1.0, MAX_FRAME_WIDTH / float(source.width))
    out_w = int(round(source.width * scale))
    out_h = int(round(source.height * scale))

    detector, is_mock = _accessory_engine(job.mode)
    engine = _engine_label(detector, is_mock)

    with job.lock:
        job.frames_total = expected
        job.message = f"Loading tracker... ({engine})"

    tracker = PersonTracker(
        model_path=person_model_path(),
        confidence=max(person_confidence(), 0.30),
        iou_threshold=0.5,
        tracker_config="botsort.yaml",
        persist=True,
        person_class_id=0,
    )
    state_mgr = StateManager()
    registry = PersonTrackRegistry(min_track_frames=3 if 0 < expected < 60 else 5)
    exporter = Exporter(
        csv_path="tracks.csv", summary_path="summary.json",
        final_report_csv="final_report.csv", final_report_json="final_report.json",
    )
    vis = Visualizer(vis_cfg={"box_thickness": 2, "text_scale": 0.6, "show_confidence": True, "show_fps": True})
    writer, raw_path = _open_raw_writer(job.id, out_fps, out_w, out_h)

    acc_cache: dict[int, tuple[list, list]] = {}   # tid -> (person box at detection, accessories)
    last_annotated = None
    poster_frame = None
    poster_score = -1
    src_idx = 0
    processed = 0
    t0 = time.time()
    logger.info(
        "Video job %s: '%s' %dx%d -> %dx%d, %.1f fps, %d frames, stride=%d, engine=%s",
        job.id, job.filename, source.width, source.height, out_w, out_h,
        src_fps, total_frames, stride, engine,
    )

    try:
        with source:
            for frame in source:
                current = src_idx
                src_idx += 1
                if current % stride:
                    continue
                if scale < 1.0:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                tracks = [t for t in tracker.track(frame, imgsz=TRACK_IMGSZ) if t["track_id"] >= 0]
                for t in tracks:
                    registry.observe(t["track_id"], current, t.get("confidence", 0.0))
                    t["accessories"] = []
                if tracks:
                    state_mgr.update(current, tracks)

                run_acc = detector is not None and tracks and processed % ACCESSORY_EVERY == 0
                if run_acc:
                    raw_accs = detect_accessories(detector, is_mock, frame, tracks)
                    acc_map = associate_accessories_to_tracks(tracks, raw_accs, head_fraction=0.45)
                    for t in tracks:
                        tid = t["track_id"]
                        t["accessories"] = acc_map.get(tid, [])
                        acc_cache[tid] = (list(t["xyxy"]), t["accessories"])
                        state_mgr.update_state(tid, t["accessories"], current)
                        state_mgr.apply_temporal_voting(tid, VOTING)
                else:
                    for t in tracks:
                        cached = acc_cache.get(t["track_id"])
                        if cached:
                            t["accessories"] = _translate_cached(cached[1], cached[0], t["xyxy"])

                exporter.log_frame(current, tracks)

                confirmed = registry.confirmed_ids
                states = state_mgr.all_states()
                conf_states = [s for tid, s in states.items() if tid in confirmed]
                live = {
                    "persons": len(confirmed),
                    "caps": sum(1 for s in conf_states if s.final_cap),
                    "masks": sum(1 for s in conf_states if s.final_mask),
                    "glasses": sum(1 for s in conf_states if s.final_glasses),
                    "headphones": sum(1 for s in conf_states if s.final_headphones),
                }
                live["none"] = len(confirmed) - sum(
                    1 for s in conf_states
                    if s.final_cap or s.final_mask or s.final_glasses or s.final_headphones
                )

                annotated = vis.draw(frame, tracks, live_totals=live)
                writer.write(annotated)
                last_annotated = annotated
                score = len(tracks) * 10 + sum(len(t["accessories"]) for t in tracks)
                if score > poster_score:
                    poster_score, poster_frame = score, annotated
                processed += 1

                ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                elapsed = max(time.time() - t0, 1e-6)
                with job.lock:
                    if ok:
                        job.frame_jpg = jpg.tobytes()
                        job.frame_seq += 1
                    job.frames_done = processed
                    job.fps = processed / elapsed
                    job.live = live
                    if expected:
                        job.progress = min(processed / expected, 0.99)
                        remaining = max(expected - processed, 0) / max(job.fps, 1e-6)
                        job.message = (
                            f"Analyzing frame {processed}/{expected} "
                            f"({job.fps:.1f} fps, ~{int(remaining)}s left)"
                        )
                    else:
                        job.message = f"Analyzing frame {processed} ({job.fps:.1f} fps)"
    finally:
        writer.release()
        try:
            exporter.save(state_mgr)
        except Exception as exc:
            logger.warning("Could not save tracks.csv/summary.json (%s).", exc)

    if last_annotated is None:
        raise ValueError("No frames could be read from the uploaded video.")

    with job.lock:
        job.message = "Finalizing results and encoding playable video..."
        job.progress = 0.99

    poster_name = f"{job.id}_poster.jpg"
    cv2.imwrite(str(JOBS_DIR / poster_name), poster_frame if poster_frame is not None else last_annotated)

    state_mgr.finalize_accessories(VOTING)
    states = state_mgr.all_states()
    summaries = []
    for tid in sorted(registry.confirmed_ids):
        s = states.get(tid)
        flags = s.final_flags() if s else {}
        summaries.append(_flags_to_summary(tid, s, flags, round(s.confidence, 3) if s else None))

    aggregate = _aggregate(summaries, {
        "frames_processed": processed,
        "source_frames": total_frames,
        "frame_stride": stride,
        "accessory_source": _engine_source(detector, is_mock),
    })
    _write_reports(aggregate, summaries)

    final_path = JOBS_DIR / f"{job.id}.mp4"
    if _transcode_h264(raw_path, final_path):
        try:
            raw_path.unlink()
        except OSError:
            pass
        video_path = final_path
    else:
        video_path = raw_path

    elapsed = time.time() - t0
    logger.info(
        "Video job %s done: %d frames in %.1fs, persons=%d caps=%d masks=%d glasses=%d headphones=%d",
        job.id, processed, elapsed, aggregate["total_unique_persons"], aggregate["total_cap"],
        aggregate["total_mask"], aggregate["total_glasses"], aggregate["total_headphones"],
    )

    result = {
        "status": "success",
        "kind": "video",
        "mode": job.mode,
        "engine": engine,
        "image_url": f"/static/jobs/{poster_name}",
        "video_url": f"/static/jobs/{video_path.name}",
        "aggregate": aggregate,
        "tracks": summaries,
        "elapsed_s": round(elapsed, 1),
    }
    with job.lock:
        job.result = result
        job.live = {
            "persons": aggregate["total_unique_persons"],
            "caps": aggregate["total_cap"],
            "masks": aggregate["total_mask"],
            "glasses": aggregate["total_glasses"],
            "headphones": aggregate["total_headphones"],
            "none": aggregate["total_none"],
        }
        job.progress = 1.0
        job.status = "done"
        job.message = f"Done: {processed} frames analysed in {elapsed:.0f}s"


# ── routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    detector, is_mock = _accessory_engine("real")
    return render_template(
        "index.html",
        mock_enabled=mock_acc_detector is not None,
        engine_label=_engine_label(detector, is_mock),
        sample_available=(UPLOAD_DIR / "classroom_sample.jpg").exists(),
    )


@app.route("/api/status")
def api_status():
    detector, is_mock = _accessory_engine("real")
    return jsonify({
        "person_model": person_model_path(),
        "accessory_engine": _engine_label(detector, is_mock),
        "trained_accessory_model": bool(acc_detector.available),
        "mock_enabled": mock_acc_detector is not None,
    })


@app.route("/api/analyze/sample")
def analyze_sample():
    mode = request.args.get("mode", "real")
    sample_path = UPLOAD_DIR / "classroom_sample.jpg"
    if not sample_path.exists():
        return jsonify({"error": "Sample image not found. Upload your own photo or video instead."}), 404
    try:
        return jsonify(analyze_image_file(str(sample_path), mode=mode))
    except Exception as exc:
        logger.exception("Sample analysis failed")
        return jsonify({"error": str(exc)}), 500


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "File is too large. Maximum upload size is 500 MB."}), 413


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/api/analyze/upload", methods=["POST"])
def analyze_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    mode = request.form.get("mode", "real")
    original_name = file.filename
    ext = Path(original_name).suffix.lower()
    if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
        return jsonify({
            "error": f"Unsupported file type '{ext or '?'}'. Upload an image (JPG/PNG/WEBP) "
                     "or a video (MP4/AVI/MOV/MKV/WEBM)."
        }), 400

    stem = secure_filename(Path(original_name).stem) or "upload"
    save_path = UPLOAD_DIR / f"{stem}_{uuid.uuid4().hex[:6]}{ext}"
    file.save(str(save_path))
    if save_path.stat().st_size == 0:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": "Uploaded file is empty (0 bytes)."}), 400
    logger.info("Saved upload '%s' as '%s'", original_name, save_path)

    if ext in IMAGE_EXTS:
        try:
            return jsonify(analyze_image_file(str(save_path), mode=mode))
        except Exception as exc:
            logger.exception("Failed to analyze image '%s'", save_path)
            return jsonify({"error": str(exc)}), 500

    _cleanup_old_job_files()
    job = VideoJob(str(save_path), original_name, mode)
    with JOBS_LOCK:
        JOBS[job.id] = job
    threading.Thread(target=run_video_job, args=(job,), daemon=True).start()
    return jsonify({"status": "queued", "kind": "video", "job_id": job.id}), 202


@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job.snapshot())


@app.route("/api/jobs/<job_id>/frame")
def job_frame(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    with job.lock:
        data = job.frame_jpg
    if not data:
        return ("", 204)
    resp = Response(data, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/report/csv")
def download_csv():
    if os.path.exists("final_report.csv"):
        return send_file("final_report.csv", as_attachment=True, download_name="final_report.csv")
    return jsonify({"error": "No CSV report available yet"}), 404


@app.route("/api/report/json")
def download_json():
    if os.path.exists("final_report.json"):
        return send_file("final_report.json", as_attachment=True, download_name="final_report.json")
    return jsonify({"error": "No JSON report available yet"}), 404


if __name__ == "__main__":
    print("\n=======================================================")
    print("  Student & Accessory Vision Analytics Web UI Online!")
    print("  Open in Browser: http://127.0.0.1:5000")
    print("=======================================================\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
