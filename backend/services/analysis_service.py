"""
Connect the API to the existing A.E.G.I.S pipeline.

Video analysis calls main.main() with a temporary config. That function is
the batch pipeline already used by `python main.py`: person tracking,
accessory detection, association, temporal voting, unique counting, and the
report exporter. This module does not reimplement any of those steps.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}

# Uploads and per-analysis reports live here. `tmp/` is already gitignored.
RUNTIME = Path(__file__).resolve().parents[2] / "tmp" / "aegis_api"

_STORE: dict[str, dict] = {}
_RUN_LOCK = threading.Lock()
_PIPELINE = None
# Completed jobs stay replayable. Older ones are removed so tmp/ does not grow without a limit.
_KEEP_FINISHED = 8


class AnalysisError(Exception):
    """Expected analysis failure. The server stays up and returns this message."""

    status_code = 500

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class UnsupportedFormat(AnalysisError):
    status_code = 415


class VideoUnreadable(AnalysisError):
    status_code = 400


class PersonModelUnavailable(AnalysisError):
    status_code = 503


class ProcessingFailed(AnalysisError):
    status_code = 500


class ReportFailed(AnalysisError):
    status_code = 500


class AnalysisBusy(AnalysisError):
    status_code = 409


@dataclass
class UploadSlot:
    analysis_id: str
    directory: Path
    video_path: Path
    original_name: str


def _project_root() -> Path:
    from app_config import PROJECT_ROOT

    return PROJECT_ROOT


def _ensure_import_path() -> None:
    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _pipeline_main():
    """Load the root main.py without colliding with backend.main."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    _ensure_import_path()
    path = _project_root() / "main.py"
    spec = importlib.util.spec_from_file_location("aegis_pipeline_main", path)
    if spec is None or spec.loader is None:
        raise ProcessingFailed(f"Cannot load the existing pipeline at {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["aegis_pipeline_main"] = module
    spec.loader.exec_module(module)
    _PIPELINE = module.main
    return _PIPELINE


def _inside_runtime(path: Path) -> bool:
    try:
        path.resolve().relative_to(RUNTIME.resolve())
    except ValueError:
        return False
    return True


def _safe_unlink(path: Path) -> None:
    try:
        resolved = path.resolve()
    except OSError:
        return
    if not _inside_runtime(resolved) or not resolved.is_file():
        return
    try:
        resolved.unlink()
    except OSError as exc:
        logger.warning("Could not remove temporary file %s (%s).", resolved, exc)


def discard(slot: UploadSlot) -> None:
    """Remove one analysis directory. Refuses to delete anything outside tmp/aegis_api."""
    directory = slot.directory.resolve()
    if _inside_runtime(directory) and directory.is_dir():
        shutil.rmtree(directory, ignore_errors=True)
    _STORE.pop(slot.analysis_id, None)


def reserve_upload(filename: str | None) -> UploadSlot:
    original = Path(filename or "").name
    suffix = Path(original).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFormat(
            "Unsupported video format. Upload a .mp4, .mov, .avi, or .mkv file."
        )

    analysis_id = uuid.uuid4().hex
    directory = RUNTIME / analysis_id
    directory.mkdir(parents=True, exist_ok=True)
    return UploadSlot(
        analysis_id=analysis_id,
        directory=directory,
        video_path=directory / f"upload{suffix}",
        original_name=original or f"upload{suffix}",
    )


def ai_engine_status() -> str:
    """Import the existing pipeline modules. Does not open a video or run inference."""
    try:
        _ensure_import_path()
        import cv2  # noqa: F401
        from accessory_detector import AccessoryDetector  # noqa: F401
        from association import associate_accessories_to_tracks  # noqa: F401
        from counter import AccessoryCounter  # noqa: F401
        from exporter import Exporter  # noqa: F401
        from state_manager import StateManager  # noqa: F401
        from tracker import PersonTracker  # noqa: F401
        from video_source import VideoSource  # noqa: F401
        from voting_config import VOTING  # noqa: F401
    except Exception:
        logger.exception("AI engine imports failed.")
        return "unavailable"
    if VOTING is None or AccessoryDetector is None:
        return "unavailable"
    return "ready"


def _abs_from_root(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return _project_root() / path


# Successful loads are kept so /api/models/status does not rebuild CLIP
# embeddings on every request. A failed load is not cached.
_PERSON_LOADED: str | None = None
_WORLD_DETECTOR = None


def _canonical_slots(labels) -> list[str]:
    """Normalized cap/mask/glasses/headphones slots the existing matcher accepts."""
    from model_loader import REQUIRED_ACCESSORY_CLASSES, match_hud_slot

    found = {match_hud_slot(name) for name in labels}
    return [slot for slot in REQUIRED_ACCESSORY_CLASSES if slot in found]


def _person_channel(configured_person: str, resolution) -> dict:
    """Load the same person weights resolve_models() selected. Active only if that load works."""
    global _PERSON_LOADED
    from model_loader import load_person_model

    model_name = resolution.person_path or configured_person
    if not _abs_from_root(model_name).is_file():
        return {
            "status": "unavailable",
            "model": configured_person,
            "detail": f"Person weights were not found at {configured_person}.",
        }
    if _PERSON_LOADED == model_name:
        return {"status": "active", "model": model_name}
    try:
        load_person_model(model_name)
    except Exception as exc:
        logger.exception("Person model failed to load.")
        return {
            "status": "unavailable",
            "model": model_name,
            "detail": f"Person model failed to load: {exc}",
        }
    _PERSON_LOADED = model_name
    return {"status": "active", "model": model_name}


def _trained_accessory_channel(resolution) -> dict | None:
    """
    Trained accessory weights, using load_accessory_detector().

    Returns None when this is not the detector the portal would run, so the
    caller can fall through to the configured YOLO-World path.
    """
    from model_loader import load_accessory_detector

    # A missing or unusable custom file is not the live detector. The portal
    # then uses YOLO-World when that fallback is enabled in config.yaml.
    if resolution.accessory_status != "active" or not resolution.accessory_path:
        return None
    weight = _abs_from_root(resolution.accessory_path)
    if not weight.is_file():
        return None

    detector = load_accessory_detector(str(weight))
    classes = _canonical_slots(getattr(detector, "class_names", ()))
    if not (detector.available and getattr(detector, "usable", True) and classes):
        return None
    return {
        "status": "active",
        "mode": resolution.mode,
        "model": resolution.accessory_path,
        "classes": classes,
    }


def _world_accessory_channel() -> dict | None:
    """Load the portal's YOLO-World detector. Active only when that object is available."""
    global _WORLD_DETECTOR
    from app_config import zero_shot_enabled, zero_shot_model_path
    from model_loader import load_zero_shot_detector

    if not zero_shot_enabled():
        return None

    model_name = zero_shot_model_path()
    if _WORLD_DETECTOR is None or not getattr(_WORLD_DETECTOR, "available", False):
        _WORLD_DETECTOR = load_zero_shot_detector()
    detector = _WORLD_DETECTOR
    if detector is None or not getattr(detector, "available", False):
        detail = "YOLO-World weights were not found."
        if detector is not None and getattr(detector, "load_error", None):
            detail = f"YOLO-World did not load ({detector.load_error})."
        return {
            "status": "unavailable",
            "mode": "world",
            "model": model_name,
            "classes": [],
            "detail": detail,
        }

    classes = _canonical_slots(getattr(detector, "vocabulary", ()) or detector.class_names)
    if not classes:
        return {
            "status": "invalid",
            "mode": "world",
            "model": Path(detector.model_path).name,
            "classes": [],
            "detail": "YOLO-World loaded but none of its prompts map to cap, mask, glasses, or headphones.",
        }
    return {
        "status": "active",
        "mode": "world",
        "model": Path(detector.model_path).name,
        "classes": classes,
    }


def model_status() -> dict:
    """
    Same accessory choice as the portal: trained weights when they load,
    otherwise the configured YOLO-World detector from model_loader.py.

    Status is active only after that existing loader succeeds.
    """
    _ensure_import_path()
    from app_config import accessory_model_path, person_model_path
    from model_loader import resolve_models

    configured_person = person_model_path()
    configured_accessory = accessory_model_path()

    try:
        resolution = resolve_models(configured_accessory)
    except Exception as exc:
        logger.exception("Model resolution failed.")
        detail = f"Model inspection failed: {exc}"
        return {
            "person": {
                "status": "unavailable",
                "model": configured_person,
                "detail": detail,
            },
            "accessory": {
                "status": "unavailable",
                "mode": None,
                "model": configured_accessory,
                "classes": [],
                "detail": detail,
            },
        }

    with _RUN_LOCK:
        person = _person_channel(configured_person, resolution)
        accessory = _trained_accessory_channel(resolution)
        if accessory is None:
            accessory = _world_accessory_channel()
        if accessory is None:
            accessory = {
                "status": "unavailable",
                "mode": None,
                "model": configured_accessory,
                "classes": [],
                "detail": f"Accessory weights were not found at {configured_accessory}.",
            }

    return {"person": person, "accessory": accessory}


def _assert_video_opens(path: Path, label: str) -> None:
    from video_source import VideoSource

    try:
        source = VideoSource(str(path))
    except FileNotFoundError as exc:
        raise VideoUnreadable(f"Video cannot be opened: {label}.") from exc

    try:
        if source.width < 1 or source.height < 1:
            raise VideoUnreadable(
                f"Video cannot be opened: {label} has no readable frames."
            )
    finally:
        source.release()


def _job_config(slot: UploadSlot) -> Path:
    """Copy the live config and point only the input and report paths at this job."""
    from app_config import load_config

    cfg = copy.deepcopy(load_config())
    if not isinstance(cfg.get("video"), dict):
        raise ProcessingFailed("config.yaml has no video section.")
    if not isinstance(cfg.get("export"), dict):
        raise ProcessingFailed("config.yaml has no export section.")

    cfg["video"]["source"] = str(slot.video_path.resolve())
    cfg["video"]["output"] = str((slot.directory / "output_annotated.mp4").resolve())
    cfg["export"]["csv_path"] = str((slot.directory / "tracks.csv").resolve())
    cfg["export"]["summary_path"] = str((slot.directory / "summary.json").resolve())
    cfg["export"]["final_report_csv"] = str((slot.directory / "final_report.csv").resolve())
    cfg["export"]["final_report_json"] = str((slot.directory / "final_report.json").resolve())

    config_path = slot.directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def _person_weights_present() -> bool:
    from app_config import person_model_path
    from model_loader import PERSON_FALLBACKS

    names = [person_model_path(), *PERSON_FALLBACKS]
    return any(_abs_from_root(name).is_file() for name in names if name)


def _classify(exc: Exception) -> AnalysisError:
    text = f"{type(exc).__name__}: {exc}"
    lower = text.lower()
    if isinstance(exc, FileNotFoundError) or "cannot open video" in lower:
        return VideoUnreadable(f"Video cannot be opened. {text}")
    if "final_report" in lower or "exporter" in lower:
        return ReportFailed(f"Report generation failed. {text}")
    model_load_failed = any(
        token in lower
        for token in ("yolov8n", "yolo11n", "download failure", "failed to load")
    )
    if model_load_failed or not _person_weights_present():
        return PersonModelUnavailable(f"Person model unavailable. {text}")
    return ProcessingFailed(f"AI processing failed. {text}")


def _build_result(slot: UploadSlot, report: dict) -> dict:
    aggregate = report.get("aggregate")
    rows = report.get("tracks")
    if not isinstance(aggregate, dict) or not isinstance(rows, list):
        raise ReportFailed("final_report.json did not contain aggregate totals and tracks.")

    source = str(aggregate.get("accessory_source") or "none (accessory AI offline)")
    measured = not source.startswith("none")
    if source.startswith("trained:"):
        accessory_state = {
            "status": "active",
            "mode": "trained",
            "model": source.split(":", 1)[1],
            "source": source,
        }
    elif source.startswith("zero-shot"):
        from app_config import zero_shot_model_path
        accessory_state = {
            "status": "active",
            "mode": "world",
            "model": zero_shot_model_path(),
            "source": source,
        }
    else:
        accessory_state = {
            "status": "unavailable",
            "mode": None,
            "model": None,
            "source": source,
        }

    tracks = []
    plain = 0
    for row in rows:
        if not isinstance(row, dict) or "track_id" not in row:
            raise ReportFailed("A track row in final_report.json is missing track_id.")
        flags = {
            key: (bool(row.get(key, False)) if measured else None)
            for key in ("cap", "mask", "glasses", "headphones")
        }
        if measured and not any(flags.values()):
            plain += 1
        tracks.append({
            "track_id": int(row["track_id"]),
            "frames_observed": int(row.get("frames_observed") or 0),
            **flags,
        })

    def _count(key: str) -> int | None:
        if not measured:
            return None
        value = aggregate.get(key, 0)
        return int(value)

    return {
        "status": "completed",
        "analysis_id": slot.analysis_id,
        "video": slot.original_name,
        "accessory_ai": accessory_state,
        "summary": {
            "total_unique_persons": int(aggregate.get("total_unique_persons") or 0),
            "wearing_cap": _count("total_cap"),
            "wearing_mask": _count("total_mask"),
            "wearing_glasses": _count("total_glasses"),
            "wearing_headphones": _count("total_headphones"),
            "plain": plain if measured else None,
        },
        "tracks": tracks,
    }


def analyze_reserved(slot: UploadSlot) -> dict:
    """Run the existing pipeline on one uploaded file and keep its reports."""
    if not slot.video_path.is_file() or slot.video_path.stat().st_size <= 0:
        discard(slot)
        raise VideoUnreadable("Uploaded file is empty or missing.")

    report_json = slot.directory / "final_report.json"
    report_csv = slot.directory / "final_report.csv"
    config_path: Path | None = None

    try:
        _assert_video_opens(slot.video_path, slot.original_name)
        config_path = _job_config(slot)
        with _RUN_LOCK:
            previous = os.getcwd()
            os.chdir(_project_root())
            try:
                _pipeline_main()(str(config_path))
            finally:
                os.chdir(previous)

        if not report_json.is_file() or not report_csv.is_file():
            raise ReportFailed(
                "The pipeline finished without writing final_report.json and final_report.csv."
            )
        report = json.loads(report_json.read_text(encoding="utf-8"))
        result = _build_result(slot, report)
    except AnalysisError:
        discard(slot)
        raise
    except Exception as exc:
        logger.exception("Pipeline failed for %s.", slot.original_name)
        failure = _classify(exc)
        discard(slot)
        raise failure from exc
    finally:
        _safe_unlink(slot.video_path)
        _safe_unlink(slot.directory / "output_annotated.mp4")
        if config_path is not None:
            _safe_unlink(config_path)
        _safe_unlink(slot.directory / "tracks.csv")
        _safe_unlink(slot.directory / "summary.json")

    _STORE[slot.analysis_id] = {
        "json_path": report_json,
        "csv_path": report_csv,
        "result": result,
    }
    return result


def report_path(analysis_id: str, kind: str) -> Path:
    item = _STORE.get(analysis_id)
    if item is None:
        raise AnalysisError("Unknown analysis id.")
    # Reuse the not-found status by raising a dedicated code below.
    path = item["json_path"] if kind == "json" else item["csv_path"]
    if not isinstance(path, Path) or not path.is_file():
        raise ReportFailed(f"The {kind.upper()} report for this analysis is missing.")
    return path


def known_analysis(analysis_id: str) -> bool:
    return analysis_id in _STORE or analysis_id in _JOBS


def _jpeg_quality() -> int:
    raw = os.environ.get("AEGIS_STREAM_JPEG_QUALITY", "75")
    try:
        quality = int(raw)
    except ValueError:
        quality = 75
    return min(80, max(70, quality))


def _pace_stream() -> bool:
    return os.environ.get("AEGIS_STREAM_PACE", "1").strip().lower() not in {"0", "false", "off", "no"}


def _summary_from_states(states: dict) -> tuple[dict, list]:
    tracks = []
    plain = 0
    caps = masks = glasses = headphones = 0
    for track_id, state in sorted(states.items()):
        if int(track_id) < 0:
            continue
        flags = {
            "cap": bool(state.final_cap),
            "mask": bool(state.final_mask),
            "glasses": bool(state.final_glasses),
            "headphones": bool(state.final_headphones),
        }
        caps += int(flags["cap"])
        masks += int(flags["mask"])
        glasses += int(flags["glasses"])
        headphones += int(flags["headphones"])
        if not any(flags.values()):
            plain += 1
        tracks.append({"track_id": int(track_id), **flags})
    summary = {
        "total_unique_persons": len(tracks),
        "wearing_cap": caps,
        "wearing_mask": masks,
        "wearing_glasses": glasses,
        "wearing_headphones": headphones,
        "plain": plain,
    }
    return summary, tracks


@dataclass
class LiveJob:
    analysis_id: str
    video: str
    status: str = "processing"
    current_frame: int = 0
    total_frames: int | None = None
    fps: float = 0.0
    summary: dict = field(default_factory=dict)
    tracks: list = field(default_factory=list)
    result: dict | None = None
    detail: str | None = None
    directory: Path | None = None
    video_ready: bool = False
    video_path: Path | None = None
    video_detail: str | None = None
    latest_jpeg: bytes | None = None
    frames: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=8))
    closed: bool = False


_JOBS: dict[str, LiveJob] = {}
_JOBS_LOCK = threading.Lock()


def _publish_frame(job: LiveJob, jpeg: bytes) -> None:
    if job.frames.full():
        try:
            job.frames.get_nowait()
        except queue.Empty:
            pass
    try:
        job.frames.put_nowait(jpeg)
    except queue.Full:
        pass
    job.latest_jpeg = jpeg


def _observe_frame(job: LiveJob, payload: dict) -> None:
    import cv2

    annotated = payload.get("annotated")
    if annotated is None:
        return
    ok, encoded = cv2.imencode(
        ".jpg",
        annotated,
        [int(cv2.IMWRITE_JPEG_QUALITY), _jpeg_quality()],
    )
    if not ok:
        return
    total = int(payload.get("total_frames") or 0)
    states = payload.get("states") or {}
    summary, tracks = _summary_from_states(states)
    persons = payload.get("live_totals") or {}
    if "persons" in persons:
        summary["total_unique_persons"] = int(persons.get("persons") or 0)
    job.current_frame = int(payload.get("frame_index") or 0) + 1
    job.total_frames = total if total > 0 else None
    job.fps = float(payload.get("fps") or 0)
    job.summary = summary
    job.tracks = tracks
    _publish_frame(job, encoded.tobytes())


def _ffmpeg_exe() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except Exception:
        return None
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    return exe if exe and Path(exe).is_file() else None


def _prepare_playback(slot: UploadSlot, job: LiveJob) -> None:
    """Convert the pipeline's annotated MP4 to browser-playable H.264. No inference."""
    source = slot.directory / "output_annotated.mp4"
    dest = slot.directory / "annotated.mp4"
    if not source.is_file() or source.stat().st_size <= 0:
        job.video_ready = False
        job.video_detail = "The pipeline did not write an annotated video."
        return

    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        job.video_ready = False
        job.video_path = source
        job.video_detail = (
            "Annotated video was saved as MPEG-4 Part 2 (mp4v), which browsers cannot play. "
            "ffmpeg is not available to convert it to H.264."
        )
        return

    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(source),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(dest),
            ],
            capture_output=True,
            timeout=180,
            check=False,
        )
    except Exception as exc:
        job.video_ready = False
        job.video_path = source
        job.video_detail = f"H.264 conversion failed: {exc}"
        logger.exception("Annotated video conversion failed for %s.", slot.analysis_id)
        return

    if completed.returncode != 0 or not dest.is_file() or dest.stat().st_size <= 0:
        tail = (completed.stderr or b"").decode("utf-8", errors="replace")[-400:].strip()
        job.video_ready = False
        job.video_path = source
        job.video_detail = f"H.264 conversion failed. {tail}".strip()
        _safe_unlink(dest)
        logger.error("Annotated video conversion failed for %s: %s", slot.analysis_id, tail)
        return

    job.video_ready = True
    job.video_path = dest
    job.video_detail = None
    stored = _STORE.get(slot.analysis_id)
    if stored is not None:
        stored["video_path"] = dest
    logger.info(
        "Annotated H.264 video ready for %s (%s bytes). Replay does not run inference.",
        slot.analysis_id,
        dest.stat().st_size,
    )


def _finish_live(slot: UploadSlot, job: LiveJob, config_path: Path | None) -> None:
    try:
        job.frames.put(None, timeout=2)
    except queue.Full:
        job.closed = True
    job.closed = True
    _safe_unlink(slot.video_path)
    playback = slot.directory / "annotated.mp4"
    if playback.is_file() and playback.stat().st_size > 0:
        _safe_unlink(slot.directory / "output_annotated.mp4")
    if config_path is not None:
        _safe_unlink(config_path)
    _safe_unlink(slot.directory / "tracks.csv")
    _safe_unlink(slot.directory / "summary.json")


def _forget_analysis(analysis_id: str) -> None:
    job = _JOBS.pop(analysis_id, None)
    _STORE.pop(analysis_id, None)
    directory = job.directory if job is not None and job.directory is not None else RUNTIME / analysis_id
    if _inside_runtime(directory) and directory.is_dir():
        shutil.rmtree(directory, ignore_errors=True)


def _prune_finished() -> None:
    finished = [aid for aid, job in _JOBS.items() if job.status in {"completed", "failed"}]
    for aid in finished[:-_KEEP_FINISHED]:
        _forget_analysis(aid)


def _run_live(slot: UploadSlot, job: LiveJob) -> None:
    config_path: Path | None = None
    try:
        _assert_video_opens(slot.video_path, slot.original_name)
        config_path = _job_config(slot)

        def observer(payload: dict) -> None:
            _observe_frame(job, payload)

        with _RUN_LOCK:
            previous = os.getcwd()
            os.chdir(_project_root())
            try:
                logger.info("AI inference started for analysis %s", slot.analysis_id)
                _pipeline_main()(str(config_path), frame_observer=observer)
            finally:
                os.chdir(previous)
        logger.info("AI inference finished for analysis %s. Replay will not run it again.", slot.analysis_id)

        report_json = slot.directory / "final_report.json"
        report_csv = slot.directory / "final_report.csv"
        if not report_json.is_file() or not report_csv.is_file():
            raise ReportFailed(
                "The pipeline finished without writing final_report.json and final_report.csv."
            )
        report = json.loads(report_json.read_text(encoding="utf-8"))
        result = _build_result(slot, report)
        _STORE[slot.analysis_id] = {
            "json_path": report_json,
            "csv_path": report_csv,
            "result": result,
        }
        _prepare_playback(slot, job)
        job.result = result
        job.summary = result["summary"]
        job.tracks = result["tracks"]
        job.status = "completed"
    except AnalysisError as exc:
        job.status = "failed"
        job.detail = exc.detail
        logger.exception("Live analysis failed for %s.", slot.original_name)
    except Exception as exc:
        failure = _classify(exc)
        job.status = "failed"
        job.detail = failure.detail
        logger.exception("Live analysis failed for %s.", slot.original_name)
    finally:
        _finish_live(slot, job, config_path)


def start_live(slot: UploadSlot) -> dict:
    if not slot.video_path.is_file() or slot.video_path.stat().st_size <= 0:
        discard(slot)
        raise VideoUnreadable("Uploaded file is empty or missing.")

    with _JOBS_LOCK:
        if any(item.status == "processing" for item in _JOBS.values()):
            discard(slot)
            raise AnalysisBusy("An analysis is already running.")
        _prune_finished()
        job = LiveJob(
            analysis_id=slot.analysis_id,
            video=slot.original_name,
            directory=slot.directory,
        )
        _JOBS[slot.analysis_id] = job

    threading.Thread(
        target=_run_live,
        args=(slot, job),
        name=f"aegis-live-{slot.analysis_id[:8]}",
        daemon=True,
    ).start()
    return {
        "analysis_id": slot.analysis_id,
        "status": "processing",
        "video": slot.original_name,
    }


def live_job(analysis_id: str) -> LiveJob:
    job = _JOBS.get(analysis_id)
    if job is None:
        raise AnalysisError("Unknown analysis id.")
    return job


def live_status(analysis_id: str) -> dict:
    job = live_job(analysis_id)
    payload = {
        "analysis_id": job.analysis_id,
        "status": job.status,
        "current_frame": job.current_frame,
        "total_frames": job.total_frames,
        "summary": job.summary or {
            "total_unique_persons": 0,
            "wearing_cap": 0,
            "wearing_mask": 0,
            "wearing_glasses": 0,
            "wearing_headphones": 0,
            "plain": 0,
        },
        "tracks": job.tracks,
        "video_ready": bool(job.video_ready and job.video_path and job.video_path.is_file()),
        "video_url": f"/api/analysis/{job.analysis_id}/video" if job.video_ready else None,
    }
    if job.detail:
        payload["detail"] = job.detail
    if job.video_detail:
        payload["video_detail"] = job.video_detail
    if job.result is not None:
        payload["result"] = job.result
    return payload


def annotated_video(analysis_id: str) -> Path:
    """Completed annotated MP4. This does not run detection or tracking."""
    logger.info("Replay video requested for %s. Inference was not started.", analysis_id)
    job = _JOBS.get(analysis_id)
    path = job.video_path if job is not None else None
    if path is None:
        stored = _STORE.get(analysis_id)
        if stored is not None:
            path = stored.get("video_path")
    if not isinstance(path, Path) or not path.is_file():
        raise AnalysisError("Annotated video is not ready.")
    return path


def iter_mjpeg(analysis_id: str):
    """Yield one MJPEG part per annotated frame. Disconnects do not stop analysis."""
    job = live_job(analysis_id)
    last_emit = 0.0
    try:
        while True:
            try:
                item = job.frames.get(timeout=1.0)
            except queue.Empty:
                if job.closed and job.frames.empty():
                    break
                continue
            if item is None:
                break
            if _pace_stream() and job.fps > 1:
                wait = (1.0 / job.fps) - (time.perf_counter() - last_emit)
                if wait > 0:
                    time.sleep(min(wait, 1.0))
            last_emit = time.perf_counter()
            header = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(item)}\r\n\r\n".encode("ascii")
            )
            yield header + item + b"\r\n"
    except GeneratorExit:
        return


def latest_jpeg(analysis_id: str) -> bytes:
    job = live_job(analysis_id)
    if not job.latest_jpeg:
        raise AnalysisError("No annotated frame is available yet.")
    return job.latest_jpeg
