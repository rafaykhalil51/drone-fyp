"""
A.E.G.I.S Vision Intelligence API.

Start from the project root:

    uvicorn backend.main:app --reload

The Streamlit and Flask portals are not started by this process.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.schemas import AnalysisResponse, HealthStatus, ModelStatus, RootStatus
from backend.services import analysis_service
from backend.services.analysis_service import AnalysisError

logger = logging.getLogger("aegis.api")

# Local dev frontends, plus one future deployed origin. Never "*".
_DEV_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_configured_origin = os.environ.get("AEGIS_FRONTEND_ORIGIN", "").strip()
_ALLOWED_ORIGINS = list(_DEV_ORIGINS)
if _configured_origin and _configured_origin not in _ALLOWED_ORIGINS:
    _ALLOWED_ORIGINS.append(_configured_origin)

app = FastAPI(
    title="A.E.G.I.S Vision Intelligence API",
    summary="HTTP access to the existing person, tracking, and accessory pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(AnalysisError)
async def analysis_error_handler(_request, exc: AnalysisError):
    status = exc.status_code
    if exc.detail == "Unknown analysis id.":
        status = 404
    return JSONResponse(status_code=status, content={"detail": exc.detail})


@app.get("/", response_model=RootStatus)
def root() -> RootStatus:
    return RootStatus(name="A.E.G.I.S Vision Intelligence API", status="online")


@app.get("/api/health", response_model=HealthStatus)
def health() -> HealthStatus:
    return HealthStatus(status="online", ai_engine=analysis_service.ai_engine_status())


@app.get("/api/models/status", response_model=ModelStatus, response_model_exclude_none=True)
def models_status() -> ModelStatus:
    return ModelStatus(**analysis_service.model_status())


@app.post("/api/analyze/video", response_model=AnalysisResponse)
async def analyze_video(video: UploadFile = File(...)) -> AnalysisResponse:
    slot = analysis_service.reserve_upload(video.filename)
    try:
        with slot.video_path.open("wb") as handle:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        result = await run_in_threadpool(analysis_service.analyze_reserved, slot)
        return AnalysisResponse(**result)
    except AnalysisError:
        raise
    except Exception as exc:
        logger.exception("Unhandled analysis error.")
        if not analysis_service.known_analysis(slot.analysis_id):
            analysis_service.discard(slot)
        raise HTTPException(
            status_code=500,
            detail=f"AI processing failed. {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        await video.close()


def _analysis_http_error(exc: AnalysisError) -> HTTPException:
    status = exc.status_code
    if exc.detail in {
        "Unknown analysis id.",
        "No annotated frame is available yet.",
        "Annotated video is not ready.",
    }:
        status = 404
    return HTTPException(status_code=status, detail=exc.detail)


@app.post("/api/analysis/start")
async def analysis_start(video: UploadFile = File(...)) -> dict:
    slot = analysis_service.reserve_upload(video.filename)
    try:
        with slot.video_path.open("wb") as handle:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        return analysis_service.start_live(slot)
    except AnalysisError:
        raise
    except Exception as exc:
        logger.exception("Unhandled live analysis error.")
        analysis_service.discard(slot)
        raise HTTPException(
            status_code=500,
            detail=f"AI processing failed. {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        await video.close()


@app.get("/api/analysis/{analysis_id}/status")
def analysis_status(analysis_id: str) -> dict:
    try:
        return analysis_service.live_status(analysis_id)
    except AnalysisError as exc:
        raise _analysis_http_error(exc) from exc


@app.get("/api/analysis/{analysis_id}/stream")
def analysis_stream(analysis_id: str):
    try:
        analysis_service.live_job(analysis_id)
    except AnalysisError as exc:
        raise _analysis_http_error(exc) from exc
    return StreamingResponse(
        analysis_service.iter_mjpeg(analysis_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, private",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/analysis/{analysis_id}/video")
def analysis_video(analysis_id: str, download: int = 0):
    """Serve the saved annotated MP4. Playback does not run the AI pipeline."""
    try:
        path = analysis_service.annotated_video(analysis_id)
    except AnalysisError as exc:
        raise _analysis_http_error(exc) from exc
    return FileResponse(
        path,
        media_type="video/mp4",
        filename="annotated.mp4",
        content_disposition_type="attachment" if download else "inline",
    )


@app.get("/api/analysis/{analysis_id}/frame")
def analysis_frame(analysis_id: str):
    try:
        jpeg = analysis_service.latest_jpeg(analysis_id)
    except AnalysisError as exc:
        raise _analysis_http_error(exc) from exc
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/reports/{analysis_id}/json")
def report_json(analysis_id: str):
    if not analysis_service.known_analysis(analysis_id):
        raise HTTPException(status_code=404, detail="Unknown analysis id.")
    path = analysis_service.report_path(analysis_id, "json")
    return FileResponse(
        path,
        media_type="application/json",
        filename="final_report.json",
    )


@app.get("/api/reports/{analysis_id}/csv")
def report_csv(analysis_id: str):
    if not analysis_service.known_analysis(analysis_id):
        raise HTTPException(status_code=404, detail="Unknown analysis id.")
    path = analysis_service.report_path(analysis_id, "csv")
    return FileResponse(
        path,
        media_type="text/csv",
        filename="final_report.csv",
    )
