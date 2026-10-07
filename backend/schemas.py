"""Response shapes for the A.E.G.I.S API.

Counts are filled from the existing exporter. Accessory fields stay null when
the accessory model did not run, so a zero is never invented.
"""

from __future__ import annotations

from pydantic import BaseModel


class RootStatus(BaseModel):
    name: str
    status: str


class HealthStatus(BaseModel):
    status: str
    ai_engine: str


class ModelChannel(BaseModel):
    status: str
    model: str
    mode: str | None = None
    classes: list[str] | None = None
    detail: str | None = None


class ModelStatus(BaseModel):
    person: ModelChannel
    accessory: ModelChannel


class TrackResult(BaseModel):
    track_id: int
    cap: bool | None
    mask: bool | None
    glasses: bool | None
    headphones: bool | None
    frames_observed: int


class AnalysisSummary(BaseModel):
    total_unique_persons: int
    wearing_cap: int | None
    wearing_mask: int | None
    wearing_glasses: int | None
    wearing_headphones: int | None
    plain: int | None


class AccessoryRunState(BaseModel):
    status: str
    mode: str | None = None
    model: str | None = None
    source: str | None = None


class AnalysisResponse(BaseModel):
    status: str
    analysis_id: str
    video: str
    accessory_ai: AccessoryRunState
    summary: AnalysisSummary
    tracks: list[TrackResult]
