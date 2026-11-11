"""FastAPI service: ``POST /ingest``, ``POST /query``, ``POST /locate``, ``GET /health`` (+ jobs,
videos and an HTML map of a location estimate at ``POST /locate/map``)."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from vidquery import __version__
from vidquery.config import Settings, load_config
from vidquery.geo.locate import GeoLocator, LocateResult
from vidquery.pipeline import IngestStats, QueryResult, VidQuery
from vidquery.types import VideoEntry

log = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    video_path: str = Field(description="Path on the server (video file or image-sequence dir).")
    video_id: str | None = None
    sample_fps: float | None = Field(None, gt=0)
    replace: bool = True
    background: bool = Field(False, description="Return a job id immediately (poll /jobs/{id}).")


class JobStatus(BaseModel):
    job_id: str
    state: Literal["queued", "running", "done", "failed"]
    video_path: str
    submitted_at: float
    finished_at: float | None = None
    result: IngestStats | None = None
    error: str | None = None


class QueryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=512)
    max_segments: int | None = Field(None, ge=1, le=100)
    kind: Literal["both", "crop", "frame"] | None = None
    verify: bool | None = None
    video_ids: list[str] | None = None


class LocateRequest(BaseModel):
    video_path: str = Field(description="Path on the server (video file or image-sequence dir).")
    sample_fps: float | None = Field(None, gt=0, le=30)
    vlm: bool | None = Field(None, description="Ask the VLM for a second opinion (geo.vlm).")
    include_frames: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    detector: str
    encoder: str
    videos: int
    vectors: int
    verifier_enabled: bool
    verifier_reachable: bool | None


def _check_allowed(path: Path, roots: list[Path]) -> None:
    if not roots:
        return
    resolved = path.resolve()
    if not any(resolved.is_relative_to(r.resolve()) for r in roots):
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"{path} is outside server.allowed_roots")


def create_app(
    settings: Settings | None = None,
    engine: VidQuery | None = None,
    locator: GeoLocator | None = None,
) -> FastAPI:
    settings = settings or (engine.settings if engine else load_config())
    state: dict[str, VidQuery] = {}
    geo: dict[str, GeoLocator] = {"locator": locator} if locator else {}
    geo_lock = threading.Lock()
    jobs: dict[str, JobStatus] = {}
    jobs_lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["engine"] = engine or VidQuery(settings)
        yield
        executor.shutdown(wait=False, cancel_futures=True)
        state["engine"].close()
        if "locator" in geo:
            geo["locator"].close()

    app = FastAPI(title="vidquery", version=__version__, lifespan=lifespan)

    def get_engine() -> VidQuery:
        return state["engine"]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        eng = get_engine()
        index = eng.index
        reachable = eng.verifier.health() if settings.verifier.enabled else None
        det, enc = settings.detector, settings.encoder
        return HealthResponse(
            status="ok",
            version=__version__,
            device=eng.device,
            detector=f"{det.model} [{det.backend}:{det.precision}]",
            encoder=f"{enc.model} [{enc.backend}:{enc.precision}]",
            videos=len(index.videos) if index else 0,
            vectors=len(index) if index else 0,
            verifier_enabled=settings.verifier.enabled,
            verifier_reachable=reachable,
        )

    @app.get("/videos", response_model=list[VideoEntry])
    def videos() -> list[VideoEntry]:
        index = get_engine().index
        return list(index.videos.values()) if index else []

    def _run_job(job_id: str, req: IngestRequest) -> None:
        with jobs_lock:
            jobs[job_id].state = "running"
        try:
            result = get_engine().ingest(req.video_path, req.video_id, req.sample_fps, req.replace)
            with jobs_lock:
                jobs[job_id].state = "done"
                jobs[job_id].result = result
        except Exception as exc:
            log.exception("Ingest job %s failed", job_id)
            with jobs_lock:
                jobs[job_id].state = "failed"
                jobs[job_id].error = str(exc)
        finally:
            with jobs_lock:
                jobs[job_id].finished_at = time.time()

    @app.post("/ingest", response_model=IngestStats | JobStatus)
    def ingest(req: IngestRequest) -> IngestStats | JobStatus:
        path = Path(req.video_path)
        _check_allowed(path, settings.server.allowed_roots)
        if not path.exists():
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"{path} does not exist")
        if req.background:
            job = JobStatus(
                job_id=uuid.uuid4().hex[:12],
                state="queued",
                video_path=str(path),
                submitted_at=time.time(),
            )
            with jobs_lock:
                jobs[job.job_id] = job
            executor.submit(_run_job, job.job_id, req)
            return job
        try:
            return get_engine().ingest(path, req.video_id, req.sample_fps, req.replace)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    @app.get("/jobs/{job_id}", response_model=JobStatus)
    def job_status(job_id: str) -> JobStatus:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown job {job_id}")
            return job.model_copy()

    @app.post("/query", response_model=QueryResult)
    def query(req: QueryRequest) -> QueryResult:
        return get_engine().query(req.text, req.max_segments, req.kind, req.verify, req.video_ids)

    def get_locator() -> GeoLocator:
        with geo_lock:  # GeoCLIP loads on the first /locate, not at startup
            if "locator" not in geo:
                geo["locator"] = GeoLocator(settings)
            return geo["locator"]

    def _locate(req: LocateRequest) -> LocateResult:
        path = Path(req.video_path)
        _check_allowed(path, settings.server.allowed_roots)
        if not path.exists():
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"{path} does not exist")
        try:
            return get_locator().locate(
                path, sample_fps=req.sample_fps, vlm=req.vlm, include_frames=req.include_frames
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    @app.post("/locate", response_model=LocateResult)
    def locate(req: LocateRequest) -> LocateResult:
        """Coarse location estimate for a video: a confidence area, never a precise pin."""
        return _locate(req)

    @app.post("/locate/map", response_class=HTMLResponse)
    def locate_map(req: LocateRequest) -> HTMLResponse:
        from vidquery.geo.mapview import render_map_html

        result = _locate(req)
        return HTMLResponse(
            render_map_html(
                result,
                tiles_url=settings.geo.map_tiles_url,
                attribution=settings.geo.map_attribution,
            )
        )

    return app


def serve(settings: Settings, host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    uvicorn.run(
        create_app(settings),
        host=host or settings.server.host,
        port=port or settings.server.port,
        log_level="info",
    )
