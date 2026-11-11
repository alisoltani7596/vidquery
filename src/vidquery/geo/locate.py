"""``GeoLocator``: where was this video filmed? Coarse by design.

Frames are sampled at ``geo.sample_fps`` (default 1), frames mostly covered by people are skipped,
the rest go through the geo backend (GeoCLIP), and :func:`vidquery.geo.aggregate.aggregate` turns
the per-frame top-k into one estimate with a confidence radius and up to three alternatives.

Every reported position is rounded to the precision its radius supports
(:func:`vidquery.geo.geometry.coarsen`): the output is an area, never a precise pin.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from vidquery.config import Settings, resolve_device
from vidquery.geo.aggregate import AggregateParams, Location, aggregate
from vidquery.geo.backends import FramePrediction, GeoBackend, build_geo_backend
from vidquery.geo.geometry import coarsen, coordinate_decimals
from vidquery.geo.people import person_coverage
from vidquery.geo.vlm import VLMGeolocator, VLMOpinion
from vidquery.models.detector import PersonDetector
from vidquery.types import SampledFrame
from vidquery.video import open_video

log = logging.getLogger(__name__)

RESPONSIBLE_USE = (
    "Intended for public scenes and landmarks. The output is a confidence area, not a precise "
    "pin, and it is not intended for locating individuals."
)


class GeoArea(BaseModel):
    lat: float
    lon: float
    radius_km: float = Field(description="Radius of the confidence circle in km.")
    share: float = Field(description="Share of the probability mass in this cluster.")


class FrameGeo(BaseModel):
    frame_idx: int
    timestamp: float
    person_fraction: float | None = None
    skipped: bool = False
    top1: tuple[float, float] | None = Field(None, description="Rounded like the estimate.")
    top1_prob: float | None = None


class LocateResult(BaseModel):
    video: str
    backend: str
    sample_fps: float
    frames_sampled: int
    frames_used: int
    frames_skipped_people: int
    estimate: GeoArea | None
    alternatives: list[GeoArea] = Field(default_factory=list)
    coverage_mass: float = Field(description="Share of the probability mass inside radius_km.")
    vlm: VLMOpinion | None = None
    frames: list[FrameGeo] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    elapsed_s: float = 0.0
    notice: str = ""


def to_area(loc: Location) -> GeoArea:
    lat, lon, radius = coarsen(loc.lat, loc.lon, loc.radius_km)
    return GeoArea(lat=lat, lon=lon, radius_km=radius, share=round(loc.share, 3))


def notice_text(mass: float, estimate: GeoArea | None) -> str:
    if estimate is None:
        return "No usable frames (all were skipped or none could be decoded). " + RESPONSIBLE_USE
    return (
        f"Coarse estimate: the circle of radius_km around the point holds about {mass:.0%} of the "
        "model's probability mass, and coordinates are rounded to match. " + RESPONSIBLE_USE
    )


def _shrink(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image.copy()
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


class GeoLocator:
    """Builds its backend, person detector and VLM client lazily; all three can be injected."""

    def __init__(
        self,
        settings: Settings,
        backend: GeoBackend | None = None,
        detector: PersonDetector | None = None,
        vlm: VLMGeolocator | None = None,
    ) -> None:
        self.settings = settings
        self.cfg = settings.geo
        self.device = resolve_device(self.cfg.device or settings.device)
        self._backend = backend
        self._detector = detector
        self._vlm = vlm
        self._lock = threading.Lock()

    @property
    def backend(self) -> GeoBackend:
        if self._backend is None:
            self._backend = build_geo_backend(self.cfg, self.device)
        return self._backend

    @property
    def detector(self) -> PersonDetector:
        if self._detector is None:
            from vidquery.models.registry import build_detector

            self._detector = build_detector(self.settings, self.device)
        return self._detector

    @property
    def vlm(self) -> VLMGeolocator:
        if self._vlm is None:
            self._vlm = VLMGeolocator(self.settings.verifier, self.cfg.vlm)
        return self._vlm

    def close(self) -> None:
        if self._vlm is not None:
            self._vlm.close()

    def locate(
        self,
        video: str | Path,
        *,
        sample_fps: float | None = None,
        vlm: bool | None = None,
        include_frames: bool = False,
        params: AggregateParams | None = None,
    ) -> LocateResult:
        with self._lock:  # one video at a time on the shared models
            return self._locate(Path(video), sample_fps, vlm, include_frames, params)

    def _locate(
        self,
        path: Path,
        sample_fps: float | None,
        vlm: bool | None,
        include_frames: bool,
        params: AggregateParams | None,
    ) -> LocateResult:
        t0 = time.perf_counter()
        cfg = self.cfg
        params = params or AggregateParams.from_config(cfg)
        use_vlm = cfg.vlm.enabled if vlm is None else vlm
        preds: list[FramePrediction] = []
        rows: list[FrameGeo] = []
        candidates: list[tuple[float, int, float, np.ndarray]] = []

        def process(batch: list[SampledFrame]) -> None:
            coverage: list[float | None] = [None] * len(batch)
            if cfg.skip_person_frames:
                dets = self.detector.detect([f.image for f in batch])
                coverage = [
                    person_coverage(d, f.image.shape[0], f.image.shape[1])
                    for d, f in zip(dets, batch, strict=True)
                ]
            keep = [i for i, c in enumerate(coverage) if c is None or c <= cfg.max_person_fraction]
            out = self.backend.predict([batch[i].image for i in keep], params.top_k)
            by_frame = dict(zip(keep, out, strict=True))
            decimals = coordinate_decimals(params.min_radius_km)
            for i, frame in enumerate(batch):
                pred = by_frame.get(i)
                cov = coverage[i]
                rows.append(
                    FrameGeo(
                        frame_idx=frame.frame_idx,
                        timestamp=round(frame.timestamp, 3),
                        person_fraction=None if cov is None else round(cov, 3),
                        skipped=pred is None,
                        top1=None
                        if pred is None
                        else (
                            round(float(pred.lats[0]), decimals),
                            round(float(pred.lons[0]), decimals),
                        ),
                        top1_prob=None if pred is None else round(pred.top1_prob, 4),
                    )
                )
                if pred is None:
                    continue
                preds.append(pred)
                if use_vlm:
                    small = _shrink(frame.image, cfg.vlm.max_image_side)
                    candidates.append((pred.top_mass, frame.frame_idx, frame.timestamp, small))
                    candidates.sort(key=lambda c: -c[0])
                    del candidates[cfg.vlm.frames :]

        with open_video(path) as src:
            fps = sample_fps or cfg.sample_fps
            duration = src.info.duration
            if duration > 0 and duration * fps > cfg.max_frames:
                fps = cfg.max_frames / duration
            batch: list[SampledFrame] = []
            sampled = 0
            for frame in src.iter_frames(fps):
                if sampled >= cfg.max_frames:
                    break
                sampled += 1
                batch.append(frame)
                if len(batch) >= cfg.batch_size:
                    process(batch)
                    batch = []
            if batch:
                process(batch)

        agg = aggregate(preds, params)
        estimate = to_area(agg.best) if agg else None
        alternatives = [to_area(a) for a in agg.alternatives] if agg else []
        opinion = None
        if use_vlm:
            frames = [(idx, t, img) for _, idx, t, img in candidates]
            opinion = self.vlm.opinion(frames)
        result = LocateResult(
            video=str(path),
            backend=self.backend.name,
            sample_fps=round(fps, 4),
            frames_sampled=sampled,
            frames_used=len(preds),
            frames_skipped_people=sampled - len(preds),
            estimate=estimate,
            alternatives=alternatives,
            coverage_mass=params.confidence_mass,
            vlm=opinion,
            frames=rows if include_frames else None,
            params=params.to_dict(),
            elapsed_s=round(time.perf_counter() - t0, 3),
            notice=notice_text(params.confidence_mass, estimate),
        )
        log.info(
            "%s: %d/%d frames used -> %s",
            path.name,
            result.frames_used,
            sampled,
            "no estimate"
            if estimate is None
            else f"({estimate.lat}, {estimate.lon}) +/- {estimate.radius_km} km",
        )
        return result
