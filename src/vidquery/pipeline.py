"""The search engine: ingest videos into the index, answer text queries."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from vidquery.config import Settings, resolve_device
from vidquery.index import VectorIndex
from vidquery.models.detector import PersonDetector
from vidquery.models.encoder import ImageEncoder, TextEncoder, crop_with_margin
from vidquery.segments import merge_hits
from vidquery.types import Detection, Hit, Record, SampledFrame, Segment, VideoEntry
from vidquery.verify import VLMVerifier
from vidquery.video import VideoSource, open_video, prefetch

log = logging.getLogger(__name__)


class IngestStats(BaseModel):
    video_id: str
    path: str
    duration_s: float
    sampled_frames: int
    crops: int
    vectors: int
    elapsed_s: float
    ingest_fps: float  # sampled frames processed per wall-clock second
    realtime_factor: float  # seconds of video indexed per wall-clock second
    stage_s: dict[str, float]


class QueryResult(BaseModel):
    query: str
    segments: list[Segment]
    verified: bool
    timings_ms: dict[str, float]


@dataclass
class _Timer:
    totals: dict[str, float] = field(default_factory=dict)

    def add(self, stage: str, seconds: float) -> None:
        self.totals[stage] = self.totals.get(stage, 0.0) + seconds


def filter_hits(hits: list[Hit], margin: float | None, min_score: float | None) -> list[Hit]:
    """Keep hits close to the best one (and above an absolute floor)."""
    if not hits:
        return hits
    best = max(h.score for h in hits)
    return [
        h
        for h in hits
        if (margin is None or h.score >= best - margin)
        and (min_score is None or h.score >= min_score)
    ]


class VidQuery:
    """Owns the models (built lazily) and the index. Thread-safe for API use."""

    def __init__(
        self,
        settings: Settings,
        *,
        detector: PersonDetector | None = None,
        image_encoder: ImageEncoder | None = None,
        text_encoder: TextEncoder | None = None,
        verifier: VLMVerifier | None = None,
        index: VectorIndex | None = None,
    ) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        self._detector = detector
        self._image_encoder = image_encoder
        self._text_encoder = text_encoder
        self._verifier = verifier
        self._index = index
        self._lock = threading.RLock()
        self._sources: OrderedDict[str, VideoSource] = OrderedDict()

    # ------------------------------------------------------------------ lazy components
    @property
    def detector(self) -> PersonDetector:
        if self._detector is None:
            from vidquery.models.registry import build_detector

            self._detector = build_detector(self.settings, self.device)
        return self._detector

    @property
    def image_encoder(self) -> ImageEncoder:
        if self._image_encoder is None:
            from vidquery.models.registry import build_image_encoder

            self._image_encoder = build_image_encoder(self.settings, self.device)
        return self._image_encoder

    @property
    def text_encoder(self) -> TextEncoder:
        if self._text_encoder is None:
            from vidquery.models.registry import build_text_encoder

            self._text_encoder = build_text_encoder(self.settings, self.device)
        return self._text_encoder

    @property
    def verifier(self) -> VLMVerifier:
        if self._verifier is None:
            self._verifier = VLMVerifier(self.settings.verifier)
        return self._verifier

    @property
    def index(self) -> VectorIndex | None:
        if self._index is None and VectorIndex.exists(self.settings.index_dir):
            self._index = VectorIndex.load(self.settings.index_dir)
            log.info(
                "Loaded index with %d vectors from %s", len(self._index), self.settings.index_dir
            )
        return self._index

    def _ensure_index(self, dim: int) -> VectorIndex:
        index = self.index
        if index is None:
            index = VectorIndex(dim, self.settings.faiss_factory, self.settings.encoder.model)
            self._index = index
        if index.dim != dim:
            raise ValueError(f"Encoder dim {dim} does not match index dim {index.dim}")
        if index.encoder_name and index.encoder_name != self.settings.encoder.model:
            raise ValueError(
                f"Index was built with encoder {index.encoder_name!r}, settings use "
                f"{self.settings.encoder.model!r}; use a different index_dir"
            )
        return index

    def save(self) -> None:
        with self._lock:
            if self._index is not None:
                self._index.save(self.settings.index_dir)

    # ------------------------------------------------------------------ ingest
    def ingest(
        self,
        video_path: str | Path,
        video_id: str | None = None,
        sample_fps: float | None = None,
        replace: bool = True,
        save: bool = True,
        progress: Callable[[int], None] | None = None,
    ) -> IngestStats:
        cfg = self.settings.ingest
        sample_fps = sample_fps or cfg.sample_fps
        path = Path(video_path)
        video_id = video_id or path.stem
        if not (cfg.embed_frames or cfg.embed_crops):
            raise ValueError("At least one of ingest.embed_frames / ingest.embed_crops must be set")
        source = open_video(path)
        timer = _Timer()
        t_start = time.perf_counter()

        pending_images: list[np.ndarray] = []
        pending_meta: list[tuple[SampledFrame, str, Detection | None]] = []
        vectors: list[np.ndarray] = []
        metas: list[tuple[SampledFrame, str, Detection | None]] = []
        n_frames = n_crops = 0

        def flush(force: bool = False) -> None:
            while pending_images and (force or len(pending_images) >= cfg.encoder_batch_size):
                take = min(len(pending_images), cfg.encoder_batch_size)
                t0 = time.perf_counter()
                emb = self.image_encoder.encode(pending_images[:take], cfg.encoder_batch_size)
                timer.add("embed", time.perf_counter() - t0)
                vectors.append(emb)
                metas.extend(pending_meta[:take])
                del pending_images[:take]
                del pending_meta[:take]

        def handle_batch(batch: list[SampledFrame]) -> None:
            nonlocal n_frames, n_crops
            dets: list[list[Detection]] = [[] for _ in batch]
            if cfg.embed_crops:
                t0 = time.perf_counter()
                dets = self.detector.detect([f.image for f in batch])
                timer.add("detect", time.perf_counter() - t0)
            t0 = time.perf_counter()
            for frame, frame_dets in zip(batch, dets, strict=True):
                n_frames += 1
                if cfg.embed_frames:
                    pending_images.append(frame.image)
                    pending_meta.append((frame, "frame", None))
                kept = [d for d in frame_dets if min(d.width, d.height) >= cfg.min_box_size][
                    : cfg.max_crops_per_frame
                ]
                for det in kept:
                    try:
                        crop = crop_with_margin(frame.image, det.box, cfg.crop_margin)
                    except ValueError:
                        continue
                    pending_images.append(crop)
                    pending_meta.append((frame, "crop", det))
                    n_crops += 1
            timer.add("crop", time.perf_counter() - t0)
            flush()
            if progress is not None:
                progress(n_frames)

        batch: list[SampledFrame] = []
        t_dec = time.perf_counter()
        for frame in prefetch(source.iter_frames(sample_fps), cfg.prefetch_frames):
            timer.add("decode_wait", time.perf_counter() - t_dec)
            batch.append(frame)
            if len(batch) >= cfg.detector_batch_size:
                handle_batch(batch)
                batch = []
            t_dec = time.perf_counter()
        if batch:
            handle_batch(batch)
        flush(force=True)
        source.close()

        if not vectors:
            raise ValueError(f"No frames decoded from {path}")
        all_vecs = np.concatenate(vectors, axis=0)
        t0 = time.perf_counter()
        with self._lock:
            index = self._ensure_index(all_vecs.shape[1])
            if video_id in index.videos:
                if not replace:
                    raise ValueError(f"Video {video_id!r} is already indexed")
                index.remove_video(video_id)
            ids = index.allocate_ids(len(metas))
            records = [
                Record(
                    id=rid,
                    video_id=video_id,
                    frame_idx=frame.frame_idx,
                    timestamp=frame.timestamp,
                    kind=kind,  # type: ignore[arg-type]
                    box=det.box if det is not None else None,
                    det_score=det.score if det is not None else None,
                )
                for rid, (frame, kind, det) in zip(ids, metas, strict=True)
            ]
            index.add(all_vecs, records)
            info = source.info
            index.register_video(
                VideoEntry(
                    video_id=video_id,
                    path=str(path.resolve()),
                    fps=info.fps,
                    num_frames=info.num_frames,
                    width=info.width,
                    height=info.height,
                    duration=info.duration,
                    sample_fps=sample_fps,
                )
            )
            self._sources.pop(video_id, None)
            if save:
                index.save(self.settings.index_dir)
        timer.add("index", time.perf_counter() - t0)

        elapsed = time.perf_counter() - t_start
        return IngestStats(
            video_id=video_id,
            path=str(path),
            duration_s=source.info.duration,
            sampled_frames=n_frames,
            crops=n_crops,
            vectors=len(records),
            elapsed_s=elapsed,
            ingest_fps=n_frames / elapsed if elapsed > 0 else 0.0,
            realtime_factor=source.info.duration / elapsed if elapsed > 0 else 0.0,
            stage_s={k: round(v, 4) for k, v in timer.totals.items()},
        )

    # ------------------------------------------------------------------ query
    def query(
        self,
        text: str,
        max_segments: int | None = None,
        kind: str | None = None,
        verify: bool | None = None,
        video_ids: list[str] | None = None,
    ) -> QueryResult:
        qcfg = self.settings.query
        max_segments = max_segments or qcfg.max_segments
        verify = self.settings.verifier.enabled if verify is None else verify
        timings: dict[str, float] = {}
        index = self.index
        if index is None or len(index) == 0:
            return QueryResult(query=text, segments=[], verified=False, timings_ms={})

        t0 = time.perf_counter()
        qvec = self.text_encoder.encode([text])[0]
        timings["encode_text"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with self._lock:
            hits = index.search(
                qvec, qcfg.top_k, kind or qcfg.kind, video_ids, qcfg.frame_score_weight
            )
            sample_fps = {v: e.sample_fps for v, e in index.videos.items()}
        hits = filter_hits(hits, qcfg.score_margin, qcfg.min_score)
        timings["search"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        pad = {v: 0.5 / fps for v, fps in sample_fps.items()}
        segments = merge_hits(hits, qcfg.merge_gap_s, pad, qcfg.min_hits)
        timings["merge"] = (time.perf_counter() - t0) * 1000

        verified = False
        if verify and segments:
            t0 = time.perf_counter()
            segments = self.verifier.rerank(text, segments, self.load_hit_image)
            verified = True
            timings["verify"] = (time.perf_counter() - t0) * 1000
        return QueryResult(
            query=text, segments=segments[:max_segments], verified=verified, timings_ms=timings
        )

    # ------------------------------------------------------------------ frames
    def source_for(self, video_id: str) -> VideoSource:
        with self._lock:
            if video_id in self._sources:
                self._sources.move_to_end(video_id)
                return self._sources[video_id]
            index = self.index
            if index is None or video_id not in index.videos:
                raise KeyError(f"Unknown video {video_id!r}")
            entry = index.videos[video_id]
            src = open_video(entry.path, entry.fps)
            self._sources[video_id] = src
            while len(self._sources) > 8:
                _, old = self._sources.popitem(last=False)
                old.close()
            return src

    def load_hit_image(self, hit: Hit) -> np.ndarray:
        """Re-decode the frame of a hit (and crop it for crop hits)."""
        with self._lock:
            frame = self.source_for(hit.video_id).read_frame(hit.frame_idx)
        if hit.box is None:
            return frame
        return crop_with_margin(frame, hit.box, self.settings.verifier.crop_margin)

    def close(self) -> None:
        for src in self._sources.values():
            src.close()
        self._sources.clear()
        if self._verifier is not None:
            self._verifier.close()
