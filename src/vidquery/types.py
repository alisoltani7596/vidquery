"""Core data types shared by the pipeline, index, API and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

RecordKind = Literal["frame", "crop"]
BoxTuple = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    """A person box in original frame pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def box(self) -> BoxTuple:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(slots=True)
class SampledFrame:
    frame_idx: int
    timestamp: float
    image: np.ndarray  # HxWx3 uint8, BGR


@dataclass(frozen=True, slots=True)
class Record:
    """Metadata stored next to every vector in the index."""

    id: int
    video_id: str
    frame_idx: int
    timestamp: float
    kind: RecordKind
    box: BoxTuple | None = None
    det_score: float | None = None


class Hit(BaseModel):
    """A single nearest-neighbour result."""

    id: int
    video_id: str
    frame_idx: int
    timestamp: float
    kind: RecordKind
    score: float
    box: list[float] | None = None
    det_score: float | None = None

    @classmethod
    def from_record(cls, record: Record, score: float) -> Hit:
        return cls(
            id=record.id,
            video_id=record.video_id,
            frame_idx=record.frame_idx,
            timestamp=record.timestamp,
            kind=record.kind,
            score=score,
            box=list(record.box) if record.box is not None else None,
            det_score=record.det_score,
        )


class Segment(BaseModel):
    """A contiguous time range in one video where the query matched."""

    video_id: str
    start: float
    end: float
    score: float = Field(description="Ranking score (fused with the VLM score when verified).")
    retrieval_score: float = Field(description="Max embedding similarity inside the segment.")
    verifier_score: float | None = Field(
        default=None, description="VLM p(yes) for the segment's best hit, if verified."
    )
    num_hits: int
    hits: list[Hit] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def best_hit(self) -> Hit:
        return max(self.hits, key=lambda h: h.score)

    def boxes(self) -> list[Hit]:
        """Hits that carry a person box, ordered by time."""
        return sorted((h for h in self.hits if h.box is not None), key=lambda h: h.timestamp)


class VideoEntry(BaseModel):
    """Registry entry for an ingested video (needed to re-read frames later)."""

    video_id: str
    path: str
    fps: float
    num_frames: int
    width: int
    height: int
    duration: float
    sample_fps: float
