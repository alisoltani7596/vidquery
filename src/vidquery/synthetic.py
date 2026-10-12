"""Synthetic test video: coloured "people" walking across a grey background.

Used by the test-suite, the weight-free demo (``configs/mock.yaml``) and CI. The ground truth is
known exactly, so the full pipeline (detect -> crop -> embed -> index -> query -> merge -> eval)
can be checked end to end without any model weights.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vidquery.video import write_video

BACKGROUND = 128


@dataclass(frozen=True)
class Actor:
    color_name: str
    bgr: tuple[int, int, int]
    start: float  # seconds on screen
    end: float
    y_frac: float  # vertical position (fraction of height)
    direction: int  # +1 walks right, -1 walks left


SCRIPT: tuple[Actor, ...] = (
    Actor("red", (0, 0, 230), 0.0, 4.0, 0.15, +1),
    Actor("green", (0, 200, 0), 2.0, 6.0, 0.55, -1),
    Actor("blue", (230, 0, 0), 5.0, 9.0, 0.15, -1),
    Actor("yellow", (0, 230, 230), 8.5, 11.5, 0.55, +1),
)
DURATION_S = 12.0


def actor_box(actor: Actor, t: float, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Ground-truth box (x1, y1, x2, y2) of ``actor`` at time ``t``, or None if off screen."""
    if not actor.start <= t < actor.end:
        return None
    h, w = size
    bw, bh = max(8, w // 14), max(20, h // 4)
    progress = (t - actor.start) / (actor.end - actor.start)
    if actor.direction < 0:
        progress = 1.0 - progress
    x1 = int(4 + progress * (w - bw - 8))
    y1 = int(actor.y_frac * h)
    return x1, y1, x1 + bw, y1 + bh


def synthetic_frames(
    fps: float = 10.0, size: tuple[int, int] = (240, 320), duration: float = DURATION_S
) -> Iterator[np.ndarray]:
    for i in range(round(duration * fps)):
        t = i / fps
        frame = np.full((*size, 3), BACKGROUND, np.uint8)
        for actor in SCRIPT:
            box = actor_box(actor, t, size)
            if box is not None:
                cv2.rectangle(frame, box[:2], box[2:], actor.bgr, -1)
        yield frame


def make_synthetic_video(
    path: str | Path,
    fps: float = 10.0,
    size: tuple[int, int] = (240, 320),
    duration: float = DURATION_S,
) -> Path:
    path = Path(path)
    write_video(synthetic_frames(fps, size, duration), path, fps)
    return path
