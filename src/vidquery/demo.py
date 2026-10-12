"""Render an annotated clip (mp4) or GIF for a query's top segments."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vidquery.pipeline import QueryResult, VidQuery
from vidquery.types import Hit, Segment
from vidquery.video import write_video

log = logging.getLogger(__name__)

BOX_COLOR = (0, 220, 255)  # BGR amber
TEXT_BG = (20, 20, 20)


def _boxes_near(segment: Segment, t: float, window: float) -> list[Hit]:
    """Crop hits from the sampled timestamp closest to ``t`` (within ``window`` seconds)."""
    boxed = segment.boxes()
    if not boxed:
        return []
    nearest = min(boxed, key=lambda h: abs(h.timestamp - t))
    if abs(nearest.timestamp - t) > window:
        return []
    return [h for h in boxed if h.timestamp == nearest.timestamp]


def _label(frame: np.ndarray, text: str, org: tuple[int, int], scale: float) -> None:
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(frame, (x, y - h - base - 2), (x + w + 4, y + 2), TEXT_BG, -1)
    cv2.putText(
        frame,
        text,
        (x + 2, y - base),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def annotate(frame: np.ndarray, hits: list[Hit], caption: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.4, min(w, h) / 900)
    thickness = max(1, round(min(w, h) / 300))
    for hit in hits:
        assert hit.box is not None
        x1, y1, x2, y2 = (round(v) for v in hit.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, thickness)
        _label(out, f"{hit.score:.3f}", (x1, max(12, y1)), scale)
    _label(out, caption, (4, int(22 * scale / 0.5)), scale)
    return out


def iter_annotated_frames(
    engine: VidQuery,
    result: QueryResult,
    num_segments: int = 1,
    context_s: float = 0.5,
    max_width: int = 960,
) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (annotated frame, fps) for each of the top segments, in rank order."""
    for rank, seg in enumerate(result.segments[:num_segments], start=1):
        src = engine.source_for(seg.video_id)
        fps = src.info.fps
        entry = engine.index.videos[seg.video_id] if engine.index else None
        window = 1.0 / entry.sample_fps if entry else 0.5
        caption = f'#{rank} "{result.query}"  {seg.video_id} {seg.start:.1f}-{seg.end:.1f}s'
        if seg.verifier_score is not None:
            caption += f"  vlm={seg.verifier_score:.2f}"
        start = max(0.0, seg.start - context_s)
        end = min(src.info.duration, seg.end + context_s)
        for frame in src.iter_range(start, end):
            img = annotate(frame.image, _boxes_near(seg, frame.timestamp, window), caption)
            if img.shape[1] > max_width:
                s = max_width / img.shape[1]
                img = cv2.resize(
                    img, (max_width, round(img.shape[0] * s)), interpolation=cv2.INTER_AREA
                )
            yield img, fps


def render_demo(
    engine: VidQuery,
    query: str,
    out_path: str | Path,
    num_segments: int = 1,
    context_s: float = 0.5,
    max_width: int | None = None,
    gif_fps: float = 8.0,
    verify: bool | None = None,
) -> tuple[Path, QueryResult]:
    """Run ``query`` and write the top segments with boxes to ``out_path`` (.mp4/.avi/.gif)."""
    out = Path(out_path)
    result = engine.query(query, max_segments=max(num_segments, 1), verify=verify)
    if not result.segments:
        raise ValueError(f"No results for {query!r}")
    is_gif = out.suffix.lower() == ".gif"
    width = max_width or (480 if is_gif else 960)
    frames = iter_annotated_frames(engine, result, num_segments, context_s, width)
    out.parent.mkdir(parents=True, exist_ok=True)
    if is_gif:
        images: list[Image.Image] = []
        step_acc = 0.0
        for img, fps in frames:
            step_acc += gif_fps / fps
            if step_acc >= 1.0 or not images:
                step_acc = max(0.0, step_acc - 1.0)
                images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if not images:
            raise ValueError("No frames rendered")
        images[0].save(
            out,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / gif_fps),
            loop=0,
            optimize=True,
        )
    else:
        first = next(frames, None)
        if first is None:
            raise ValueError("No frames rendered")
        fps = first[1]

        h0, w0 = first[0].shape[:2]

        def all_frames() -> Iterator[np.ndarray]:
            yield first[0]
            for img, _ in frames:  # segments from other videos may differ in size
                yield img if img.shape[:2] == (h0, w0) else cv2.resize(img, (w0, h0))

        write_video(all_frames(), out, fps)
    log.info("Wrote %s", out)
    return out, result
