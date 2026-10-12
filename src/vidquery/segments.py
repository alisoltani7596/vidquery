"""Merge per-frame hits into time segments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from vidquery.types import Hit, Segment


def merge_hits(
    hits: Iterable[Hit],
    gap_s: float,
    pad_s: float | Mapping[str, float] = 0.0,
    min_hits: int = 1,
) -> list[Segment]:
    """Group hits per video into segments, best segment first.

    Hits closer than ``gap_s`` seconds (from the previous hit in time) join the same segment.
    Each segment is padded by ``pad_s`` on both sides (typically half the sampling period, so a
    single sampled frame covers the time it represents) and clipped at zero. ``pad_s`` may be a
    per-video mapping when videos were sampled at different rates.
    """
    by_video: dict[str, list[Hit]] = defaultdict(list)
    for hit in hits:
        by_video[hit.video_id].append(hit)

    segments: list[Segment] = []
    for video_id, vhits in by_video.items():
        pad = pad_s.get(video_id, 0.0) if isinstance(pad_s, Mapping) else pad_s
        vhits.sort(key=lambda h: (h.timestamp, -h.score))
        group: list[Hit] = [vhits[0]]
        for hit in vhits[1:]:
            if hit.timestamp - group[-1].timestamp <= gap_s:
                group.append(hit)
            else:
                segments.append(_make_segment(video_id, group, pad))
                group = [hit]
        segments.append(_make_segment(video_id, group, pad))

    segments = [s for s in segments if s.num_hits >= min_hits]
    segments.sort(key=lambda s: (-s.score, s.video_id, s.start))
    return segments


def _make_segment(video_id: str, hits: list[Hit], pad_s: float) -> Segment:
    best = max(h.score for h in hits)
    return Segment(
        video_id=video_id,
        start=max(0.0, hits[0].timestamp - pad_s),
        end=hits[-1].timestamp + pad_s,
        score=best,
        retrieval_score=best,
        num_hits=len(hits),
        hits=sorted(hits, key=lambda h: (h.timestamp, -h.score)),
    )


def temporal_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = temporal_overlap(a, b)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0
