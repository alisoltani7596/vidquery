from __future__ import annotations

import pytest

from vidquery.segments import merge_hits, temporal_iou, temporal_overlap
from vidquery.types import Hit


def hit(t: float, score: float, video: str = "v", box: bool = True, id_: int = 0) -> Hit:
    return Hit(
        id=id_,
        video_id=video,
        frame_idx=int(t * 10),
        timestamp=t,
        kind="crop" if box else "frame",
        score=score,
        box=[0, 0, 10, 10] if box else None,
    )


def test_merges_within_gap_and_splits_beyond() -> None:
    hits = [hit(0.0, 0.2), hit(0.5, 0.3), hit(1.0, 0.25), hit(5.0, 0.9), hit(5.5, 0.1)]
    segs = merge_hits(hits, gap_s=1.0, pad_s=0.25)
    assert len(segs) == 2
    assert segs[0].start == pytest.approx(4.75) and segs[0].end == pytest.approx(5.75)
    assert segs[0].score == pytest.approx(0.9) and segs[0].num_hits == 2
    assert segs[1].start == 0.0  # clipped at zero
    assert segs[1].end == pytest.approx(1.25)
    assert segs[1].num_hits == 3


def test_gap_is_measured_from_previous_hit_not_segment_start() -> None:
    hits = [hit(t, 0.5) for t in (0.0, 1.0, 2.0, 3.0, 4.0)]
    segs = merge_hits(hits, gap_s=1.0)
    assert len(segs) == 1 and segs[0].end == pytest.approx(4.0)


def test_videos_never_merge_and_pad_per_video() -> None:
    hits = [hit(1.0, 0.5, "a"), hit(1.0, 0.6, "b"), hit(1.2, 0.4, "a")]
    segs = merge_hits(hits, gap_s=1.0, pad_s={"a": 0.1, "b": 0.5})
    assert [s.video_id for s in segs] == ["b", "a"]
    assert segs[0].start == pytest.approx(0.5)
    assert segs[1].start == pytest.approx(0.9) and segs[1].end == pytest.approx(1.3)


def test_same_timestamp_multiple_boxes_and_min_hits() -> None:
    hits = [hit(2.0, 0.5, id_=1), hit(2.0, 0.7, id_=2), hit(9.0, 0.95, box=False, id_=3)]
    segs = merge_hits(hits, gap_s=0.5, min_hits=2)
    assert len(segs) == 1
    seg = segs[0]
    assert seg.best_hit.id == 2
    assert [h.id for h in seg.boxes()] == [2, 1]


def test_empty_and_temporal_metrics() -> None:
    assert merge_hits([], gap_s=1.0) == []
    assert temporal_overlap((0, 2), (1, 3)) == 1
    assert temporal_iou((0, 2), (1, 3)) == pytest.approx(1 / 3)
    assert temporal_iou((0, 1), (2, 3)) == 0.0
    assert temporal_iou((0, 0), (0, 0)) == 0.0
