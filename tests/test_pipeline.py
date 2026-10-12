"""End-to-end: synthetic video -> detect -> crop -> embed -> FAISS -> query -> segments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vidquery.config import Settings
from vidquery.export.onnx_export import export_component
from vidquery.models.detector import box_iou
from vidquery.pipeline import VidQuery, filter_hits
from vidquery.synthetic import SCRIPT, actor_box
from vidquery.types import Hit

SIZE = (240, 320)


def _actor(name: str):  # type: ignore[no-untyped-def]
    return next(a for a in SCRIPT if a.color_name == name)


def test_ingest_stats(mock_settings: Settings, synthetic_video: Path) -> None:
    engine = VidQuery(mock_settings)
    stats = engine.ingest(synthetic_video)
    assert stats.video_id == "synthetic"
    assert stats.sampled_frames == 24  # 12 s at 2 fps
    # One crop per visible actor per sampled frame.
    expected_crops = sum(actor_box(a, i / 2.0, SIZE) is not None for i in range(24) for a in SCRIPT)
    assert stats.crops == expected_crops
    assert stats.vectors == stats.sampled_frames + stats.crops
    assert stats.ingest_fps > 0 and set(stats.stage_s) >= {"detect", "embed", "index"}
    assert (mock_settings.index_dir / "info.json").exists()


@pytest.mark.parametrize("color", ["red", "green", "blue", "yellow"])
def test_query_finds_the_right_time_and_boxes(
    mock_settings: Settings, synthetic_video: Path, color: str
) -> None:
    engine = VidQuery(mock_settings)
    engine.ingest(synthetic_video)
    result = engine.query(f"a person wearing {color}")
    actor = _actor(color)
    top = result.segments[0]
    assert top.video_id == "synthetic"
    # The segment covers the actor's screen time (to within one sampling period).
    assert top.start <= actor.start + 0.5 and top.end >= actor.end - 0.75
    assert top.start >= actor.start - 0.5 and top.end <= actor.end + 0.5
    boxed = top.boxes()
    assert boxed, "crop hits must carry boxes"
    for h in boxed:
        gt = actor_box(actor, h.timestamp, SIZE)
        assert gt is not None
        assert box_iou(np.array([gt]), np.array([h.box]))[0, 0] > 0.7


def test_kind_filter_and_persistence(mock_settings: Settings, synthetic_video: Path) -> None:
    VidQuery(mock_settings).ingest(synthetic_video, video_id="clip")
    engine = VidQuery(mock_settings)  # fresh engine loads the saved index
    crops = engine.query("red", kind="crop")
    frames = engine.query("red", kind="frame")
    assert all(h.kind == "crop" for s in crops.segments for h in s.hits)
    assert all(h.kind == "frame" for s in frames.segments for h in s.hits)
    assert crops.segments[0].video_id == "clip"


def test_reingest_replaces_and_no_replace_raises(
    mock_settings: Settings, synthetic_video: Path
) -> None:
    engine = VidQuery(mock_settings)
    first = engine.ingest(synthetic_video)
    engine.ingest(synthetic_video)
    assert engine.index is not None and len(engine.index) == first.vectors
    with pytest.raises(ValueError, match="already indexed"):
        engine.ingest(synthetic_video, replace=False)


def test_empty_index_returns_nothing(mock_settings: Settings) -> None:
    result = VidQuery(mock_settings).query("anything")
    assert result.segments == [] and not result.verified


def test_load_hit_image_crops(mock_settings: Settings, synthetic_video: Path) -> None:
    engine = VidQuery(mock_settings)
    engine.ingest(synthetic_video)
    seg = engine.query("red").segments[0]
    hit = seg.boxes()[0]
    img = engine.load_hit_image(hit)
    assert img.shape[0] < SIZE[0] and img.shape[1] < SIZE[1]
    frame_hit = hit.model_copy(update={"box": None, "kind": "frame"})
    assert engine.load_hit_image(frame_hit).shape[:2] == SIZE


def test_filter_hits() -> None:
    hits = [
        Hit(id=i, video_id="v", frame_idx=i, timestamp=i, kind="frame", score=s)
        for i, s in enumerate([0.30, 0.28, 0.2, 0.1])
    ]
    assert [h.id for h in filter_hits(hits, 0.05, None)] == [0, 1]
    assert [h.id for h in filter_hits(hits, None, 0.15)] == [0, 1, 2]
    assert filter_hits([], 0.1, 0.1) == []


def test_onnxruntime_backend_gives_same_results(
    mock_settings: Settings, synthetic_video: Path
) -> None:
    """The whole pipeline on ONNX Runtime matches the PyTorch pipeline."""
    torch_engine = VidQuery(mock_settings)
    torch_engine.ingest(synthetic_video)
    for comp in ("image_encoder", "text_encoder"):
        export_component(mock_settings, comp, exporter="torchscript")
    ort_settings = mock_settings.model_copy(deep=True)
    ort_settings.index_dir = mock_settings.index_dir.parent / "index-ort"
    ort_settings.encoder.backend = "onnxruntime"
    ort_settings.encoder.text_backend = "onnxruntime"
    ort_engine = VidQuery(ort_settings)
    ort_engine.ingest(synthetic_video)
    for q in ("red", "blue"):
        a = torch_engine.query(q).segments[0]
        b = ort_engine.query(q).segments[0]
        assert (a.start, a.end, a.num_hits) == (b.start, b.end, b.num_hits)
        assert a.score == pytest.approx(b.score, abs=1e-5)


def test_tiny_models_pipeline(tiny_settings: Settings, synthetic_video: Path) -> None:
    engine = VidQuery(tiny_settings)
    stats = engine.ingest(synthetic_video)
    assert stats.sampled_frames == 12 and stats.vectors >= 12
    result = engine.query("a man in a blue shirt walking left", max_segments=3)
    assert 1 <= len(result.segments) <= 3
    assert set(result.timings_ms) >= {"encode_text", "search", "merge"}
