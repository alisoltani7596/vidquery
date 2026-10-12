from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vidquery.index import VectorIndex
from vidquery.models.encoder import l2_normalize
from vidquery.types import Record, VideoEntry


def _entry(video_id: str) -> VideoEntry:
    return VideoEntry(
        video_id=video_id,
        path=f"/v/{video_id}.mp4",
        fps=25.0,
        num_frames=250,
        width=640,
        height=480,
        duration=10.0,
        sample_fps=2.0,
    )


def _populate(index: VectorIndex, video_id: str, vecs: np.ndarray, kinds: list[str]) -> list[int]:
    ids = index.allocate_ids(len(vecs))
    recs = [
        Record(
            id=i,
            video_id=video_id,
            frame_idx=n * 12,
            timestamp=n * 0.5,
            kind=k,  # type: ignore[arg-type]
            box=(1.0, 2.0, 3.0, 4.0) if k == "crop" else None,
            det_score=0.9 if k == "crop" else None,
        )
        for n, (i, k) in enumerate(zip(ids, kinds, strict=True))
    ]
    index.add(vecs, recs)
    index.register_video(_entry(video_id))
    return ids


@pytest.fixture
def index() -> VectorIndex:
    rng = np.random.default_rng(0)
    idx = VectorIndex(dim=16, encoder_name="test")
    _populate(idx, "a", l2_normalize(rng.standard_normal((10, 16))), ["frame", "crop"] * 5)
    _populate(idx, "b", l2_normalize(rng.standard_normal((6, 16))), ["crop"] * 6)
    return idx


def test_search_exact_match_and_kinds(index: VectorIndex) -> None:
    rec = index.records[3]
    target = index._indexes["crop"].reconstruct(3)
    hits = index.search(target, k=5)
    assert hits[0].id == 3 and hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].box == [1.0, 2.0, 3.0, 4.0] and hits[0].timestamp == rec.timestamp
    assert all(h.kind == "frame" for h in index.search(target, k=5, kind="frame"))
    assert all(h.kind == "crop" for h in index.search(target, k=5, kind="crop"))
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert index.count() == 16 and index.count("frame") == 5 and index.count("crop") == 11


def test_video_filter_and_frame_weight(index: VectorIndex) -> None:
    q = l2_normalize(np.ones((1, 16)))[0]
    assert {h.video_id for h in index.search(q, k=20, video_ids=["b"])} == {"b"}
    frames = index.search(q, k=20, kind="frame", frame_weight=0.0)
    assert all(h.score == 0.0 for h in frames)


def test_remove_video(index: VectorIndex) -> None:
    assert index.remove_video("a") == 10
    assert "a" not in index.videos
    q = l2_normalize(np.ones((1, 16)))[0]
    assert {h.video_id for h in index.search(q, k=50)} == {"b"}
    assert index.count("frame") == 0


def test_save_load_roundtrip(index: VectorIndex, tmp_path: Path) -> None:
    q = l2_normalize(np.arange(16, dtype=np.float32)[None])[0]
    before = index.search(q, k=8)
    index.save(tmp_path / "idx")
    index.save(tmp_path / "idx")  # overwrite in place
    assert VectorIndex.exists(tmp_path / "idx")
    loaded = VectorIndex.load(tmp_path / "idx")
    after = loaded.search(q, k=8)
    assert [h.model_dump() for h in after] == [h.model_dump() for h in before]
    assert loaded.videos == index.videos
    assert loaded.encoder_name == "test"
    # New ids continue after the persisted ones.
    assert loaded.allocate_ids(1) == [16]


def test_add_validates_shapes() -> None:
    idx = VectorIndex(dim=4)
    with pytest.raises(ValueError, match="same length"):
        idx.add(np.zeros((2, 4)), [])
    rec = Record(id=0, video_id="v", frame_idx=0, timestamp=0.0, kind="frame")
    with pytest.raises(ValueError, match="shape"):
        idx.add(np.zeros((1, 3)), [rec])
    with pytest.raises(ValueError, match="dim"):
        idx.search(np.zeros(5), k=1)
