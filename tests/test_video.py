from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vidquery.video import (
    ImageSequenceSource,
    frames_to_video,
    open_video,
    prefetch,
    sample_indices,
)


def test_sample_indices_grid() -> None:
    assert sample_indices(30, 10.0, 2.0) == [0, 5, 10, 15, 20, 25]
    # Non-integer ratio: one frame per 1/3 s slot at 25 fps.
    idx = sample_indices(100, 25.0, 3.0)
    times = np.array(idx) / 25.0
    assert np.all(np.diff(times) > 0.28)
    assert len(idx) == 12
    # sample_fps >= fps or <= 0 means every frame.
    assert sample_indices(5, 10.0, 30.0) == [0, 1, 2, 3, 4]
    assert sample_indices(5, 10.0, 0.0) == [0, 1, 2, 3, 4]
    assert sample_indices(0, 10.0, 2.0) == []


def test_file_source_sampling_and_seek(synthetic_video: Path) -> None:
    with open_video(synthetic_video) as src:
        assert src.info.fps == pytest.approx(10.0)
        assert src.info.num_frames == 120
        assert src.info.duration == pytest.approx(12.0)
        frames = list(src.iter_frames(2.0))
        assert [f.frame_idx for f in frames[:3]] == [0, 5, 10]
        assert frames[1].timestamp == pytest.approx(0.5)
        assert len(frames) == 24
        again = src.read_frame(frames[3].frame_idx)
        assert np.abs(again.astype(int) - frames[3].image.astype(int)).mean() < 2
        rng = list(src.iter_range(1.0, 1.5))
        assert [f.frame_idx for f in rng] == [10, 11, 12, 13, 14, 15]
        with pytest.raises(IndexError):
            src.read_frame(10_000)


def test_image_sequence_source_reads_seqinfo(tmp_path: Path) -> None:
    seq = tmp_path / "MOT17-99"
    (seq / "img1").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text("[Sequence]\nname=MOT17-99\nframeRate=14\n")
    for i in range(7):
        img = np.full((48, 64, 3), i * 30, np.uint8)
        cv2.imwrite(str(seq / "img1" / f"{i + 1:06d}.jpg"), img)
    src = open_video(seq)
    assert isinstance(src, ImageSequenceSource)
    assert src.info.fps == 14.0
    assert src.info.num_frames == 7
    assert (src.info.width, src.info.height) == (64, 48)
    assert [f.frame_idx for f in src.iter_frames(7.0)] == [0, 2, 4, 6]

    out = tmp_path / "seq.mp4"
    assert frames_to_video(seq, out) == 7
    with open_video(out) as video:
        assert video.info.num_frames == 7
        assert video.info.fps == pytest.approx(14.0)


def test_open_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_video(tmp_path / "nope.mp4")


def test_prefetch_preserves_order_and_errors() -> None:
    assert list(prefetch(iter(range(100)), 4)) == list(range(100))
    assert list(prefetch(iter(range(5)), 0)) == list(range(5))

    def boom():  # type: ignore[no-untyped-def]
        yield 1
        raise RuntimeError("decoder failed")

    with pytest.raises(RuntimeError, match="decoder failed"):
        list(prefetch(boom(), 2))
