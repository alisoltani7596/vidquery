"""Video decoding and frame sampling.

Two sources are supported behind one interface:

* ``FileVideoSource``: anything OpenCV/FFmpeg can decode (mp4, avi, webm, ...).
* ``ImageSequenceSource``: a directory of images, e.g. a MOT17 ``img1/`` folder. The frame rate
  is read from a MOTChallenge ``seqinfo.ini`` when present.
"""

from __future__ import annotations

import configparser
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np

from vidquery.types import SampledFrame

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_FPS = 30.0

T = TypeVar("T")


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    num_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0


def sample_indices(num_frames: int, fps: float, sample_fps: float) -> list[int]:
    """Frame indices that approximate a ``sample_fps`` grid starting at t=0."""
    if num_frames <= 0:
        return []
    if sample_fps <= 0 or sample_fps >= fps:
        return list(range(num_frames))
    out: list[int] = []
    period = 1.0 / sample_fps
    next_t = 0.0
    for idx in range(num_frames):
        t = idx / fps
        if t + 1e-9 >= next_t:
            out.append(idx)
            next_t += period
            # Skip grid points we already passed (keeps one frame per slot).
            while next_t <= t + 1e-9:
                next_t += period
    return out


class VideoSource(ABC):
    """Common interface for decoding frames from a video-like source."""

    info: VideoInfo

    @abstractmethod
    def iter_frames(self, sample_fps: float | None = None) -> Iterator[SampledFrame]:
        """Yield frames sampled at ``sample_fps`` (all frames if None)."""

    @abstractmethod
    def read_frame(self, frame_idx: int) -> np.ndarray:
        """Random access to one frame (BGR uint8)."""

    def iter_range(self, start_s: float, end_s: float) -> Iterator[SampledFrame]:
        """Every frame with ``start_s <= t <= end_s``."""
        fps = self.info.fps
        first = max(0, int(np.floor(start_s * fps)))
        last = min(self.info.num_frames - 1, int(np.ceil(end_s * fps)))
        for idx in range(first, last + 1):
            yield SampledFrame(idx, idx / fps, self.read_frame(idx))

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release decoder resources."""

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class FileVideoSource(VideoSource):
    def __init__(self, path: str | Path, fps_hint: float | None = None) -> None:
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video {self.path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0 or not np.isfinite(fps):
            fps = fps_hint or DEFAULT_FPS
        num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if num <= 0:  # Some containers do not report a frame count: count by grabbing.
            num = self._count_frames()
        self.info = VideoInfo(self.path, fps, num, width, height)
        self._seek_cap: cv2.VideoCapture | None = None
        self._seek_pos = -1

    def _count_frames(self) -> int:
        cap = cv2.VideoCapture(self.path)
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        return n

    def iter_frames(self, sample_fps: float | None = None) -> Iterator[SampledFrame]:
        wanted = set(sample_indices(self.info.num_frames, self.info.fps, sample_fps or 0.0))
        cap = cv2.VideoCapture(self.path)
        try:
            idx = 0
            while True:
                if not cap.grab():
                    break
                if idx in wanted:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        yield SampledFrame(idx, idx / self.info.fps, frame)
                idx += 1
        finally:
            cap.release()

    def read_frame(self, frame_idx: int) -> np.ndarray:
        if self._seek_cap is None:
            self._seek_cap = cv2.VideoCapture(self.path)
            self._seek_pos = -1
        cap = self._seek_cap
        # Sequential reads (demo rendering) avoid a costly seek.
        if frame_idx != self._seek_pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise IndexError(f"Frame {frame_idx} out of range for {self.path}")
        self._seek_pos = frame_idx
        return np.asarray(frame)

    def close(self) -> None:
        if self._seek_cap is not None:
            self._seek_cap.release()
            self._seek_cap = None


def _read_seqinfo_fps(seq_dir: Path) -> float | None:
    ini = seq_dir / "seqinfo.ini"
    if not ini.exists():
        return None
    parser = configparser.ConfigParser()
    parser.read(ini)
    try:
        return float(parser["Sequence"]["frameRate"])
    except (KeyError, ValueError):
        return None


class ImageSequenceSource(VideoSource):
    def __init__(self, path: str | Path, fps_hint: float | None = None) -> None:
        root = Path(path)
        img_dir = root / "img1" if (root / "img1").is_dir() else root
        files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise FileNotFoundError(f"No images found in {img_dir}")
        fps = fps_hint or _read_seqinfo_fps(img_dir.parent) or _read_seqinfo_fps(root)
        first = cv2.imread(str(files[0]))
        if first is None:
            raise OSError(f"Cannot read {files[0]}")
        self.files = files
        self.path = str(root)
        self.info = VideoInfo(
            self.path, fps or DEFAULT_FPS, len(files), first.shape[1], first.shape[0]
        )

    def iter_frames(self, sample_fps: float | None = None) -> Iterator[SampledFrame]:
        for idx in sample_indices(self.info.num_frames, self.info.fps, sample_fps or 0.0):
            yield SampledFrame(idx, idx / self.info.fps, self.read_frame(idx))

    def read_frame(self, frame_idx: int) -> np.ndarray:
        if not 0 <= frame_idx < len(self.files):
            raise IndexError(f"Frame {frame_idx} out of range for {self.path}")
        img = cv2.imread(str(self.files[frame_idx]))
        if img is None:
            raise OSError(f"Cannot read {self.files[frame_idx]}")
        return np.asarray(img)


def open_video(path: str | Path, fps_hint: float | None = None) -> VideoSource:
    """Open a video file or an image-sequence directory."""
    p = Path(path)
    if p.is_dir():
        return ImageSequenceSource(p, fps_hint)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {p}")
    return FileVideoSource(p, fps_hint)


def prefetch(iterable: Iterable[T], size: int) -> Iterator[T]:
    """Run ``iterable`` in a background thread so decoding overlaps inference."""
    if size <= 0:
        yield from iterable
        return
    q: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=size)
    stop = threading.Event()

    def worker() -> None:
        try:
            for item in iterable:
                if stop.is_set():
                    return
                q.put(("item", item))
            q.put(("done", None))
        except BaseException as exc:  # propagate decoder errors to the consumer
            q.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        while True:
            tag, payload = q.get()
            if tag == "done":
                break
            if tag == "error":
                assert isinstance(payload, BaseException)
                raise payload
            yield payload  # type: ignore[misc]
    finally:
        stop.set()
        # Drain so the worker can exit if it is blocked on put().
        while thread.is_alive():
            try:
                q.get_nowait()
            except queue.Empty:
                thread.join(timeout=0.05)


def write_video(
    frames: Iterable[np.ndarray],
    out_path: str | Path,
    fps: float,
    on_frame: Callable[[int], None] | None = None,
) -> int:
    """Write BGR frames to an mp4/avi file. Returns the number of frames written."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*("MJPG" if out.suffix.lower() == ".avi" else "mp4v"))
    writer: cv2.VideoWriter | None = None
    n = 0
    try:
        for frame in frames:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise OSError(f"Cannot open video writer for {out}")
            writer.write(frame)
            n += 1
            if on_frame is not None:
                on_frame(n)
    finally:
        if writer is not None:
            writer.release()
    return n


def frames_to_video(seq_path: str | Path, out_path: str | Path, fps: float | None = None) -> int:
    """Convert an image sequence (e.g. a MOT17 sequence) to a video file."""
    src = ImageSequenceSource(seq_path, fps)
    frames = (src.read_frame(i) for i in range(src.info.num_frames))
    return write_video(frames, out_path, src.info.fps)
