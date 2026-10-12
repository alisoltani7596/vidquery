"""Calibration data for INT8: real frames (detector) and frame/crop mixes (image encoder)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from vidquery.config import Settings
from vidquery.models.detector import preprocess_letterbox
from vidquery.models.encoder import preprocess_images
from vidquery.models.registry import Component, image_params
from vidquery.video import IMAGE_EXTENSIONS, open_video


def collect_frames(
    sources: Sequence[str | Path], num_frames: int, sample_fps: float = 0.5
) -> list[np.ndarray]:
    """Up to ``num_frames`` BGR frames spread over the given videos / image directories."""
    import cv2

    frames: list[np.ndarray] = []
    per_source = max(1, num_frames // max(1, len(sources)))
    for src in sources:
        p = Path(src)
        if (
            p.is_dir()
            and not (p / "img1").is_dir()
            and not any(f.suffix.lower() in IMAGE_EXTENSIONS for f in p.iterdir())
        ):
            # A directory of videos/sequences: recurse one level.
            frames.extend(collect_frames(sorted(p.iterdir()), per_source, sample_fps))
            continue
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
            continue
        with open_video(p) as video:
            for i, frame in enumerate(video.iter_frames(sample_fps)):
                if i >= per_source:
                    break
                frames.append(frame.image)
        if len(frames) >= num_frames:
            break
    if not frames:
        raise ValueError(f"No calibration frames found in {list(map(str, sources))}")
    return frames[:num_frames]


def _random_crops(frames: list[np.ndarray], n: int, seed: int) -> list[np.ndarray]:
    """Person-shaped (tall) random crops, to cover the crop distribution the encoder sees."""
    rng = np.random.default_rng(seed)
    crops = []
    for _ in range(n):
        img = frames[int(rng.integers(len(frames)))]
        h, w = img.shape[:2]
        ch = int(rng.uniform(0.15, 0.6) * h)
        cw = max(8, int(ch * rng.uniform(0.3, 0.6)))
        cw = min(cw, w)
        y = int(rng.integers(0, max(1, h - ch)))
        x = int(rng.integers(0, max(1, w - cw)))
        crops.append(img[y : y + ch, x : x + cw])
    return crops


def calibration_batches(
    settings: Settings,
    component: Component,
    sources: Sequence[str | Path],
    *,
    num_images: int = 512,
    batch_size: int = 8,
) -> tuple[Iterator[np.ndarray], np.ndarray]:
    """Preprocessed calibration batches plus the stacked array (for ModelOpt)."""
    if component == "text_encoder":
        raise ValueError("INT8 calibration is only implemented for image inputs")
    frames = collect_frames(
        sources, max(16, num_images // 2 if component == "image_encoder" else num_images)
    )
    if component == "detector":
        data, _ = preprocess_letterbox(frames, settings.detector.input_size)
    else:
        size, mean, std = image_params(settings)
        images = frames + _random_crops(frames, num_images - len(frames), settings.seed)
        data = preprocess_images(images[:num_images], size, mean, std)
    usable = (len(data) // batch_size) * batch_size
    if usable == 0:
        raise ValueError("Not enough calibration images for one batch")
    data = data[:usable]
    batches = iter([data[i : i + batch_size] for i in range(0, usable, batch_size)])
    return batches, data
