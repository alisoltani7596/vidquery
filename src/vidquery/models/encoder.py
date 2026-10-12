"""Image and text encoders (SigLIP-style) on top of the backend interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import cv2
import numpy as np

from vidquery.backends.base import InferenceBackend


class Tokenizer(Protocol):
    max_length: int

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """Return int64 token ids of shape ``[B, max_length]``."""
        ...


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norm, eps)).astype(np.float32)


def preprocess_images(
    images: Sequence[np.ndarray],
    size: int,
    mean: Sequence[float] = (0.5, 0.5, 0.5),
    std: Sequence[float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """BGR uint8 images -> float32 ``[B, 3, size, size]`` (squash-resize, like SigLIP)."""
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    batch = np.empty((len(images), 3, size, size), dtype=np.float32)
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
        resized = cv2.resize(img, (size, size), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)
        batch[i] = (rgb / 255.0 - mean_arr) / std_arr
    return batch


class ImageEncoder:
    def __init__(
        self,
        backend: InferenceBackend,
        image_size: int,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
        name: str = "",
    ) -> None:
        self.backend = backend
        self.image_size = image_size
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.name = name

    def preprocess(self, images: Sequence[np.ndarray]) -> np.ndarray:
        return preprocess_images(images, self.image_size, self.mean, self.std)

    def encode(self, images: Sequence[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalised embeddings ``[N, D]``."""
        if not images:
            raise ValueError("encode() needs at least one image")
        bs = min(batch_size, self.backend.max_batch_size or batch_size)
        chunks = [
            self.backend(self.preprocess(images[i : i + bs]))[0] for i in range(0, len(images), bs)
        ]
        return l2_normalize(np.concatenate(chunks, axis=0))


class TextEncoder:
    def __init__(self, backend: InferenceBackend, tokenizer: Tokenizer, name: str = "") -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.name = name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("encode() needs at least one text")
        ids = self.tokenizer(list(texts))
        return l2_normalize(self.backend(ids)[0])


def crop_with_margin(
    image: np.ndarray, box: Sequence[float], margin: float, min_size: int = 2
) -> np.ndarray:
    """Crop ``box`` (xyxy) expanded by ``margin`` of its size, clipped to the image."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    xa, ya = max(0, int(np.floor(x1 - mx))), max(0, int(np.floor(y1 - my)))
    xb, yb = min(w, int(np.ceil(x2 + mx))), min(h, int(np.ceil(y2 + my)))
    if xb - xa < min_size or yb - ya < min_size:
        raise ValueError(f"Degenerate crop for box {box}")
    return image[ya:yb, xa:xb]
