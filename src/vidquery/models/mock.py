"""A deterministic "colour" dual encoder with real semantics, for end-to-end tests.

Images embed to their dominant saturated colour, texts embed to the colour words they mention,
so ``"person in red"`` retrieves red blobs. Both towers are plain ``nn.Module`` graphs, so they
also run through ONNX Runtime and TensorRT like the real encoders.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

COLOR_WORDS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "purple": (1.0, 0.0, 1.0),
}
MOCK_EMBED_DIM = 8
MOCK_IMAGE_SIZE = 32
MOCK_TEXT_LENGTH = 8
_VOCAB = list(COLOR_WORDS)
_WORD_RE = re.compile(r"[a-z]+")


class ColorImageNet(nn.Module):
    """Mean per-channel "colour excess" over the image, projected to ``MOCK_EMBED_DIM``."""

    proj: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        proj = torch.zeros(3, MOCK_EMBED_DIM)
        proj[0, 0] = proj[1, 1] = proj[2, 2] = 1.0
        self.register_buffer("proj", proj)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        rgb = pixel_values * 0.5 + 0.5  # undo SigLIP normalisation
        excess = torch.relu(rgb - rgb.mean(dim=1, keepdim=True))
        feat = excess.mean(dim=(2, 3))
        return feat @ self.proj


class ColorTextNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        table = torch.zeros(len(_VOCAB) + 1, MOCK_EMBED_DIM)
        for i, word in enumerate(_VOCAB, start=1):
            table[i, :3] = torch.tensor(COLOR_WORDS[word])
        self.embed = nn.Embedding.from_pretrained(table, freeze=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids).sum(dim=1)


class ColorTokenizer:
    def __init__(self, max_length: int = MOCK_TEXT_LENGTH) -> None:
        self.max_length = max_length

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.max_length), dtype=np.int64)
        for i, text in enumerate(texts):
            ids = [_VOCAB.index(w) + 1 for w in _WORD_RE.findall(text.lower()) if w in COLOR_WORDS]
            ids = ids[: self.max_length]
            out[i, : len(ids)] = ids
        return out
