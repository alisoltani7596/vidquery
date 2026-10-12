"""Tiny, randomly initialised models for CPU tests, ONNX parity checks and smoke benchmarks.

They exercise the same code paths as the real models (the tiny SigLIP *is* a Hugging Face
``SiglipModel``, just with a toy config) but need no weights and no network access.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

TINY_IMAGE_SIZE = 32
TINY_TEXT_LENGTH = 16
TINY_VOCAB = 256
TINY_EMBED_DIM = 32
TINY_DETECTOR_SIZE = 64


class TinyDetectorNet(nn.Module):
    """A 3-conv YOLO-ish head emitting the *dense* layout ``[B, 4 + nc, N]``."""

    grid: torch.Tensor

    def __init__(self, num_classes: int = 1, width: int = 8, input_size: int = TINY_DETECTOR_SIZE):
        super().__init__()
        self.stride = 8
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(2 * width, 4 + num_classes, 1)
        g = input_size // self.stride
        ys, xs = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
        grid = torch.stack([xs, ys], dim=0).reshape(1, 2, g * g).float()
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        f = self.head(self.backbone(images))
        b, c = f.shape[0], f.shape[1]
        f = f.reshape(b, c, -1)
        xy = (self.grid + torch.sigmoid(f[:, 0:2])) * self.stride
        wh = torch.sigmoid(f[:, 2:4]) * (4 * self.stride)
        scores = torch.sigmoid(f[:, 4:])
        return torch.cat([xy, wh, scores], dim=1)


def build_tiny_detector(seed: int = 0, input_size: int = TINY_DETECTOR_SIZE) -> TinyDetectorNet:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TinyDetectorNet(input_size=input_size).eval()


def build_tiny_siglip(seed: int = 0) -> Any:
    """A randomly initialised Hugging Face ``SiglipModel`` with a toy config."""
    from transformers import SiglipConfig, SiglipModel

    text = {
        "hidden_size": TINY_EMBED_DIM,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "vocab_size": TINY_VOCAB,
        "max_position_embeddings": TINY_TEXT_LENGTH,
        "projection_size": TINY_EMBED_DIM,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    vision = {
        "hidden_size": TINY_EMBED_DIM,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "image_size": TINY_IMAGE_SIZE,
        "patch_size": 8,
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SiglipModel(SiglipConfig(text_config=text, vision_config=vision))
    return model.eval()


_WORD_RE = re.compile(r"[a-z0-9]+")


class HashTokenizer:
    """Deterministic whitespace tokenizer: word -> crc32 bucket. Pads to ``max_length``."""

    def __init__(self, vocab_size: int = TINY_VOCAB, max_length: int = TINY_TEXT_LENGTH):
        self.vocab_size = vocab_size
        self.max_length = max_length

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.max_length), dtype=np.int64)
        for i, text in enumerate(texts):
            words = _WORD_RE.findall(text.lower())[: self.max_length]
            for j, word in enumerate(words):
                out[i, j] = 3 + zlib.crc32(word.encode()) % (self.vocab_size - 3)
        return out
