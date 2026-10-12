"""Hugging Face SigLIP / SigLIP 2 (FixRes) towers wrapped as export-friendly modules."""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)

# FixRes SigLIP 2 checkpoints use the original SigLIP architecture (static image size), which is
# what TensorRT wants. NaFlex variants take variable patch sequences and are not supported here.
DEFAULT_SIGLIP = "google/siglip2-so400m-patch16-384"


class SiglipImageTower(nn.Module):
    """pixel_values ``[B, 3, S, S]`` -> L2-normalised embeddings ``[B, D]``."""

    def __init__(self, vision_model: nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision_model(pixel_values=pixel_values)
        return F.normalize(out.pooler_output, dim=-1)


class SiglipTextTower(nn.Module):
    """input_ids ``[B, L]`` (padded to max_length, no attention mask) -> ``[B, D]``."""

    def __init__(self, text_model: nn.Module) -> None:
        super().__init__()
        self.text_model = text_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.text_model(input_ids=input_ids)
        return F.normalize(out.pooler_output, dim=-1)


@functools.lru_cache(maxsize=2)
def load_hf_siglip(name: str) -> Any:
    """Load (and cache) a SigLIP-architecture checkpoint in fp32."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(name, attn_implementation="sdpa")
    model_type = getattr(model.config, "model_type", "")
    if model_type == "siglip2":
        raise ValueError(
            f"{name} is a NaFlex SigLIP 2 checkpoint (variable patch count). Use a FixRes "
            f"checkpoint such as {DEFAULT_SIGLIP} for static-shape ONNX/TensorRT export."
        )
    if not hasattr(model, "vision_model") or not hasattr(model, "text_model"):
        raise ValueError(f"{name} does not look like a SigLIP dual encoder")
    return model.float().eval()


@functools.lru_cache(maxsize=4)
def siglip_config(name: str) -> Any:
    """The checkpoint config only (no weights), e.g. for ONNX/TensorRT runs."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(name)


def siglip_image_params(name: str) -> tuple[int, tuple[float, ...], tuple[float, ...]]:
    """(image_size, mean, std) for a checkpoint, falling back to SigLIP defaults."""
    size = int(siglip_config(name).vision_config.image_size)
    mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    std: tuple[float, ...] = (0.5, 0.5, 0.5)
    try:
        from transformers import AutoImageProcessor

        proc = AutoImageProcessor.from_pretrained(name)
        mean = tuple(float(v) for v in proc.image_mean)
        std = tuple(float(v) for v in proc.image_std)
    except Exception as exc:  # processor config is optional
        log.warning("Using default SigLIP normalisation for %s (%s)", name, exc)
    return size, mean, std


class HFTokenizer:
    """SigLIP text preprocessing: lowercase, pad to ``max_length`` (SigLIP uses no mask)."""

    def __init__(self, name: str, max_length: int = 64, lowercase: bool = True) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.max_length = max_length
        self.lowercase = lowercase

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t.lower() for t in texts] if self.lowercase else list(texts)
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        return np.asarray(enc["input_ids"], dtype=np.int64)
