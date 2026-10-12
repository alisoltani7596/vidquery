"""PyTorch eager backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from vidquery.backends.base import InferenceBackend, as_float32


class TorchBackend(InferenceBackend):
    kind = "pytorch"

    def __init__(
        self,
        module: nn.Module,
        input_names: Sequence[str],
        output_names: Sequence[str] = ("output",),
        device: str = "cpu",
        precision: str = "fp32",
    ) -> None:
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"pytorch backend supports fp32/fp16, got {precision}")
        if precision == "fp16" and not device.startswith("cuda"):
            raise ValueError("fp16 with the pytorch backend requires a CUDA device")
        self.device = torch.device(device)
        self.precision = precision
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.module = module.eval().to(self.device, dtype=self.dtype)
        self.input_names = list(input_names)
        self.output_names = list(output_names)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device, non_blocking=True)
        if t.is_floating_point():
            t = t.to(self.dtype)
        return t

    @torch.inference_mode()
    def run(self, inputs: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        tensors = [self._to_tensor(inputs[name]) for name in self.input_names]
        out = self.module(*tensors)
        outs = list(out) if isinstance(out, (tuple, list)) else [out]
        return [
            as_float32(o.float().cpu().numpy() if o.is_floating_point() else o.cpu().numpy())
            for o in outs[: len(self.output_names)]
        ]

    def close(self) -> None:
        del self.module
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
