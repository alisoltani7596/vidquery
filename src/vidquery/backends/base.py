"""The backend interface shared by PyTorch, ONNX Runtime and TensorRT.

A backend runs *only the network*: numpy arrays in, numpy arrays out. Pre- and post-processing
live in the model wrappers (``vidquery.models``) so they are identical across backends, which is
what makes the parity checks meaningful.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

BackendKind = Literal["pytorch", "onnxruntime", "tensorrt"]
BACKEND_KINDS: tuple[BackendKind, ...] = ("pytorch", "onnxruntime", "tensorrt")
PRECISIONS = ("fp32", "fp16", "int8")


@dataclass(frozen=True)
class BackendSpec:
    """A backend plus numeric precision, written as ``kind[:precision]`` on the CLI."""

    kind: BackendKind
    precision: str = "fp32"

    @classmethod
    def parse(cls, text: str) -> BackendSpec:
        kind, _, precision = text.partition(":")
        kind = kind.strip().lower()
        aliases = {
            "torch": "pytorch",
            "ort": "onnxruntime",
            "onnx": "onnxruntime",
            "trt": "tensorrt",
        }
        kind = aliases.get(kind, kind)
        if kind not in BACKEND_KINDS:
            raise ValueError(f"Unknown backend {kind!r}; expected one of {BACKEND_KINDS}")
        precision = (precision or "fp32").strip().lower()
        if precision not in PRECISIONS:
            raise ValueError(f"Unknown precision {precision!r}; expected one of {PRECISIONS}")
        if precision == "int8" and kind != "tensorrt":
            raise ValueError("int8 is only supported by the tensorrt backend")
        if precision == "fp16" and kind == "onnxruntime":
            raise ValueError("onnxruntime runs the fp32 ONNX graph; use tensorrt:fp16 for fp16")
        return cls(kind, precision)

    def __str__(self) -> str:
        return f"{self.kind}:{self.precision}"


class InferenceBackend(ABC):
    """Runs a network on host numpy arrays."""

    kind: BackendKind
    precision: str = "fp32"
    input_names: list[str]
    output_names: list[str]

    @property
    def max_batch_size(self) -> int | None:
        """Largest batch accepted (TensorRT optimisation profiles); None means unbounded."""
        return None

    @property
    def description(self) -> str:
        return f"{self.kind}:{self.precision}"

    @abstractmethod
    def run(self, inputs: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        """Run inference and return outputs in ``output_names`` order (float32 for floats)."""

    def __call__(self, *arrays: np.ndarray) -> list[np.ndarray]:
        if len(arrays) != len(self.input_names):
            raise ValueError(f"Expected {len(self.input_names)} inputs, got {len(arrays)}")
        return self.run(dict(zip(self.input_names, arrays, strict=True)))

    def close(self) -> None:  # noqa: B027 - optional hook
        """Free device resources."""


def as_float32(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.floating) and arr.dtype != np.float32:
        return arr.astype(np.float32)
    return arr
