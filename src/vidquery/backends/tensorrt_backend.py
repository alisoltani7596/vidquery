"""TensorRT backend (TensorRT >= 10 named-tensor API).

Device buffers are torch CUDA tensors, so no pycuda/cuda-python dependency is needed. Inputs are
copied from host and outputs back to host on every call, which keeps the interface identical to
the other backends (and makes benchmark numbers comparable: all include H2D/D2H copies).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vidquery.backends.base import InferenceBackend, as_float32


def _trt_dtype_to_torch(trt: Any, dtype: Any) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "int64"):
        mapping[trt.int64] = torch.int64
    if hasattr(trt, "bfloat16"):
        mapping[trt.bfloat16] = torch.bfloat16
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype {dtype}")
    return mapping[dtype]


class TensorRTBackend(InferenceBackend):
    kind = "tensorrt"

    def __init__(self, engine_path: str | Path, device: str = "cuda", precision: str = "fp32"):
        import tensorrt as trt

        if not torch.cuda.is_available():
            raise RuntimeError("The tensorrt backend needs a CUDA device")
        path = Path(engine_path)
        if not path.exists():
            raise FileNotFoundError(
                f"TensorRT engine not found: {path}. Run `vidquery export trt` first."
            )
        self.trt = trt
        self.precision = precision
        self.device = torch.device(device if device.startswith("cuda") else "cuda")
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as fh, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize {path} (TensorRT version or GPU mismatch?)")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [
            n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        ]
        self._dtypes = {n: _trt_dtype_to_torch(trt, self.engine.get_tensor_dtype(n)) for n in names}
        self.path = path
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream()

    @property
    def max_batch_size(self) -> int | None:
        name = self.input_names[0]
        shapes = self.engine.get_tensor_profile_shape(name, 0)
        return int(shapes[2][0])

    def run(self, inputs: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        keep_alive: list[torch.Tensor] = []
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            for name in self.input_names:
                arr = np.ascontiguousarray(inputs[name])
                t = torch.from_numpy(arr).to(
                    self.device, dtype=self._dtypes[name], non_blocking=True
                )
                t = t.contiguous()
                keep_alive.append(t)
                self.context.set_input_shape(name, tuple(t.shape))
                self.context.set_tensor_address(name, t.data_ptr())
            outputs: list[torch.Tensor] = []
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(d < 0 for d in shape):
                    raise RuntimeError(f"Unresolved output shape {shape} for {name}")
                out = torch.empty(shape, dtype=self._dtypes[name], device=self.device)
                self.context.set_tensor_address(name, out.data_ptr())
                outputs.append(out)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            host = [o.float().cpu() if o.is_floating_point() else o.cpu() for o in outputs]
        self.stream.synchronize()
        return [as_float32(h.numpy()) for h in host]

    def close(self) -> None:
        del self.context
        del self.engine
        torch.cuda.empty_cache()
