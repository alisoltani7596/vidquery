"""ONNX Runtime backend (CPU or CUDA execution provider)."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from vidquery.backends.base import InferenceBackend, as_float32

log = logging.getLogger(__name__)

_ORT_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def _usable_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS
        return os.cpu_count() or 1


class OnnxRuntimeBackend(InferenceBackend):
    kind = "onnxruntime"

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        intra_op_threads: int | None = None,
    ) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {path}. Run `vidquery export onnx` first."
            )
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # ORT's default sizes the pool to every core on the node and pins threads, which fails
        # inside a Slurm cgroup; use the CPUs this process may actually run on.
        opts.intra_op_num_threads = intra_op_threads or _usable_cpus()
        providers: list[str | tuple[str, dict[str, str]]] = ["CPUExecutionProvider"]
        if device.startswith("cuda"):
            dev_id = device.split(":", 1)[1] if ":" in device else "0"
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, ("CUDAExecutionProvider", {"device_id": dev_id}))
            else:
                log.warning("CUDAExecutionProvider unavailable; onnxruntime will run on CPU")
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        self.providers = self.session.get_providers()
        if len(providers) > 1 and "CUDAExecutionProvider" not in self.providers:
            # ORT only warns when the CUDA provider fails to load (CUDA major or cuDNN mismatch).
            log.error(
                "CUDAExecutionProvider failed to load for %s; running on %s", path, self.providers
            )
        self.path = path
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._input_dtypes = {
            i.name: _ORT_TO_NUMPY.get(i.type, np.float32) for i in self.session.get_inputs()
        }

    @property
    def description(self) -> str:
        provider = self.providers[0].replace("ExecutionProvider", "") if self.providers else "?"
        return f"onnxruntime:{self.precision} ({provider})"

    def run(self, inputs: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        feed: dict[str, np.ndarray] = {
            name: np.ascontiguousarray(inputs[name], dtype=self._input_dtypes[name])
            for name in self.input_names
        }
        return [as_float32(np.asarray(o)) for o in self.session.run(None, feed)]

    def close(self) -> None:
        del self.session
