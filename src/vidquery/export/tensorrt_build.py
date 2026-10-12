"""Build TensorRT engines from ONNX: FP32, FP16 and INT8.

INT8 has two paths:

* ``calibrator`` (default): implicit quantization with ``IInt8EntropyCalibrator2`` fed by real
  frames/crops. Deprecated in TensorRT 10 but still supported, and needs no extra packages.
* ``modelopt``: explicit Q/DQ quantization with NVIDIA ModelOpt (``pip install
  vidquery[int8]``); the quantized ONNX is then built as a strongly typed network. This is the
  path NVIDIA recommends going forward.

Engines are specific to the GPU model and TensorRT version that built them: build them on the
cluster, inside the same container that runs them.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vidquery.export.onnx_export import onnx_input_shape

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShapeProfile:
    min_batch: int = 1
    opt_batch: int = 8
    max_batch: int = 32

    def shapes(self, sample_shape: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        return tuple((b, *sample_shape) for b in (self.min_batch, self.opt_batch, self.max_batch))


def _make_calibrator(
    trt: Any, batches: Iterator[np.ndarray], batch_size: int, cache_file: Path
) -> Any:
    import torch

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self._device_batch: torch.Tensor | None = None
            self._count = 0

        def get_batch_size(self) -> int:
            return batch_size

        def get_batch(self, names: list[str]) -> list[int] | None:
            try:
                arr = next(batches)
            except StopIteration:
                log.info("Calibration done after %d batches", self._count)
                return None
            self._count += 1
            self._device_batch = torch.from_numpy(np.ascontiguousarray(arr, np.float32)).cuda()
            return [int(self._device_batch.data_ptr())]

        def read_calibration_cache(self) -> bytes | None:
            if cache_file.exists():
                log.info("Using calibration cache %s", cache_file)
                return cache_file.read_bytes()
            return None

        def write_calibration_cache(self, cache: Any) -> None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(bytes(cache))

    return EntropyCalibrator()


def quantize_with_modelopt(onnx_path: Path, calib_data: np.ndarray, out_path: Path) -> Path:
    """Insert Q/DQ nodes with NVIDIA ModelOpt (explicit INT8)."""
    from modelopt.onnx.quantization import quantize

    quantize(
        onnx_path=str(onnx_path),
        quantize_mode="int8",
        calibration_data=calib_data,
        calibration_method="entropy",
        output_path=str(out_path),
    )
    return out_path


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    sample_shape: Sequence[int],
    precision: str = "fp16",
    profile: ShapeProfile | None = None,
    workspace_gb: float = 8.0,
    allow_tf32: bool = False,
    calib_batches: Iterator[np.ndarray] | None = None,
    calib_cache: Path | None = None,
    strongly_typed: bool = False,
    timing_cache: Path | None = None,
) -> Path:
    """Parse ``onnx_path`` and serialise an engine to ``engine_path``.

    On TensorRT releases without weakly-typed precision flags (TensorRT 11+), FP16 is built
    from an FP16 copy of the ONNX graph as a strongly typed network, and INT8 requires the
    ModelOpt (Q/DQ) path.
    """
    import tensorrt as trt

    has_precision_flags = hasattr(trt.BuilderFlag, "FP16")
    if not has_precision_flags and not strongly_typed:
        if precision == "fp16":
            onnx_path = convert_onnx_fp16(onnx_path)
            strongly_typed = True
        elif precision == "int8" and calib_batches is not None:
            raise RuntimeError(
                f"TensorRT {trt.__version__} has no implicit INT8 calibration; "
                "use --int8-method modelopt"
            )
        else:
            strongly_typed = True
    profile = profile or ShapeProfile()
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    flags = 0
    if strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if not allow_tf32 and not strongly_typed and hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)  # "fp32" means real FP32 for parity baselines
    if not strongly_typed:
        if precision in ("fp16", "int8"):
            config.set_flag(trt.BuilderFlag.FP16)  # INT8 engines fall back to FP16, not FP32
        if precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)

    tprofile = builder.create_optimization_profile()
    min_s, opt_s, max_s = profile.shapes(sample_shape)
    for i in range(network.num_inputs):
        name = network.get_input(i).name
        tprofile.set_shape(name, min_s, opt_s, max_s)
    config.add_optimization_profile(tprofile)

    if precision == "int8" and calib_batches is None:
        log.info("No calibrator given: expecting Q/DQ nodes in %s (explicit INT8)", onnx_path)
    elif precision == "int8" and not strongly_typed:
        assert calib_batches is not None
        cache = calib_cache or engine_path.with_suffix(".calib")
        config.int8_calibrator = _make_calibrator(trt, calib_batches, profile.opt_batch, cache)
        cal_profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            cal_profile.set_shape(network.get_input(i).name, opt_s, opt_s, opt_s)
        config.set_calibration_profile(cal_profile)

    cache_obj = None
    if timing_cache is not None:
        data = timing_cache.read_bytes() if timing_cache.exists() else b""
        cache_obj = config.create_timing_cache(data)
        config.set_timing_cache(cache_obj, ignore_mismatch=False)

    log.info("Building %s engine from %s (profile %s)", precision, onnx_path, (min_s, opt_s, max_s))
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    if timing_cache is not None and cache_obj is not None:
        timing_cache.parent.mkdir(parents=True, exist_ok=True)
        timing_cache.write_bytes(bytes(cache_obj.serialize()))
    log.info("Wrote %s in %.1fs", engine_path, time.perf_counter() - t0)
    return engine_path


def convert_onnx_fp16(onnx_path: Path) -> Path:
    """FP16 copy of an ONNX graph (I/O kept FP32) for strongly typed TensorRT builds."""
    import onnx
    from onnxconverter_common import float16

    out = onnx_path.with_name(onnx_path.stem + ".fp16.onnx")
    model = onnx.load(str(onnx_path))
    fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(fp16, str(out))
    log.info("Wrote FP16 ONNX %s for a strongly typed build", out)
    return out


def resolve_sample_shape(onnx_path: Path, fallback: Sequence[int]) -> tuple[int, ...]:
    """Per-item input shape: static dims from the ONNX graph, dynamic ones from ``fallback``."""
    dims = onnx_input_shape(onnx_path)[1:]
    if len(dims) != len(fallback):
        raise ValueError(f"ONNX input rank {len(dims) + 1} does not match {len(fallback) + 1}")
    return tuple(
        d if isinstance(d, int) and d > 0 else int(f) for d, f in zip(dims, fallback, strict=True)
    )
