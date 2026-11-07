"""GPU-only tests, skipped without CUDA. On the cluster: ``pytest -m gpu``."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from vidquery.config import Settings
from vidquery.export.onnx_export import export_component
from vidquery.models.registry import create_backend
from vidquery.parity import run_parity

pytestmark = pytest.mark.gpu

needs_trt = pytest.mark.skipif(
    importlib.util.find_spec("tensorrt") is None, reason="TensorRT python package not installed"
)


def _gpu(settings: Settings) -> Settings:
    s = settings.model_copy(deep=True)
    s.device = "cuda"
    return s


def test_torch_fp16_parity(tiny_settings: Settings) -> None:
    s = _gpu(tiny_settings)
    report = run_parity(s, "image_encoder", ["pytorch:fp16"], device="cuda")
    assert report.passed, report.to_markdown()


def test_onnxruntime_cuda(tiny_settings: Settings) -> None:
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CPU-only onnxruntime installed")
    s = _gpu(tiny_settings)
    export_component(s, "image_encoder", exporter="torchscript")
    backend = create_backend(s, "image_encoder", "onnxruntime", "cuda")
    # onnxruntime-gpu is installed, so a CPU session means the CUDA provider failed to load.
    assert "CUDA" in backend.description, f"fell back to CPU: {backend.description}"
    report = run_parity(s, "image_encoder", ["onnxruntime"], device="cuda")
    assert report.passed, report.to_markdown()


@needs_trt
@pytest.mark.parametrize("component", ["detector", "image_encoder", "text_encoder"])
def test_tensorrt_engines_parity(tiny_settings: Settings, component: str) -> None:
    from vidquery.export import build_trt_engines

    s = _gpu(tiny_settings)
    export_component(s, component, exporter="torchscript")  # type: ignore[arg-type]
    built = build_trt_engines(s, component, ["fp32", "fp16"], max_batch=8, opt_batch=4)  # type: ignore[arg-type]
    assert set(built) == {"fp32", "fp16"}
    report = run_parity(s, component, ["tensorrt:fp32", "tensorrt:fp16"], device="cuda")  # type: ignore[arg-type]
    assert report.passed, report.to_markdown()


@needs_trt
@pytest.mark.xfail(
    reason="TensorRT 11 FP16 fallback is unsupported: the FP16 ONNX copy still feeds an FP32 "
    "tensor into the patch-embedding Conv, which a strongly typed network rejects. The "
    "container is pinned to 26.03 (TensorRT 10.16), which uses the weakly typed FP16 path.",
    raises=RuntimeError,
    strict=False,
)
def test_tensorrt_strongly_typed_fp16(tiny_settings: Settings) -> None:
    """The TensorRT 11 FP16 route (FP16 ONNX copy, strongly typed network), forced on 10.x."""
    from vidquery.export.tensorrt_build import (
        ShapeProfile,
        build_engine,
        convert_onnx_fp16,
        resolve_sample_shape,
    )
    from vidquery.models.registry import component_spec, engine_path, onnx_path

    s = _gpu(tiny_settings)
    export_component(s, "image_encoder", exporter="torchscript")
    src = onnx_path(s, "image_encoder")
    build_engine(
        convert_onnx_fp16(src),
        engine_path(s, "image_encoder", "fp16"),
        sample_shape=resolve_sample_shape(src, component_spec(s, "image_encoder").sample_shape),
        precision="fp16",
        profile=ShapeProfile(1, 4, 8),
        strongly_typed=True,
    )
    report = run_parity(s, "image_encoder", ["tensorrt:fp16"], device="cuda")
    assert report.passed, report.to_markdown()


@needs_trt
def test_tensorrt_int8_calibration(tiny_settings: Settings, synthetic_video) -> None:  # type: ignore[no-untyped-def]
    from vidquery.export import build_trt_engines

    s = _gpu(tiny_settings)
    export_component(s, "image_encoder", exporter="torchscript")
    built = build_trt_engines(
        s,
        "image_encoder",
        ["int8"],
        max_batch=8,
        opt_batch=4,
        calib_sources=[synthetic_video],
        calib_images=64,
    )
    backend = create_backend(s, "image_encoder", "tensorrt:int8", "cuda")
    out = backend(np.zeros((2, 3, 32, 32), np.float32))[0]
    assert built["int8"].exists() and out.shape == (2, 32)
