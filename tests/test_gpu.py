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
    s = _gpu(tiny_settings)
    export_component(s, "image_encoder", exporter="torchscript")
    backend = create_backend(s, "image_encoder", "onnxruntime", "cuda")
    if "CUDA" not in backend.description:
        pytest.skip("onnxruntime-gpu CUDA execution provider not available")
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
