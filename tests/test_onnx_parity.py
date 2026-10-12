"""torch -> ONNX export and onnxruntime parity using the tiny, randomly initialised models."""

from __future__ import annotations

import numpy as np
import onnx
import pytest

from vidquery.backends.base import BackendSpec
from vidquery.backends.onnx_backend import OnnxRuntimeBackend
from vidquery.config import Settings
from vidquery.export.onnx_export import export_component, onnx_input_shape
from vidquery.models.registry import COMPONENTS, component_spec, create_backend, onnx_path
from vidquery.parity import DEFAULT_TOLERANCES, Tolerance, compare, detection_f1, run_parity
from vidquery.types import Detection


@pytest.mark.parametrize("component", COMPONENTS)
def test_torchscript_export_dynamic_batch_and_parity(
    tiny_settings: Settings, component: str
) -> None:
    path = export_component(tiny_settings, component, exporter="torchscript", opset=17)  # type: ignore[arg-type]
    model = onnx.load(str(path))
    assert any(o.version >= 17 for o in model.opset_import if o.domain in ("", "ai.onnx"))
    assert isinstance(onnx_input_shape(path)[0], str), "batch axis must be dynamic"

    spec = component_spec(tiny_settings, component)  # type: ignore[arg-type]
    torch_b = create_backend(tiny_settings, component, "pytorch", "cpu")  # type: ignore[arg-type]
    ort_b = OnnxRuntimeBackend(path)
    for batch in (1, 5):
        x = spec.example_input(batch, seed=batch)
        ref, out = torch_b(x)[0], ort_b(x)[0]
        assert out.shape[0] == batch
        np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)

    report = run_parity(tiny_settings, component, ["onnxruntime"], batch_size=3)  # type: ignore[arg-type]
    assert report.passed, report.to_markdown()
    if component == "detector":
        assert report.results[0].det_f1 == pytest.approx(1.0)


@pytest.mark.slow
def test_dynamo_export_parity(tiny_settings: Settings) -> None:
    path = export_component(tiny_settings, "image_encoder", exporter="dynamo")
    assert path == onnx_path(tiny_settings, "image_encoder")
    report = run_parity(tiny_settings, "image_encoder", ["onnxruntime"])
    assert report.passed, report.to_markdown()


def test_parity_detects_mismatch() -> None:
    ref = np.ones((4, 8), np.float32)
    ok, failures = compare(ref, ref + 1e-6, DEFAULT_TOLERANCES["fp32"])
    assert not failures and ok["allclose"]
    bad = ref.copy()
    bad[0] = -1
    _, failures = compare(ref, bad, DEFAULT_TOLERANCES["fp32"])
    assert any("allclose" in f for f in failures) and any("cosine" in f for f in failures)
    _, failures = compare(ref, ref[:2], Tolerance(min_cosine=0.9))
    assert failures and "shape" in failures[0]


def test_detection_f1() -> None:
    a = [[Detection(0, 0, 10, 10, 0.9), Detection(20, 20, 30, 30, 0.8)], []]
    assert detection_f1(a, a) == 1.0
    b = [[Detection(0, 0, 10, 10, 0.9)], [Detection(1, 1, 2, 2, 0.5)]]
    assert detection_f1(a, b) == pytest.approx(2 / 4)
    assert detection_f1([[]], [[]]) == 1.0


def test_missing_artifacts_are_reported(tiny_settings: Settings) -> None:
    report = run_parity(tiny_settings, "image_encoder", ["onnxruntime"])
    assert not report.passed
    assert "unavailable" in report.results[0].failures[0]


def test_backend_spec_parsing() -> None:
    assert BackendSpec.parse("trt:fp16") == BackendSpec("tensorrt", "fp16")
    assert str(BackendSpec.parse("ort")) == "onnxruntime:fp32"
    with pytest.raises(ValueError):
        BackendSpec.parse("onnxruntime:int8")
    with pytest.raises(ValueError):
        BackendSpec.parse("openvino")
