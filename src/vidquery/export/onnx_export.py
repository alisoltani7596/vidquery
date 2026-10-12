"""Export detector / image encoder / text encoder to ONNX with a dynamic batch axis."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from vidquery.config import Settings
from vidquery.models.registry import Component, ComponentSpec, component_spec, onnx_path

log = logging.getLogger(__name__)

Exporter = Literal["dynamo", "torchscript"]
MIN_OPSET = 17
DEFAULT_OPSET = {"dynamo": 18, "torchscript": 17}
EXTERNAL_DATA_THRESHOLD = 1_800_000_000  # bytes; protobuf caps a single file at 2 GB


def _param_bytes(module: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def export_module(
    spec: ComponentSpec,
    out_path: Path,
    *,
    opset: int | None = None,
    exporter: Exporter = "dynamo",
    max_batch: int = 64,
    example_batch: int = 2,
) -> Path:
    """Export ``spec``'s torch module to ``out_path`` (batch axis dynamic)."""
    opset = opset or DEFAULT_OPSET[exporter]
    if opset < MIN_OPSET:
        raise ValueError(f"opset must be >= {MIN_OPSET}, got {opset}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.custom_onnx_export is not None:
        log.info("Exporting %s with its native exporter", spec.model_name)
        return spec.custom_onnx_export(out_path, opset, max_batch)

    module = spec.build_module().eval().float().cpu()
    example = torch.from_numpy(spec.example_input(example_batch))
    big = _param_bytes(module) > EXTERNAL_DATA_THRESHOLD
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if exporter == "dynamo":
            batch = torch.export.Dim("batch", min=1, max=max_batch)
            program = torch.onnx.export(
                module,
                (example,),
                dynamo=True,
                input_names=[spec.input_name],
                output_names=[spec.output_name],
                opset_version=opset,
                dynamic_shapes={spec.input_name: {0: batch}},
                external_data=big,
                optimize=True,
            )
            if program is None:
                raise RuntimeError("Dynamo exporter returned no program")
            program.save(str(out_path), external_data=big)
        else:
            torch.onnx.export(
                module,
                (example,),
                str(out_path),
                dynamo=False,
                input_names=[spec.input_name],
                output_names=[spec.output_name],
                opset_version=opset,
                dynamic_axes={spec.input_name: {0: "batch"}, spec.output_name: {0: "batch"}},
                do_constant_folding=True,
            )
    check_onnx(out_path)
    return out_path


def check_onnx(path: Path) -> None:
    """Validate the graph and report its input/output signature."""
    import onnx

    onnx.checker.check_model(str(path))  # path-based check supports external data
    model = onnx.load(str(path), load_external_data=False)
    sig = [
        (i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim])
        for i in model.graph.input
    ]
    opsets = {o.domain or "ai.onnx": o.version for o in model.opset_import}
    log.info("Exported %s: inputs=%s opsets=%s", path, sig, opsets)


def export_component(
    settings: Settings,
    component: Component,
    *,
    out_path: Path | None = None,
    opset: int | None = None,
    exporter: Exporter = "dynamo",
    max_batch: int = 64,
) -> Path:
    spec = component_spec(settings, component)
    target = out_path or onnx_path(settings, component)
    return export_module(spec, target, opset=opset, exporter=exporter, max_batch=max_batch)


def onnx_input_shape(path: Path) -> list[int | str]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    return [d.dim_param if d.dim_param else int(d.dim_value) for d in dims]


def describe_signature(spec: ComponentSpec) -> str:
    dtype = np.dtype(spec.input_dtype).name
    return f"{spec.input_name}: [batch, {', '.join(map(str, spec.sample_shape))}] {dtype}"
