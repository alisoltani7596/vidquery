"""Export pipeline: torch -> ONNX -> TensorRT (FP32 / FP16 / INT8)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from vidquery.config import Settings
from vidquery.models.registry import Component, component_spec, engine_path, onnx_path

log = logging.getLogger(__name__)


def build_trt_engines(
    settings: Settings,
    component: Component,
    precisions: Sequence[str],
    *,
    min_batch: int = 1,
    opt_batch: int = 8,
    max_batch: int = 32,
    workspace_gb: float = 8.0,
    allow_tf32: bool = False,
    calib_sources: Sequence[str | Path] = (),
    calib_images: int = 512,
    int8_method: str = "calibrator",
    timing_cache: Path | None = None,
) -> dict[str, Path]:
    """Build one engine per precision for ``component`` from its exported ONNX file."""
    from vidquery.export.tensorrt_build import (
        ShapeProfile,
        build_engine,
        quantize_with_modelopt,
        resolve_sample_shape,
    )

    src = onnx_path(settings, component)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found; run `vidquery export onnx` first")
    spec = component_spec(settings, component)
    sample_shape = resolve_sample_shape(src, spec.sample_shape)
    profile = ShapeProfile(min_batch, opt_batch, max_batch)
    built: dict[str, Path] = {}
    for precision in precisions:
        out = engine_path(settings, component, precision)
        if precision != "int8":
            built[precision] = build_engine(
                src,
                out,
                sample_shape=sample_shape,
                precision=precision,
                profile=profile,
                workspace_gb=workspace_gb,
                allow_tf32=allow_tf32,
                timing_cache=timing_cache,
            )
            continue
        if not calib_sources:
            raise ValueError("INT8 needs --calib (videos, sequence dirs or images)")
        from vidquery.export.calibration import calibration_batches

        batches, data = calibration_batches(
            settings, component, calib_sources, num_images=calib_images, batch_size=opt_batch
        )
        if int8_method == "modelopt":
            qdq = src.with_name(src.stem + ".int8-qdq.onnx")
            quantize_with_modelopt(src, data, qdq)
            # Explicit quantization: Q/DQ nodes pin INT8 layers, FP16 is allowed elsewhere.
            built[precision] = build_engine(
                qdq,
                out,
                sample_shape=sample_shape,
                precision="int8",
                profile=profile,
                workspace_gb=workspace_gb,
                timing_cache=timing_cache,
            )
        elif int8_method == "calibrator":
            built[precision] = build_engine(
                src,
                out,
                sample_shape=sample_shape,
                precision="int8",
                profile=profile,
                workspace_gb=workspace_gb,
                calib_batches=batches,
                calib_cache=out.with_suffix(".calib"),
                timing_cache=timing_cache,
            )
        else:
            raise ValueError(f"Unknown int8 method {int8_method!r}")
    return built
