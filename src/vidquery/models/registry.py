"""Build detectors and encoders from settings, on any backend.

A *component* is one exportable network: ``detector``, ``image_encoder`` or ``text_encoder``.
``component_spec`` describes how to build its torch module and what its input looks like; the
export scripts, parity checker, benchmark and pipeline all go through it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from torch import nn

from vidquery.backends.base import BackendSpec, InferenceBackend
from vidquery.config import Settings
from vidquery.models.detector import ColorBlobDetector, PersonDetector, YoloDetector
from vidquery.models.encoder import ImageEncoder, TextEncoder, Tokenizer

Component = Literal["detector", "image_encoder", "text_encoder"]
COMPONENTS: tuple[Component, ...] = ("detector", "image_encoder", "text_encoder")
NETWORK_FREE_DETECTORS = ("color-blob",)


@dataclass
class ComponentSpec:
    component: Component
    model_name: str
    input_name: str
    sample_shape: tuple[int, ...]
    input_dtype: type[np.generic]
    output_name: str
    build_module: Callable[[], nn.Module]
    int_high: int = 1  # exclusive upper bound for random integer inputs (token ids)
    custom_onnx_export: Callable[[Path, int, int], Path] | None = None

    def example_input(self, batch: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        shape = (batch, *self.sample_shape)
        if np.issubdtype(self.input_dtype, np.integer):
            return rng.integers(0, self.int_high, size=shape).astype(self.input_dtype)
        if self.component == "detector":
            return rng.random(shape, dtype=np.float32)
        return rng.standard_normal(shape).astype(np.float32).clip(-1, 1)


def model_slug(name: str) -> str:
    base = Path(name).name if name.endswith((".pt", ".pth", ".onnx")) else name
    base = re.sub(r"\.(pt|pth|onnx)$", "", base)
    return re.sub(r"[^A-Za-z0-9._-]+", "--", base).strip("-")


def _component_model(settings: Settings, component: Component) -> str:
    return settings.detector.model if component == "detector" else settings.encoder.model


def onnx_path(settings: Settings, component: Component) -> Path:
    explicit = {
        "detector": settings.detector.onnx_path,
        "image_encoder": settings.encoder.onnx_path,
        "text_encoder": settings.encoder.text_onnx_path,
    }[component]
    if explicit is not None:
        return explicit
    slug = model_slug(_component_model(settings, component))
    return settings.artifacts_dir / "onnx" / f"{component}__{slug}.onnx"


def engine_path(settings: Settings, component: Component, precision: str) -> Path:
    cfg_engine, cfg_precision = (
        (settings.detector.engine_path, settings.detector.precision)
        if component == "detector"
        else (settings.encoder.engine_path, settings.encoder.precision)
    )
    if cfg_engine is not None and component != "text_encoder" and precision == cfg_precision:
        return cfg_engine
    slug = model_slug(_component_model(settings, component))
    return settings.artifacts_dir / "trt" / f"{component}__{slug}__{precision}.engine"


def component_spec(settings: Settings, component: Component) -> ComponentSpec:
    if component == "detector":
        return _detector_spec(settings)
    return _encoder_spec(settings, component)


def _detector_spec(settings: Settings) -> ComponentSpec:
    cfg = settings.detector
    if cfg.model in NETWORK_FREE_DETECTORS:
        raise ValueError(f"Detector {cfg.model!r} has no network to export or benchmark")
    shape = (3, cfg.input_size, cfg.input_size)
    if cfg.model == "tiny":
        from vidquery.models.tiny import build_tiny_detector

        def build() -> nn.Module:
            return build_tiny_detector(settings.seed, cfg.input_size)

        return ComponentSpec("detector", "tiny", "images", shape, np.float32, "output", build)

    from vidquery.models.yolo import UltralyticsDetectorNet, export_ultralytics_onnx

    def export(out: Path, opset: int, max_batch: int) -> Path:
        return export_ultralytics_onnx(cfg.model, out, cfg.input_size, opset, max_batch)

    return ComponentSpec(
        "detector",
        cfg.model,
        "images",
        shape,
        np.float32,
        "output0",
        lambda: UltralyticsDetectorNet(cfg.model),
        custom_onnx_export=export,
    )


def _encoder_spec(settings: Settings, component: Component) -> ComponentSpec:
    name = settings.encoder.model
    image = component == "image_encoder"
    if name == "tiny":
        from vidquery.models.siglip import SiglipImageTower, SiglipTextTower
        from vidquery.models.tiny import (
            TINY_IMAGE_SIZE,
            TINY_TEXT_LENGTH,
            TINY_VOCAB,
            build_tiny_siglip,
        )

        if image:
            return ComponentSpec(
                component, name, "pixel_values", (3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE),
                np.float32, "embeddings",
                lambda: SiglipImageTower(build_tiny_siglip(settings.seed).vision_model),
            )
        return ComponentSpec(
            component, name, "input_ids", (TINY_TEXT_LENGTH,), np.int64, "embeddings",
            lambda: SiglipTextTower(build_tiny_siglip(settings.seed).text_model),
            int_high=TINY_VOCAB,
        )
    if name == "color-mock":
        from vidquery.models.mock import (
            MOCK_IMAGE_SIZE,
            MOCK_TEXT_LENGTH,
            ColorImageNet,
            ColorTextNet,
        )

        if image:
            return ComponentSpec(
                component, name, "pixel_values", (3, MOCK_IMAGE_SIZE, MOCK_IMAGE_SIZE),
                np.float32, "embeddings", ColorImageNet,
            )
        return ComponentSpec(
            component, name, "input_ids", (MOCK_TEXT_LENGTH,), np.int64, "embeddings",
            ColorTextNet, int_high=8,
        )

    from vidquery.models.siglip import (
        SiglipImageTower,
        SiglipTextTower,
        load_hf_siglip,
        siglip_config,
    )

    if image:
        size = int(siglip_config(name).vision_config.image_size)
        return ComponentSpec(
            component, name, "pixel_values", (3, size, size), np.float32, "embeddings",
            lambda: SiglipImageTower(load_hf_siglip(name).vision_model),
        )
    text_cfg = siglip_config(name).text_config
    length = min(settings.encoder.max_text_length, int(text_cfg.max_position_embeddings))
    return ComponentSpec(
        component, name, "input_ids", (length,), np.int64, "embeddings",
        lambda: SiglipTextTower(load_hf_siglip(name).text_model),
        int_high=int(text_cfg.vocab_size),
    )


def create_backend(
    settings: Settings,
    component: Component,
    backend: BackendSpec | str,
    device: str,
) -> InferenceBackend:
    """Instantiate ``component`` on the requested backend."""
    spec_b = BackendSpec.parse(backend) if isinstance(backend, str) else backend
    spec = component_spec(settings, component)
    if spec_b.kind == "pytorch":
        from vidquery.backends.torch_backend import TorchBackend

        return TorchBackend(
            spec.build_module(), [spec.input_name], [spec.output_name], device, spec_b.precision
        )
    if spec_b.kind == "onnxruntime":
        from vidquery.backends.onnx_backend import OnnxRuntimeBackend

        return OnnxRuntimeBackend(onnx_path(settings, component), device)
    from vidquery.backends.tensorrt_backend import TensorRTBackend

    return TensorRTBackend(
        engine_path(settings, component, spec_b.precision), device, spec_b.precision
    )


def build_detector(
    settings: Settings, device: str, backend: BackendSpec | None = None
) -> PersonDetector:
    cfg = settings.detector
    if cfg.model == "color-blob":
        return ColorBlobDetector(max_detections=cfg.max_detections)
    backend = backend or BackendSpec(cfg.backend, cfg.precision)
    return YoloDetector(
        create_backend(settings, "detector", backend, device),
        input_size=cfg.input_size,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        person_class=cfg.person_class,
        max_detections=cfg.max_detections,
    )


def image_params(settings: Settings) -> tuple[int, tuple[float, ...], tuple[float, ...]]:
    name = settings.encoder.model
    if name == "tiny":
        from vidquery.models.tiny import TINY_IMAGE_SIZE

        return TINY_IMAGE_SIZE, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if name == "color-mock":
        from vidquery.models.mock import MOCK_IMAGE_SIZE

        return MOCK_IMAGE_SIZE, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    from vidquery.models.siglip import siglip_image_params

    return siglip_image_params(name)


def build_image_encoder(
    settings: Settings, device: str, backend: BackendSpec | None = None
) -> ImageEncoder:
    cfg = settings.encoder
    backend = backend or BackendSpec(cfg.backend, cfg.precision)
    size, mean, std = image_params(settings)
    return ImageEncoder(
        create_backend(settings, "image_encoder", backend, device), size, mean, std, cfg.model
    )


def build_tokenizer(settings: Settings) -> Tokenizer:
    name = settings.encoder.model
    if name == "tiny":
        from vidquery.models.tiny import HashTokenizer

        return HashTokenizer()
    if name == "color-mock":
        from vidquery.models.mock import ColorTokenizer

        return ColorTokenizer()
    from vidquery.models.siglip import HFTokenizer

    length = component_spec(settings, "text_encoder").sample_shape[0]
    return HFTokenizer(name, max_length=length)


def build_text_encoder(settings: Settings, device: str) -> TextEncoder:
    backend = BackendSpec.parse(settings.encoder.text_backend)
    return TextEncoder(
        create_backend(settings, "text_encoder", backend, device),
        build_tokenizer(settings),
        settings.encoder.model,
    )
