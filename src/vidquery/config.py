"""Typed configuration loaded from YAML, with dotted ``key=value`` overrides.

String values may reference environment variables (``${SCRATCH}/vidquery``); they are
expanded at load time so the same YAML works locally and on the cluster.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

BackendName = Literal["pytorch", "onnxruntime", "tensorrt"]
Precision = Literal["fp32", "fp16", "int8"]
HitKind = Literal["both", "crop", "frame"]

CONFIG_ENV_VAR = "VIDQUERY_CONFIG"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DetectorConfig(_Base):
    """Person detector. ``model`` is an Ultralytics weights file, ``tiny`` or ``color-blob``."""

    model: str = "yolo26s.pt"
    backend: BackendName = "pytorch"
    precision: Precision = "fp32"
    input_size: int = Field(640, ge=32)
    conf_threshold: float = Field(0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    person_class: int = 0
    max_detections: int = Field(100, ge=1)
    onnx_path: Path | None = None
    engine_path: Path | None = None


class EncoderConfig(_Base):
    """SigLIP-style dual encoder. ``model`` is a HF id, ``tiny`` or ``color-mock``."""

    model: str = "google/siglip2-so400m-patch16-384"
    backend: BackendName = "pytorch"
    precision: Precision = "fp32"
    text_backend: Literal["pytorch", "onnxruntime"] = "pytorch"
    max_text_length: int = Field(64, ge=1)
    onnx_path: Path | None = None
    engine_path: Path | None = None
    text_onnx_path: Path | None = None


class IngestConfig(_Base):
    sample_fps: float = Field(2.0, gt=0.0)
    detector_batch_size: int = Field(8, ge=1)
    encoder_batch_size: int = Field(32, ge=1)
    crop_margin: float = Field(0.1, ge=0.0, le=1.0)
    min_box_size: int = Field(24, ge=1)
    max_crops_per_frame: int = Field(30, ge=0)
    embed_frames: bool = True
    embed_crops: bool = True
    prefetch_frames: int = Field(32, ge=0)


class QueryConfig(_Base):
    top_k: int = Field(200, ge=1, description="Raw nearest neighbours fetched before merging.")
    max_segments: int = Field(10, ge=1)
    merge_gap_s: float = Field(1.5, ge=0.0)
    min_hits: int = Field(1, ge=1)
    kind: HitKind = "both"
    frame_score_weight: float = Field(1.0, ge=0.0)
    score_margin: float | None = Field(
        0.05,
        ge=0.0,
        description="Drop hits scoring more than this below the best hit before merging "
        "(cosine units; tune with scripts/eval_retrieval.py). None disables it.",
    )
    min_score: float | None = Field(None, description="Absolute similarity floor for hits.")


class VerifierConfig(_Base):
    """VLM verifier reached over vLLM's OpenAI-compatible API."""

    enabled: bool = False
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3.5-4B"
    api_key: str = "EMPTY"
    top_n: int = Field(10, ge=1, description="How many top segments to re-check.")
    hits_per_segment: int = Field(1, ge=1)
    mode: Literal["rerank", "filter"] = "rerank"
    alpha: float = Field(0.7, ge=0.0, le=1.0, description="Weight of p(yes) in the fused score.")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Min p(yes) kept in filter mode.")
    timeout_s: float = Field(60.0, gt=0.0)
    max_concurrency: int = Field(8, ge=1)
    max_image_side: int = Field(448, ge=32)
    crop_margin: float = Field(0.15, ge=0.0, le=1.0)
    max_tokens: int = Field(1, ge=1)
    disable_thinking: bool = True
    prompt_template: str = (
        "Look carefully at the image. Does it show {query}? Answer with a single word: yes or no."
    )


GEO_VLM_PROMPT = (
    "Where was this picture taken? Use only visible clues: language and script on signs, "
    "licence plates, road markings, driving side, architecture, vegetation, landmarks. Do not "
    "use the appearance of people as a clue. Reply with JSON only, no other text: "
    '{"country": "<country or null>", "city": "<city or null>", "clues": ["<short clue>"]}. '
    "Give a city only if a visible clue such as a sign or a landmark supports it; otherwise "
    "use null. If nothing points to a country, use null for it too."
)


class GeoVLMConfig(_Base):
    """Optional VLM second opinion for ``locate``. Uses the endpoint and model in ``verifier``."""

    enabled: bool = False
    frames: int = Field(3, ge=1, le=16, description="Most confident frames sent, one per request.")
    max_tokens: int = Field(300, ge=16)
    max_image_side: int = Field(768, ge=64)
    prompt: str = GEO_VLM_PROMPT


class GeoConfig(_Base):
    """Visual geolocation (``vidquery locate``). The aggregation fields are tuned on a validation
    split by ``scripts/geo_eval.py tune`` and then frozen; see README "Where was this filmed?"."""

    backend: Literal["geoclip", "mock"] = "geoclip"
    clip_model: str = Field(
        "openai/clip-vit-large-patch14", description="CLIP backbone (HF id or local path)."
    )
    device: str | None = Field(None, description="Defaults to the top-level device.")
    sample_fps: float = Field(1.0, gt=0.0)
    max_frames: int = Field(120, ge=1, description="Sample fewer frames per second on long videos.")
    batch_size: int = Field(16, ge=1)
    top_k: int = Field(10, ge=1, le=1000, description="Gallery points kept per frame.")
    temperature: float = Field(1.0, gt=0.0, description="Softmax temperature over the top-k.")
    frame_weighting: Literal["uniform", "mass"] = Field(
        "uniform", description="'mass' weights frames by the probability held by their top-k."
    )
    cluster_km: float = Field(200.0, gt=0.0, description="Clustering bandwidth.")
    confidence_mass: float = Field(
        0.68, gt=0.0, lt=1.0, description="The radius holds this share of the probability mass."
    )
    min_radius_km: float = Field(25.0, gt=0.0, description="Never report a smaller radius.")
    alternatives: int = Field(3, ge=0, le=10)
    skip_person_frames: bool = True
    max_person_fraction: float = Field(
        0.5, ge=0.0, le=1.0, description="Skip frames where person boxes cover more than this."
    )
    map_tiles_url: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    map_attribution: str = (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    )
    vlm: GeoVLMConfig = Field(default_factory=GeoVLMConfig)


class ServerConfig(_Base):
    host: str = "0.0.0.0"
    port: int = 8080
    allowed_roots: list[Path] = Field(
        default_factory=list,
        description="If non-empty, /ingest and /locate only accept videos under these directories.",
    )


class Settings(_Base):
    device: str = "auto"
    index_dir: Path = Path("data/index")
    artifacts_dir: Path = Path("models")
    faiss_factory: str = "Flat"
    seed: int = 0
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    geo: GeoConfig = Field(default_factory=GeoConfig)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dotted!r}: {key!r} is not a section")
        node = child
    node[keys[-1]] = value


def parse_overrides(overrides: Sequence[str]) -> dict[str, Any]:
    """Turn ``["a.b=1", "c=foo"]`` into a nested dict; values are parsed as YAML scalars."""
    data: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like key.path=value")
        key, raw = item.split("=", 1)
        _set_dotted(data, key.strip(), yaml.safe_load(raw))
    return data


def _deep_merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    path: str | Path | None = None,
    overrides: Sequence[str] = (),
) -> Settings:
    """Load settings from ``path`` (or ``$VIDQUERY_CONFIG``) and apply overrides."""
    data: dict[str, Any] = {}
    path = path or os.environ.get(CONFIG_ENV_VAR)
    if path:
        with open(path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file {path} must contain a mapping")
        data = loaded
    data = _deep_merge(data, parse_overrides(overrides))
    return Settings.model_validate(_expand_env(data))


def resolve_device(device: str) -> str:
    """Map ``auto`` to ``cuda`` when available, else ``cpu``."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"
