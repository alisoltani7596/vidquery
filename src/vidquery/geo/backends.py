"""Image -> location backends. Each returns, per frame, its top-k gallery locations and logits.

* :class:`GeoCLIPBackend`: GeoCLIP (Vivanco Cepeda et al., NeurIPS 2023; MIT licence,
  ``pip install geoclip``). The pip package ships the GeoCLIP heads and its 100K-location GPS
  gallery; the CLIP ViT-L/14 backbone comes from the Hugging Face cache (``geo.clip_model``).
* :class:`MockGeoBackend`: deterministic and weight-free, for tests and the mock config.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np

from vidquery.geo.geometry import KM_PER_DEGREE

if TYPE_CHECKING:
    from vidquery.config import GeoConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FramePrediction:
    """Top-k gallery locations for one frame, best first.

    ``logits`` are the backend's similarity logits for those locations (temperature 1) and
    ``log_norm`` is the log-sum-exp over the *whole* gallery, so ``exp(logits - log_norm)`` are
    the full-gallery softmax probabilities of the kept locations.
    """

    lats: np.ndarray
    lons: np.ndarray
    logits: np.ndarray
    log_norm: float

    def __post_init__(self) -> None:
        if not (len(self.lats) == len(self.lons) == len(self.logits) > 0):
            raise ValueError("lats, lons and logits must be non-empty and the same length")

    @property
    def top_mass(self) -> float:
        """Full-gallery probability held by the kept locations (how peaked the frame is)."""
        return float(np.exp(np.asarray(self.logits, np.float64) - self.log_norm).sum())

    @property
    def top1_prob(self) -> float:
        return float(np.exp(float(self.logits[0]) - self.log_norm))

    def truncate(self, k: int) -> FramePrediction:
        if k >= len(self.lats):
            return self
        return FramePrediction(self.lats[:k], self.lons[:k], self.logits[:k], self.log_norm)


class GeoBackend(Protocol):
    name: str

    def predict(self, frames: Sequence[np.ndarray], top_k: int) -> list[FramePrediction]:
        """``frames`` are HxWx3 uint8 BGR images."""
        ...


def _bgr_to_pil(frames: Sequence[np.ndarray]) -> list[Any]:
    from PIL import Image

    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]


class GeoCLIPBackend:
    """GeoCLIP with the gallery's location features computed once and kept on the device."""

    name = "geoclip"

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 16,
        clip_model: str | None = "openai/clip-vit-large-patch14",
        gallery_chunk: int = 8192,
    ) -> None:
        """``clip_model=None`` builds the CLIP backbone with random weights (tests only: the
        GeoCLIP heads and gallery still load from the geoclip package, nothing is downloaded)."""
        import torch
        import torch.nn.functional as F

        try:
            from geoclip import GeoCLIP
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise ImportError(
                "The geoclip backend needs the 'geo' extra: pip install 'vidquery[geo]'"
            ) from exc

        self.device = device
        self.batch_size = batch_size
        if clip_model is None:
            model = GeoCLIP(clip_pretrained=False)
        else:
            model = GeoCLIP(clip_path=clip_model)
        model.eval().to(device)
        self.model = model
        with torch.no_grad():
            gallery = model.gps_gallery.to(device)
            feats = [
                F.normalize(model.location_encoder(gallery[i : i + gallery_chunk]), dim=1)
                for i in range(0, len(gallery), gallery_chunk)
            ]
            self.gallery_features = torch.cat(feats)
            self.logit_scale = float(model.logit_scale.exp())
        self.gallery = gallery.detach().cpu().numpy().astype(np.float64)
        log.info("GeoCLIP ready on %s with a %d-location gallery", device, len(self.gallery))

    def predict(self, frames: Sequence[np.ndarray], top_k: int) -> list[FramePrediction]:
        import torch
        import torch.nn.functional as F

        top_k = min(top_k, len(self.gallery))
        out: list[FramePrediction] = []
        for start in range(0, len(frames), self.batch_size):
            batch = frames[start : start + self.batch_size]
            with torch.no_grad():
                pixels = self.model.image_encoder.preprocess_image(_bgr_to_pil(batch))
                feats = F.normalize(self.model.image_encoder(pixels.to(self.device)), dim=1)
                logits = self.logit_scale * feats @ self.gallery_features.T
                log_norm = torch.logsumexp(logits, dim=1)
                top = torch.topk(logits, top_k, dim=1)
            values = top.values.float().cpu().numpy()
            indices = top.indices.cpu().numpy()
            norms = log_norm.float().cpu().numpy()
            for vals, idx, norm in zip(values, indices, norms, strict=True):
                pts = self.gallery[idx]
                out.append(FramePrediction(pts[:, 0], pts[:, 1], vals, float(norm)))
        return out


#: Anchor locations of the mock backend: Paris, New York, Tokyo, Sydney.
MOCK_ANCHORS: tuple[tuple[float, float], ...] = (
    (48.8566, 2.3522),
    (40.7128, -74.0060),
    (35.6762, 139.6503),
    (-33.8688, 151.2093),
)


def mock_anchor(frame: np.ndarray) -> int:
    """Which anchor the mock backend picks for a frame: its mean-brightness quartile."""
    return min(len(MOCK_ANCHORS) - 1, int(float(frame.mean()) // 64))


class MockGeoBackend:
    """Deterministic stand-in: a frame's brightness quartile picks an anchor city.

    About three quarters of the top-k sit within ``spread_km`` of that anchor with decreasing
    logits and the rest go to the other anchors with lower logits, so aggregation has a clear
    winner and real alternatives to find.
    """

    name = "mock"

    def __init__(self, spread_km: float = 20.0) -> None:
        self.spread_km = spread_km

    def _one(self, anchor: int, top_k: int) -> FramePrediction:
        n_main = max(1, round(0.75 * top_k))
        lat0, lon0 = MOCK_ANCHORS[anchor]
        others = [i for i in range(len(MOCK_ANCHORS)) if i != anchor]
        lats, lons, logits = [], [], []
        for k in range(top_k):
            if k < n_main:
                ring = self.spread_km * (k / max(1, n_main - 1)) / KM_PER_DEGREE
                ang = 2.399963 * k  # golden angle: spread the points evenly around the anchor
                lats.append(lat0 + ring * np.sin(ang))
                lons.append(lon0 + ring * np.cos(ang) / max(0.1, np.cos(np.radians(lat0))))
                logits.append(12.0 - 0.1 * k)
            else:
                j = k - n_main
                lat, lon = MOCK_ANCHORS[others[j % len(others)]]
                lats.append(lat)
                lons.append(lon)
                logits.append(10.0 - 0.2 * j)
        arr = np.asarray(logits, np.float32)
        # Pretend the rest of the gallery holds a third of the probability mass.
        norm = float(np.log(np.exp(arr.astype(np.float64)).sum() * 1.5))
        return FramePrediction(np.asarray(lats), np.asarray(lons), arr, norm)

    def predict(self, frames: Sequence[np.ndarray], top_k: int) -> list[FramePrediction]:
        return [self._one(mock_anchor(f), top_k) for f in frames]


def build_geo_backend(cfg: GeoConfig, device: str) -> GeoBackend:
    if cfg.backend == "mock":
        return MockGeoBackend()
    return GeoCLIPBackend(device=device, batch_size=cfg.batch_size, clip_model=cfg.clip_model)
