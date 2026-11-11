"""Video-level aggregation of per-frame location predictions.

Every frame contributes its top-k gallery locations, weighted by a softmax over their logits (at
``temperature``) times the frame's weight. The pooled points are clustered on the sphere with a
leader pass at ``cluster_km`` followed by a few re-centring passes. The heaviest cluster is the
estimate; its centre is the weighted spherical mean of its points.

The confidence radius is the distance from that centre within which ``confidence_mass`` of *all*
the pooled probability lies (floored at ``min_radius_km``). When the frames disagree the radius
grows to cover the competing clusters, so a split decision reads as a large area instead of a
falsely precise point. The next clusters (more than ``cluster_km`` from the chosen ones) are the
alternatives; ``share`` is the share of the probability mass each cluster holds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from vidquery.geo.backends import FramePrediction
from vidquery.geo.geometry import (
    MAX_DISTANCE_KM,
    angular_km,
    from_unit_vectors,
    to_unit_vectors,
    weighted_quantile,
)

if TYPE_CHECKING:
    from vidquery.config import GeoConfig

FrameWeighting = Literal["uniform", "mass"]


@dataclass(frozen=True)
class AggregateParams:
    """The tunable aggregation settings (mirrors the fields of the same name in GeoConfig)."""

    top_k: int = 10
    temperature: float = 1.0
    frame_weighting: FrameWeighting = "uniform"
    cluster_km: float = 200.0
    confidence_mass: float = 0.68
    min_radius_km: float = 25.0
    alternatives: int = 3

    @classmethod
    def from_config(cls, cfg: GeoConfig) -> AggregateParams:
        return cls(**{f.name: getattr(cfg, f.name) for f in fields(cls)})

    def with_overrides(self, **kwargs: Any) -> AggregateParams:
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class Location:
    lat: float
    lon: float
    radius_km: float
    share: float


@dataclass(frozen=True)
class Aggregate:
    best: Location
    alternatives: list[Location] = field(default_factory=list)
    frames: int = 0


@dataclass(frozen=True)
class PooledPoints:
    xyz: np.ndarray  # (N, 3) unit vectors
    weights: np.ndarray  # (N,) sums to 1


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, np.float64) / temperature
    z = np.exp(z - z.max())
    return np.asarray(z / z.sum())


def pool_points(preds: Sequence[FramePrediction], params: AggregateParams) -> PooledPoints:
    """Stack every frame's top-k points with weight frame_weight * softmax(logits / T)."""
    if not preds:
        raise ValueError("need at least one frame prediction")
    kept = [p.truncate(params.top_k) for p in preds]
    if params.frame_weighting == "mass":
        fw = np.array([p.top_mass for p in kept], np.float64)
        fw = fw / fw.sum() if fw.sum() > 0 else np.full(len(kept), 1.0 / len(kept))
    else:
        fw = np.full(len(kept), 1.0 / len(kept))
    xyz = np.concatenate([to_unit_vectors(p.lats, p.lons) for p in kept])
    w = np.concatenate(
        [f * softmax(p.logits, params.temperature) for f, p in zip(fw, kept, strict=True)]
    )
    return PooledPoints(xyz, w / w.sum())


def cluster(
    xyz: np.ndarray, weights: np.ndarray, cluster_km: float, iters: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Leader clustering on the sphere, heaviest points first, then re-centring passes.

    Returns (labels (N,), centres (C, 3) as unit vectors).
    """
    labels = np.empty(len(xyz), np.int64)
    centers: list[np.ndarray] = []
    for i in np.argsort(-weights, kind="stable"):
        if centers:
            d = angular_km(xyz[i : i + 1], np.asarray(centers))[0]
            j = int(np.argmin(d))
            if d[j] <= cluster_km:
                labels[i] = j
                continue
        centers.append(xyz[i])
        labels[i] = len(centers) - 1
    c = _centres(xyz, weights, labels, len(centers))
    for _ in range(iters):
        new = np.argmin(angular_km(xyz, c), axis=1)
        if np.array_equal(new, labels):
            break
        used, labels = np.unique(new, return_inverse=True)
        c = _centres(xyz, weights, labels, len(used))
    return labels, c


def _centres(xyz: np.ndarray, weights: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    sums = np.zeros((n, 3))
    np.add.at(sums, labels, xyz * weights[:, None])
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    for j in np.flatnonzero(norms[:, 0] < 1e-12):  # cancelled out: use the heaviest member
        members = np.flatnonzero(labels == j)
        sums[j] = xyz[members[np.argmax(weights[members])]]
        norms[j] = 1.0
    return np.asarray(sums / norms)


def _radius(dist_km: np.ndarray, weights: np.ndarray, params: AggregateParams) -> float:
    r = weighted_quantile(dist_km, weights, params.confidence_mass)
    return float(min(MAX_DISTANCE_KM, max(params.min_radius_km, r)))


def aggregate(preds: Sequence[FramePrediction], params: AggregateParams) -> Aggregate | None:
    """Best location, confidence radius and alternatives for a set of frames (None if empty)."""
    if not preds:
        return None
    pts = pool_points(preds, params)
    labels, centers = cluster(pts.xyz, pts.weights, params.cluster_km)
    mass = np.bincount(labels, weights=pts.weights, minlength=len(centers))
    order = np.argsort(-mass, kind="stable")
    best = int(order[0])
    lat, lon = from_unit_vectors(centers[best])
    dist_all = angular_km(pts.xyz, centers[best : best + 1])[:, 0]
    best_loc = Location(
        float(lat), float(lon), _radius(dist_all, pts.weights, params), float(mass[best])
    )

    alts: list[Location] = []
    chosen = [centers[best]]
    for j in order[1:]:
        if len(alts) >= params.alternatives or mass[j] <= 0:
            break
        if angular_km(centers[j : j + 1], np.asarray(chosen)).min() <= params.cluster_km:
            continue
        members = labels == j
        d = angular_km(pts.xyz[members], centers[j : j + 1])[:, 0]
        a_lat, a_lon = from_unit_vectors(centers[j])
        alts.append(
            Location(
                float(a_lat), float(a_lon), _radius(d, pts.weights[members], params), float(mass[j])
            )
        )
        chosen.append(centers[j])
    return Aggregate(best_loc, alts, len(preds))
