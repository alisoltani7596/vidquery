"""Spherical geometry helpers: great-circle distance, weighted means and coarse rounding."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEGREE = math.pi * EARTH_RADIUS_KM / 180.0  # ~111.2 km along a meridian
MAX_DISTANCE_KM = math.pi * EARTH_RADIUS_KM  # half the circumference

FloatArray = NDArray[np.float64]


def haversine_km(lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike) -> FloatArray:
    """Great-circle distance in km between (lat1, lon1) and (lat2, lon2), in degrees.

    Inputs broadcast like numpy arrays; the result is always an ndarray (0-d for scalars).
    """
    p1, p2 = np.radians(np.asarray(lat1, np.float64)), np.radians(np.asarray(lat2, np.float64))
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2, np.float64) - np.asarray(lon1, np.float64))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return np.asarray(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def to_unit_vectors(lat: ArrayLike, lon: ArrayLike) -> FloatArray:
    """(N,) degrees -> (N, 3) points on the unit sphere."""
    phi = np.radians(np.asarray(lat, np.float64))
    lmb = np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(phi) * np.cos(lmb), np.cos(phi) * np.sin(lmb), np.sin(phi)], axis=-1)


def from_unit_vectors(xyz: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """(..., 3) vectors (any length > 0) -> latitude, longitude in degrees."""
    v = np.asarray(xyz, np.float64)
    norm = np.linalg.norm(v, axis=-1)
    lat = np.degrees(np.arcsin(np.clip(v[..., 2] / np.maximum(norm, 1e-300), -1.0, 1.0)))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return np.asarray(lat), np.asarray(lon)


def angular_km(xyz: FloatArray, centers: FloatArray) -> FloatArray:
    """Distances in km between unit vectors (N, 3) and unit centres (C, 3) -> (N, C)."""
    dots = np.clip(xyz @ centers.T, -1.0, 1.0)
    return np.asarray(EARTH_RADIUS_KM * np.arccos(dots))


def spherical_mean(
    lat: ArrayLike, lon: ArrayLike, weights: ArrayLike | None = None
) -> tuple[float, float]:
    """Weighted mean position on the sphere (handles the antimeridian and the poles).

    Falls back to the heaviest point when the weighted vectors cancel out (antipodal mass).
    """
    xyz = to_unit_vectors(lat, lon).reshape(-1, 3)
    w = np.ones(len(xyz)) if weights is None else np.asarray(weights, np.float64).reshape(-1)
    if len(xyz) == 0 or len(w) != len(xyz) or w.sum() <= 0:
        raise ValueError("spherical_mean needs at least one point with positive weight")
    v = (xyz * w[:, None]).sum(axis=0)
    if np.linalg.norm(v) < 1e-9 * w.sum():
        v = xyz[int(np.argmax(w))]
    la, lo = from_unit_vectors(v)
    return float(la), float(lo)


def weighted_quantile(values: ArrayLike, weights: ArrayLike, q: float) -> float:
    """Smallest value v such that the weight of {values <= v} is at least q of the total."""
    v = np.asarray(values, np.float64).reshape(-1)
    w = np.asarray(weights, np.float64).reshape(-1)
    if len(v) == 0 or len(v) != len(w):
        raise ValueError("values and weights must be non-empty and the same length")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    order = np.argsort(v, kind="stable")
    cum = np.cumsum(w[order])
    if cum[-1] <= 0:
        raise ValueError("weights must sum to a positive number")
    idx = int(np.searchsorted(cum, q * cum[-1] * (1 - 1e-12), side="left"))
    return float(v[order][min(idx, len(v) - 1)])


def round_up_sig(x: float, digits: int = 2) -> float:
    """Round ``x`` up to ``digits`` significant figures (37.2 -> 38, 1234 -> 1300)."""
    if x <= 0:
        return 0.0
    step = 10.0 ** (math.floor(math.log10(x)) - digits + 1)
    return float(round(math.ceil(x / step - 1e-9) * step, 10))


def coordinate_decimals(radius_km: float) -> int:
    """Decimals for coordinates whose rounding step (10^-d degrees) is no coarser than the radius
    and no finer than a tenth of it: 25 km -> 1 (0.1 deg ~ 11 km), 300 km -> 0 (1 deg)."""
    if radius_km <= 0:
        raise ValueError("radius_km must be positive")
    return int(min(2, max(-1, math.ceil(math.log10(KM_PER_DEGREE / radius_km)))))


def coarsen(lat: float, lon: float, radius_km: float) -> tuple[float, float, float]:
    """Round a position to the precision its radius supports.

    Returns (lat, lon, radius_km). The coordinates are rounded to :func:`coordinate_decimals`;
    the radius grows by the distance the rounding moved the point and is then rounded *up* to two
    significant figures, so the reported circle always contains the unrounded one.
    """
    d = coordinate_decimals(radius_km)
    rlat = float(min(90.0, max(-90.0, round(lat, d))))
    rlon = float(round(lon, d))
    if rlon > 180.0:
        rlon -= 360.0
    elif rlon < -180.0:
        rlon += 360.0
    shift = float(haversine_km(lat, lon, rlat, rlon))
    radius = min(MAX_DISTANCE_KM, round_up_sig(radius_km + shift))
    return rlat, rlon, radius
