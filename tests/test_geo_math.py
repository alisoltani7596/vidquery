"""Spherical geometry and coarse rounding used by vidquery locate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vidquery.geo.geometry import (
    EARTH_RADIUS_KM,
    coarsen,
    coordinate_decimals,
    haversine_km,
    round_up_sig,
    spherical_mean,
    weighted_quantile,
)

PARIS = (48.8566, 2.3522)
LONDON = (51.5074, -0.1278)


def test_haversine_known_distances() -> None:
    assert float(haversine_km(*PARIS, *LONDON)) == pytest.approx(343.5, abs=1.0)
    assert float(haversine_km(*PARIS, *PARIS)) == pytest.approx(0.0, abs=1e-9)
    # A quarter of the equator and antipodes.
    assert float(haversine_km(0, 0, 0, 90)) == pytest.approx(math.pi * EARTH_RADIUS_KM / 2)
    assert float(haversine_km(10, 20, -10, -160)) == pytest.approx(math.pi * EARTH_RADIUS_KM)
    # Across the antimeridian: 0.2 degrees of longitude on the equator, not 359.8.
    assert float(haversine_km(0, 179.9, 0, -179.9)) == pytest.approx(22.24, abs=0.05)
    # Symmetric and broadcasting.
    d = haversine_km(PARIS[0], PARIS[1], [LONDON[0], PARIS[0]], [LONDON[1], PARIS[1]])
    assert d.shape == (2,) and d[0] == pytest.approx(float(haversine_km(*LONDON, *PARIS)))


def test_spherical_mean() -> None:
    lat, lon = spherical_mean([0, 0], [179, -179])
    assert lat == pytest.approx(0, abs=1e-9) and abs(lon) == pytest.approx(180, abs=1e-9)
    lat, lon = spherical_mean([10, 20], [0, 0], [3, 1])
    assert 10 < lat < 15 and lon == pytest.approx(0, abs=1e-9)
    lat, _ = spherical_mean([89, 89, 89], [0, 120, 240])
    assert lat == pytest.approx(90, abs=1e-6)
    # Antipodal mass cancels: fall back to the heaviest point.
    assert spherical_mean([0, 0], [0, 180], [2, 2 - 1e-12])[1] == pytest.approx(0, abs=1e-6)
    with pytest.raises(ValueError):
        spherical_mean([], [])


def test_weighted_quantile() -> None:
    assert weighted_quantile([1, 2, 3, 4], [1, 1, 1, 1], 0.5) == 2
    assert weighted_quantile([1, 2, 3, 4], [1, 1, 1, 1], 0.51) == 3
    assert weighted_quantile([4, 1, 3, 2], [0, 0, 0, 1], 0.1) == 2
    assert weighted_quantile([5.0], [2.0], 0.9) == 5.0
    assert weighted_quantile([1, 100], [0.7, 0.3], 0.68) == 1
    with pytest.raises(ValueError):
        weighted_quantile([1, 2], [1], 0.5)


def test_round_up_sig_and_decimals() -> None:
    assert round_up_sig(37.2) == 38 and round_up_sig(25) == 25 and round_up_sig(1234) == 1300
    assert round_up_sig(0.123) == pytest.approx(0.13)
    assert [coordinate_decimals(r) for r in (1, 10, 25, 300, 3000, 20000)] == [2, 2, 1, 0, -1, -1]


@pytest.mark.parametrize("radius", [1.0, 12.0, 25.0, 80.0, 300.0, 1500.0, 6000.0])
@pytest.mark.parametrize(
    ("lat", "lon"), [PARIS, (-33.8688, 151.2093), (64.1466, -21.9426), (0.04, 179.97)]
)
def test_coarsen_never_claims_more_precision_than_the_radius(
    lat: float, lon: float, radius: float
) -> None:
    rlat, rlon, rradius = coarsen(lat, lon, radius)
    d = coordinate_decimals(radius)
    # The rounding step is no finer than a tenth of the radius ...
    step_km = 10.0 ** (-d) * 111.19
    assert step_km >= radius / 10 - 1e-9 or d == 2
    # ... the printed coordinates carry no more decimals than that ...
    assert rlat == round(rlat, max(d, 0)) and rlon == round(rlon, max(d, 0))
    # ... and the reported circle contains the whole unrounded circle.
    assert float(haversine_km(lat, lon, rlat, rlon)) + radius <= rradius + 1e-6
    assert -90 <= rlat <= 90 and -180 <= rlon <= 180


def test_coarsen_examples() -> None:
    assert coarsen(*PARIS, 25.0) == (48.9, 2.4, 31.0)
    lat, lon, r = coarsen(*PARIS, 300.0)
    assert (lat, lon) == (49.0, 2.0) and r >= 300
    lat, lon, r = coarsen(*PARIS, 3000.0)
    assert (lat, lon) == (50.0, 0.0) and r >= 3000
    assert np.isfinite(coarsen(89.99, 10.0, 25.0)[2])
