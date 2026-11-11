"""Video-level aggregation of per-frame location predictions."""

from __future__ import annotations

import numpy as np
import pytest

from vidquery.config import GeoConfig
from vidquery.geo.aggregate import AggregateParams, aggregate, cluster, pool_points
from vidquery.geo.backends import MOCK_ANCHORS, FramePrediction, MockGeoBackend, mock_anchor
from vidquery.geo.geometry import haversine_km, to_unit_vectors

PARIS, TOKYO = MOCK_ANCHORS[0], MOCK_ANCHORS[2]


def frame(
    points: list[tuple[float, float]], logits: list[float], extra_mass: float = 1.5
) -> FramePrediction:
    arr = np.asarray(logits, np.float32)
    return FramePrediction(
        np.array([p[0] for p in points]),
        np.array([p[1] for p in points]),
        arr,
        float(np.log(np.exp(arr.astype(np.float64)).sum() * extra_mass)),
    )


def near(center: tuple[float, float], n: int, km: float = 10.0) -> list[tuple[float, float]]:
    return [
        (center[0] + km / 111.0 * np.sin(k), center[1] + km / 111.0 * np.cos(k)) for k in range(n)
    ]


def test_majority_cluster_wins_with_alternatives() -> None:
    preds = [frame(near(PARIS, 4), [5, 4.9, 4.8, 4.7]) for _ in range(7)]
    preds += [frame(near(TOKYO, 4), [5, 4.9, 4.8, 4.7]) for _ in range(3)]
    agg = aggregate(preds, AggregateParams(top_k=4, cluster_km=100, min_radius_km=25))
    assert agg is not None and agg.frames == 10
    assert float(haversine_km(agg.best.lat, agg.best.lon, *PARIS)) < 15
    assert agg.best.share == pytest.approx(0.7, abs=1e-6)
    assert len(agg.alternatives) == 1
    alt = agg.alternatives[0]
    assert float(haversine_km(alt.lat, alt.lon, *TOKYO)) < 15
    assert alt.share == pytest.approx(0.3, abs=1e-6)
    # 70% of the mass is near Paris, so the 68% radius stays local (but never below the floor).
    assert 25 <= agg.best.radius_km < 50


def test_split_decision_gives_a_large_radius() -> None:
    preds = [frame(near(PARIS, 3), [3, 3, 3]), frame(near(TOKYO, 3), [3, 3, 3])]
    agg = aggregate(preds, AggregateParams(top_k=3, cluster_km=100, confidence_mass=0.68))
    assert agg is not None
    assert agg.best.share == pytest.approx(0.5, abs=1e-6)
    # Covering 68% of the mass needs the other city: ~9,700 km.
    assert agg.best.radius_km > 9000


def test_single_point_uses_min_radius() -> None:
    agg = aggregate([frame([PARIS], [1.0])], AggregateParams(top_k=1, min_radius_km=25))
    assert agg is not None
    assert agg.best.radius_km == 25 and agg.best.share == pytest.approx(1.0)
    assert (agg.best.lat, agg.best.lon) == pytest.approx(PARIS)
    assert aggregate([], AggregateParams()) is None


def test_temperature_and_top_k() -> None:
    f = frame([PARIS, TOKYO], [10.0, 8.0])
    sharp = aggregate([f], AggregateParams(top_k=2, temperature=0.25, cluster_km=100))
    flat = aggregate([f], AggregateParams(top_k=2, temperature=10.0, cluster_km=100))
    only1 = aggregate([f], AggregateParams(top_k=1, temperature=10.0, cluster_km=100))
    assert sharp and flat and only1
    assert sharp.best.share > 0.99 and 0.5 < flat.best.share < 0.6
    assert only1.best.share == pytest.approx(1.0) and not only1.alternatives


def test_mass_weighting_trusts_confident_frames() -> None:
    confident = frame([TOKYO], [20.0], extra_mass=1.01)  # holds ~99% of its gallery mass
    vague = [frame([PARIS], [1.0], extra_mass=50.0) for _ in range(3)]  # ~2% each
    uniform = aggregate([confident, *vague], AggregateParams(top_k=1, frame_weighting="uniform"))
    mass = aggregate([confident, *vague], AggregateParams(top_k=1, frame_weighting="mass"))
    assert uniform and mass
    assert float(haversine_km(uniform.best.lat, uniform.best.lon, *PARIS)) < 1
    assert float(haversine_km(mass.best.lat, mass.best.lon, *TOKYO)) < 1


def test_clustering_across_the_antimeridian() -> None:
    pts = [(-17.7, 179.95), (-17.7, -179.95), (-17.75, 179.9)]
    agg = aggregate([frame(pts, [2, 2, 2])], AggregateParams(top_k=3, cluster_km=50))
    assert agg is not None and agg.best.share == pytest.approx(1.0)
    assert abs(agg.best.lon) > 179.8 and not agg.alternatives


def test_cluster_labels_and_alternative_cap() -> None:
    cities = [MOCK_ANCHORS[i % 4] for i in range(8)]
    xyz = to_unit_vectors([c[0] for c in cities], [c[1] for c in cities])
    labels, centers = cluster(xyz, np.full(8, 1 / 8), cluster_km=200)
    assert len(centers) == 4 and len(set(labels.tolist())) == 4
    preds = [frame(MOCK_ANCHORS[:4], [4, 3, 2, 1])]
    agg = aggregate(preds, AggregateParams(top_k=4, cluster_km=200, alternatives=2))
    assert agg is not None and len(agg.alternatives) == 2
    shares = [agg.best.share] + [a.share for a in agg.alternatives]
    assert shares == sorted(shares, reverse=True) and sum(shares) < 1


def test_pool_points_weights_sum_to_one() -> None:
    preds = MockGeoBackend().predict([np.full((8, 8, 3), v, np.uint8) for v in (10, 100, 200)], 8)
    pts = pool_points(preds, AggregateParams(top_k=5, frame_weighting="mass"))
    assert pts.xyz.shape == (15, 3) and pts.weights.sum() == pytest.approx(1.0)


def test_mock_backend_is_deterministic() -> None:
    img = np.full((16, 16, 3), 130, np.uint8)
    assert mock_anchor(img) == 2
    a, b = MockGeoBackend().predict([img], 10)[0], MockGeoBackend().predict([img], 10)[0]
    np.testing.assert_array_equal(a.lats, b.lats)
    assert float(haversine_km(a.lats[0], a.lons[0], *TOKYO)) < 1
    assert 0 < a.top_mass < 1 and a.truncate(3).logits.shape == (3,)


def test_params_mirror_geo_config() -> None:
    cfg = GeoConfig(top_k=7, cluster_km=50, frame_weighting="mass")
    p = AggregateParams.from_config(cfg)
    assert (p.top_k, p.cluster_km, p.frame_weighting) == (7, 50, "mass")
    assert set(p.to_dict()) <= set(GeoConfig.model_fields)
    assert p.with_overrides(top_k=3).top_k == 3
