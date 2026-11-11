"""GeoCLIPBackend against the real geoclip package, with a randomly initialised CLIP backbone
(no download). Skipped unless the optional ``geo`` extra is installed."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("geoclip")

from vidquery.geo.aggregate import AggregateParams, aggregate
from vidquery.geo.backends import GeoCLIPBackend


@pytest.mark.slow
def test_geoclip_backend_shapes() -> None:
    backend = GeoCLIPBackend(device="cpu", batch_size=2, clip_model=None)
    assert backend.gallery.shape == (100_000, 2)
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (120, 200, 3), dtype=np.uint8) for _ in range(3)]
    preds = backend.predict(frames, top_k=5)
    assert len(preds) == 3
    for p in preds:
        assert p.lats.shape == (5,) and np.all(np.diff(p.logits) <= 1e-6)
        assert np.all(np.abs(p.lats) <= 90) and np.all(np.abs(p.lons) <= 180)
        assert 0 < p.top_mass <= 1
    assert aggregate(preds, AggregateParams(top_k=5)) is not None
