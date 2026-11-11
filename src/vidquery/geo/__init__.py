"""Visual geolocation: "where was this video filmed?", answered coarsely by design.

See :class:`vidquery.geo.locate.GeoLocator` and README "Where was this filmed?".
"""

from __future__ import annotations

from vidquery.geo.aggregate import Aggregate, AggregateParams, Location, aggregate
from vidquery.geo.backends import FramePrediction, GeoBackend, MockGeoBackend
from vidquery.geo.geometry import coarsen, haversine_km
from vidquery.geo.locate import GeoArea, GeoLocator, LocateResult

__all__ = [
    "Aggregate",
    "AggregateParams",
    "FramePrediction",
    "GeoArea",
    "GeoBackend",
    "GeoLocator",
    "LocateResult",
    "Location",
    "MockGeoBackend",
    "aggregate",
    "coarsen",
    "haversine_km",
]
