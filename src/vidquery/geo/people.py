"""How much of a frame people cover, so person-dominated frames can be left out of geolocation.

A frame filled with people says little about where it was filmed, and leaving those frames out
also keeps the location estimate about the scene rather than the individuals in it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vidquery.types import Detection


def person_coverage(dets: Sequence[Detection], height: int, width: int, grid: int = 128) -> float:
    """Fraction of the frame covered by the union of person boxes (rasterised on a coarse grid)."""
    if not dets or height <= 0 or width <= 0:
        return 0.0
    scale = grid / max(height, width)
    gh, gw = max(1, round(height * scale)), max(1, round(width * scale))
    mask = np.zeros((gh, gw), bool)
    for d in dets:
        x1 = int(np.clip(np.floor(d.x1 * scale), 0, gw))
        y1 = int(np.clip(np.floor(d.y1 * scale), 0, gh))
        x2 = int(np.clip(np.ceil(d.x2 * scale), 0, gw))
        y2 = int(np.clip(np.ceil(d.y2 * scale), 0, gh))
        mask[y1:y2, x1:x2] = True
    return float(mask.mean())
