#!/usr/bin/env python
"""Qualitative geolocation of the MOT17 videos against data/mot17_locations.json.

    python scripts/geo_mot17.py --videos $SCRATCH/vidquery/data/MOT17/videos \
        --params results/geo-params.yaml --md results/geo-mot17.md --maps results/geo-maps
"""

from vidquery.geo.mot17 import main

if __name__ == "__main__":
    raise SystemExit(main())
