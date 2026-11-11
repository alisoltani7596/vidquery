#!/usr/bin/env python
"""Geolocation benchmark: cache predictions, tune aggregation on val, report test once.

    python scripts/geo_eval.py predict --manifest data/osv5m/manifest-images.jsonl --out preds.npz
    python scripts/geo_eval.py tune --manifest ... --preds preds.npz \
        --params-out results/geo-params.yaml --md results/geo-val.md
    python scripts/geo_eval.py test --manifest ... --preds preds.npz \
        --params results/geo-params.yaml --md results/geo-eval.md

See vidquery.geo.evaluation for the manifest format and the metrics.
"""

from vidquery.geo.evaluation import main

if __name__ == "__main__":
    raise SystemExit(main())
