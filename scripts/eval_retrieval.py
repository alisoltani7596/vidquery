#!/usr/bin/env python
"""Retrieval evaluation: recall@1/5/10, segment-level hit rate, precision and MRR, with and
without the VLM verifier. Reads a queries.jsonl with lines like

    {"query": "man in a blue shirt walking left", "video_id": "MOT17-04", "gt": [[12.0, 18.5]]}

Example:
    python scripts/eval_retrieval.py --config configs/fir.yaml --queries data/queries.jsonl \
        --verifier both --out results/eval.json --md results/eval.md
"""

from vidquery.evaluation import main

if __name__ == "__main__":
    raise SystemExit(main())
