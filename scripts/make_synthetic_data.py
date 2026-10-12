#!/usr/bin/env python
"""Write a synthetic video and a matching queries.jsonl for weight-free demos and CI.

Coloured rectangles ("people") walk across a grey background; the mock configuration
(configs/mock.yaml: colour-blob detector + colour encoder) can search them by colour:

    python scripts/make_synthetic_data.py --out data/synthetic
    vidquery ingest data/synthetic/synthetic.mp4 -c configs/mock.yaml
    vidquery query "person in red" -c configs/mock.yaml
    vidquery demo "person in blue" -c configs/mock.yaml --out results/demo.gif
    python scripts/eval_retrieval.py -c configs/mock.yaml --queries data/synthetic/queries.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vidquery.synthetic import SCRIPT, make_synthetic_video


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--size", nargs=2, type=int, default=[240, 320], metavar=("H", "W"))
    args = p.parse_args()

    out = Path(args.out)
    video = make_synthetic_video(out / "synthetic.mp4", fps=args.fps, size=tuple(args.size))
    with open(out / "queries.jsonl", "w") as fh:
        for i, actor in enumerate(SCRIPT):
            row = {
                "id": f"q{i}",
                "query": f"a person in {actor.color_name}",
                "video_id": "synthetic",
                "gt": [[actor.start, actor.end]],
            }
            fh.write(json.dumps(row) + "\n")
    print(f"Wrote {video} and {out / 'queries.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
