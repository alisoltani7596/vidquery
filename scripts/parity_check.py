#!/usr/bin/env python
"""Compare backend outputs against PyTorch FP32 with per-precision tolerances.

    python scripts/parity_check.py --config configs/fir.yaml \
        --components detector image_encoder \
        --backends onnxruntime tensorrt:fp32 tensorrt:fp16 tensorrt:int8 \
        --images $SCRATCH/vidquery/data/MOT17/videos --out results/parity.json

Exits non-zero when any candidate is outside tolerance. Same flags as `vidquery parity`.
"""

import sys

from vidquery.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["parity", *sys.argv[1:]]))
