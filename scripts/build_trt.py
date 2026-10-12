#!/usr/bin/env python
"""Build TensorRT engines from the exported ONNX files (FP32, FP16, optional INT8).

    python scripts/build_trt.py --config configs/fir.yaml --components detector image_encoder \
        --precisions fp32 fp16
    # INT8 with entropy calibration on real frames:
    python scripts/build_trt.py --config configs/fir.yaml --precisions int8 \
        --calib $SCRATCH/vidquery/data/MOT17/videos --calib-images 512

Same flags as `vidquery export trt`. Run it on the GPU type you will serve on.
"""

import sys

from vidquery.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["export", "trt", *sys.argv[1:]]))
