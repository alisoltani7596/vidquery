#!/usr/bin/env python
"""Export detector / image encoder / text encoder to ONNX (dynamic batch, opset >= 17).

    python scripts/export_onnx.py --config configs/fir.yaml \
        --components detector image_encoder text_encoder --exporter dynamo --opset 18

Same flags as `vidquery export onnx`.
"""

import sys

from vidquery.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["export", "onnx", *sys.argv[1:]]))
