#!/usr/bin/env python
"""Benchmark backends: p50/p95 latency, throughput and peak GPU memory per batch size, plus
end-to-end ingest FPS. Writes JSON and a markdown table.

Example (GPU):
    python scripts/benchmark.py --config configs/fir.yaml \
        --backends pytorch:fp32 pytorch:fp16 onnxruntime tensorrt:fp32 tensorrt:fp16 \
        --batch-sizes 1 4 8 16 --video $SCRATCH/vidquery/data/MOT17/videos/MOT17-04.mp4

Example (CPU smoke test, no weights):
    python scripts/benchmark.py --config configs/tiny.yaml --backends pytorch onnxruntime \
        --video synthetic --iters 10
"""

from vidquery.bench import main

if __name__ == "__main__":
    raise SystemExit(main())
