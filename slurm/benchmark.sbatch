#!/bin/bash
#SBATCH --job-name=vq-bench
#SBATCH --account=def-branzana
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/vidquery/logs/%x-%j.out
# Latency/throughput/memory per backend and batch size + end-to-end ingest FPS.
#   sbatch slurm/benchmark.sbatch
#   BACKENDS="pytorch:fp16 tensorrt:fp16 tensorrt:int8" sbatch slurm/benchmark.sbatch
set -euo pipefail
source "${VQ_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
vq_banner

read -r -a BACKENDS <<<"${BACKENDS:-pytorch:fp32 pytorch:fp16 onnxruntime tensorrt:fp32 tensorrt:fp16}"
VIDEO="${VIDEO:-$VQ_DATA/MOT17/videos/MOT17-04.mp4}"
[[ -f "$VIDEO" ]] || VIDEO=synthetic
STAMP="$(date +%Y%m%d-%H%M)"

vq_exec python scripts/benchmark.py \
    --backends "${BACKENDS[@]}" \
    --components detector image_encoder text_encoder \
    --batch-sizes 1 4 8 16 --warmup 20 --iters 200 \
    --video "$VIDEO" \
    --out "$VQ_RESULTS/benchmark-$STAMP.json" --md "$VQ_RESULTS/benchmark-$STAMP.md"
cp "$VQ_RESULTS/benchmark-$STAMP.md" "$VQ_RESULTS/benchmark.md"
