#!/bin/bash
#SBATCH --job-name=vq-export
#SBATCH --account=def-branzana
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/vidquery/logs/%x-%j.out
# Export ONNX, build TensorRT engines (FP32/FP16, optional INT8), run parity + GPU tests.
#   sbatch slurm/export.sbatch               # from the repo root
#   INT8=1 sbatch slurm/export.sbatch        # also build and check INT8 engines
set -euo pipefail
source "${VQ_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
vq_banner

CALIB="${CALIB:-$VQ_DATA/MOT17/videos}"
PRECISIONS=(fp32 fp16)
CANDIDATES=(onnxruntime tensorrt:fp32 tensorrt:fp16)
if [[ "${INT8:-0}" == "1" ]]; then
    PRECISIONS+=(int8)
    CANDIDATES+=(tensorrt:int8)
fi

vq_exec vidquery export onnx --components detector image_encoder text_encoder \
    --exporter "${EXPORTER:-dynamo}" --max-batch 64
vq_exec vidquery export trt --components detector image_encoder \
    --precisions "${PRECISIONS[@]}" --opt-batch 8 --max-batch 32 \
    --calib "$CALIB" --calib-images 512 --int8-method "${INT8_METHOD:-calibrator}" \
    --timing-cache "$VQ_ROOT/artifacts/trt/timing.cache"
# Query-time text tower (int64 token ids; no INT8 calibration for text).
vq_exec vidquery export trt --components text_encoder --precisions fp32 fp16 \
    --opt-batch 1 --max-batch 16 --timing-cache "$VQ_ROOT/artifacts/trt/timing.cache"

PARITY_IMAGES=()
[[ -d "$CALIB" ]] && PARITY_IMAGES=(--images "$CALIB" --num-images 64)
# Keep going after a failed check so one run reports everything; exit non-zero at the end.
status=0
vq_exec vidquery parity --components detector image_encoder \
    --backends "${CANDIDATES[@]}" "${PARITY_IMAGES[@]}" --device cuda \
    --out "$VQ_RESULTS/parity.json" | tee "$VQ_RESULTS/parity.md" || status=1
vq_exec vidquery parity --components text_encoder --device cuda \
    --backends onnxruntime tensorrt:fp32 tensorrt:fp16 \
    --out "$VQ_RESULTS/parity-text.json" | tee -a "$VQ_RESULTS/parity.md" || status=1

vq_exec python -m pytest -q -m gpu "$VQ_REPO/tests" | tee "$VQ_RESULTS/gpu-tests.txt" || status=1
echo "export job finished with status $status (parity report: $VQ_RESULTS/parity.md)"
exit "$status"
