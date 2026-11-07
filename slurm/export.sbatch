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
#   INT8=1 sbatch slurm/export.sbatch        # also build INT8 engines and report their parity
#                                            # (experimental: never fails the job, not deployed)
set -euo pipefail
source "${VQ_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
vq_banner

CALIB="${CALIB:-$VQ_DATA/MOT17/videos}"
PRECISIONS=(fp32 fp16)
[[ "${INT8:-0}" == "1" ]] && PRECISIONS+=(int8)
# Keep going after a failed check so one run reports everything; exit non-zero at the end.
status=0

# torch/TensorRT/ONNX Runtime really on the GPU (ORT otherwise falls back to CPU quietly).
vq_exec python "$VQ_REPO/scripts/gpu_check.py" | tee "$VQ_RESULTS/gpu-check.txt" || status=1

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
REPORT="$VQ_RESULTS/parity.md"
printf '# Parity vs PyTorch FP32 (%s, job %s)\n\n' "$(date -I)" "$SLURM_JOB_ID" >"$REPORT"
vq_exec vidquery parity --components detector image_encoder \
    --backends onnxruntime tensorrt:fp32 tensorrt:fp16 "${PARITY_IMAGES[@]}" --device cuda \
    --out "$VQ_RESULTS/parity.json" | tee -a "$REPORT" || status=1
vq_exec vidquery parity --components text_encoder --device cuda \
    --backends onnxruntime tensorrt:fp32 tensorrt:fp16 \
    --out "$VQ_RESULTS/parity-text.json" | tee -a "$REPORT" || status=1
if [[ "${INT8:-0}" == "1" ]]; then
    # INT8 is not in the deployed config; report it, but don't fail the job on it.
    printf '\n## INT8 (experimental, not deployed)\n\n' >>"$REPORT"
    vq_exec vidquery parity --components detector image_encoder --backends tensorrt:int8 \
        "${PARITY_IMAGES[@]}" --device cuda --out "$VQ_RESULTS/parity-int8.json" \
        | tee -a "$REPORT" || echo "INT8 parity is below the bar (reported only)"
fi

vq_exec python -m pytest -q -m gpu "$VQ_REPO/tests" | tee "$VQ_RESULTS/gpu-tests.txt" || status=1
echo "export job finished with status $status (parity report: $VQ_RESULTS/parity.md)"
exit "$status"
