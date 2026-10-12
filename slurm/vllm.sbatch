#!/bin/bash
#SBATCH --job-name=vq-vllm
#SBATCH --account=def-branzana
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/vidquery/logs/%x-%j.out
# Standalone vLLM server (OpenAI-compatible API) for the verifier.
# Writes its URL to $SCRATCH/vidquery/vllm_endpoint once healthy, then serves until the
# time limit. Other jobs on the cluster network can reach it:
#   JOB=$(sbatch --parsable slurm/vllm.sbatch)
#   # wait until $SCRATCH/vidquery/vllm_endpoint appears, then:
#   VLM_URL=$(cat $SCRATCH/vidquery/vllm_endpoint) sbatch slurm/eval.sbatch
set -euo pipefail
source "${VQ_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
vq_banner

ENDPOINT_FILE="$VQ_ROOT/vllm_endpoint"
rm -f "$ENDPOINT_FILE"
trap 'vq_stop_vllm; rm -f "$ENDPOINT_FILE"' EXIT
vq_start_vllm "${VLLM_MEM:-0.85}"
echo "$VQ_VLLM_URL" >"$ENDPOINT_FILE"
echo "Serving $VQ_VLM_MODEL at $VQ_VLLM_URL (endpoint file: $ENDPOINT_FILE)"
wait "$VQ_VLLM_PID"
