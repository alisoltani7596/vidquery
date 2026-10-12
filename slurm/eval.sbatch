#!/bin/bash
#SBATCH --job-name=vq-eval
#SBATCH --account=def-branzana
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/vidquery/logs/%x-%j.out
# Retrieval eval with and without the VLM verifier.
# By default vLLM is started inside this job on the same H100 (it gets 45% of memory).
# To use a server started by slurm/vllm.sbatch instead:
#   VLM_URL=$(cat $SCRATCH/vidquery/vllm_endpoint) sbatch slurm/eval.sbatch
#   QUERIES=path/to/queries.jsonl VERIFIER=off sbatch slurm/eval.sbatch
set -euo pipefail
source "${VQ_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
vq_banner

QUERIES="${QUERIES:-$VQ_DATA/queries.jsonl}"
VERIFIER="${VERIFIER:-both}"
STAMP="$(date +%Y%m%d-%H%M)"
[[ -f "$QUERIES" ]] || { echo "No queries file at $QUERIES (see data/queries.example.jsonl)" >&2; exit 1; }

URL="${VLM_URL:-}"
if [[ "$VERIFIER" != "off" && -z "$URL" ]]; then
    trap vq_stop_vllm EXIT
    vq_start_vllm "${VLLM_MEM:-0.45}"
    URL="http://127.0.0.1:$VQ_VLLM_PORT/v1"
fi

vq_exec python scripts/eval_retrieval.py --queries "$QUERIES" --verifier "$VERIFIER" \
    --set "verifier.base_url=${URL:-http://127.0.0.1:$VQ_VLLM_PORT/v1}" \
    --set "verifier.model=$VQ_VLM_MODEL" \
    --out "$VQ_RESULTS/eval-$STAMP.json" --md "$VQ_RESULTS/eval-$STAMP.md"
cp "$VQ_RESULTS/eval-$STAMP.md" "$VQ_RESULTS/eval.md"
