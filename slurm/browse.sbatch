#!/bin/bash
#SBATCH --job-name=vq-browse
#SBATCH --account=def-branzana
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/vidquery/logs/%x-%j.out
# Print the top segments for some queries and save one contact sheet per query (plus optional
# timeline sheets: one frame every EVERY seconds per video) to help write queries.jsonl.
#   sbatch slurm/browse.sbatch "a woman pushing a stroller" "a man in a red jacket"
#   QUERIES=my_queries.txt K=20 sbatch slurm/browse.sbatch
#   TIMELINE=all EVERY=1 sbatch slurm/browse.sbatch
# Sheets go to $OUT (default $VQ_RESULTS/browse); retrieval never uses the verifier here.
set -euo pipefail
source "${VQ_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
vq_banner

ARGS=(--out "${OUT:-$VQ_RESULTS/browse}" -k "${K:-10}")
if [[ -n "${QUERIES:-}" ]]; then ARGS+=(--queries "$QUERIES"); fi
if [[ -n "${TIMELINE:-}" ]]; then
    read -r -a TL <<<"$TIMELINE"
    ARGS+=(--timeline "${TL[@]}" --every "${EVERY:-1}")
fi
read -r -a EXTRA_ARGS <<<"${EXTRA:-}"
vq_exec vidquery browse "$@" "${ARGS[@]}" "${EXTRA_ARGS[@]}"
