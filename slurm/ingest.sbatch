#!/bin/bash
#SBATCH --job-name=vq-ingest
#SBATCH --account=def-branzana
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/vidquery/logs/%x-%j.out
# Ingest a dataset (all videos / image sequences in DATASET) into the index from the config.
#   sbatch slurm/ingest.sbatch                                   # MOT17 mp4s
#   DATASET=$SCRATCH/vidquery/data/MOT17/train sbatch slurm/ingest.sbatch   # raw sequences
#   EXTRA="--set detector.backend=pytorch" sbatch slurm/ingest.sbatch
set -euo pipefail
source "${VQ_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
vq_banner

DATASET="${DATASET:-$VQ_DATA/MOT17/videos}"
mapfile -t ITEMS < <(find "$DATASET" -mindepth 1 -maxdepth 1 \( -name '*.mp4' -o -name '*.avi' \
    -o -name '*.mkv' -o -name '*.webm' -o -type d \) | sort)
if [[ ${#ITEMS[@]} -eq 0 ]]; then
    echo "Nothing to ingest in $DATASET" >&2
    exit 1
fi
echo "Ingesting ${#ITEMS[@]} items from $DATASET"
read -r -a EXTRA_ARGS <<<"${EXTRA:-}"
vq_exec vidquery ingest "${ITEMS[@]}" --json "${EXTRA_ARGS[@]}" \
    | tee "$VQ_RESULTS/ingest-${SLURM_JOB_ID}.jsonl"
vq_exec vidquery videos "${EXTRA_ARGS[@]}"
