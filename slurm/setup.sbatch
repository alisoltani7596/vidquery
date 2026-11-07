#!/bin/bash
#SBATCH --job-name=vq-setup
#SBATCH --account=def-branzana
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/vidquery/logs/%x-%j.out
# One-time setup in a CPU job: container pulls (SIF builds are CPU heavy), venv, weights, MOT17
# unzip + mp4 conversion, CPU tests. Fir compute nodes have internet access.
#   mkdir -p $SCRATCH/vidquery/logs && sbatch slurm/setup.sbatch    # from the repo root
#   STEPS="venv weights" sbatch slurm/setup.sbatch                  # only some setup steps
set -euo pipefail
export VQ_REPO="${VQ_REPO:-$SLURM_SUBMIT_DIR}" VQ_ONLINE=1
cd "$VQ_REPO"

read -ra steps <<<"${STEPS:-containers venv weights}"
bash slurm/setup_env.sh "${steps[@]}"
bash scripts/download_mot17.sh
# Last, so a test failure can't block the data preparation above.
bash slurm/setup_env.sh check
