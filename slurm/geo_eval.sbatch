#!/bin/bash
#SBATCH --job-name=vq-geo-eval
#SBATCH --account=def-branzana
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/vidquery/logs/%x-%j.out
# Geolocation benchmark and MOT17 qualitative run (vidquery locate).
#
# Steps (STEPS="data predict tune test mot17" by default):
#   data     download the OSV-5M test split (CC BY-SA 4.0, ~9.4 GB in 6 files) and write fixed
#            val/test manifests; also caches the CLIP ViT-L/14 backbone. Online.
#   predict  GeoCLIP top-50 for every manifest image, cached in one .npz.
#   tune     grid search of the aggregation settings on val only -> results/geo-params.yaml
#   test     the test split, once, with the frozen settings (refuses to overwrite)
#   mot17    locate each MOT17 video (+ VLM second opinion unless VLM=0), maps as HTML
#
#   sbatch slurm/geo_eval.sbatch
#   STEPS="mot17" VLM=0 sbatch slurm/geo_eval.sbatch
#   OSV_SHARDS="00" sbatch slurm/geo_eval.sbatch      # fewer image zips (2.2 GB each)
#
# Needs the geo packages in the venv image: STEPS="venv weights" sbatch slurm/setup.sbatch
set -euo pipefail
source "${VQ_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
vq_banner

read -ra STEP_LIST <<<"${STEPS:-data predict tune test mot17}"
GEO_DATA="${GEO_DATA:-$VQ_DATA/osv5m}"
PREDS="$GEO_DATA/geoclip-top50.npz"
MANIFESTS=(--manifest "$GEO_DATA/manifest-images.jsonl" --manifest "$GEO_DATA/manifest-sequences.jsonl")
PARAMS="$VQ_RESULTS/geo-params.yaml"
# The detector only decides which frames are mostly people; the PyTorch backend avoids a
# dependency on prebuilt engines.
GEO_SET=(--set detector.backend=pytorch --set detector.precision=fp32)
has() { [[ " ${STEP_LIST[*]} " == *" $1 "* ]]; }

vq_exec python -c 'import geoclip, pandas; print("geoclip", geoclip.__file__)' || {
    echo 'geoclip is not in the venv image; rebuild it: STEPS="venv weights" sbatch slurm/setup.sbatch' >&2
    exit 1
}

if has data; then
    (
        export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
        read -ra shards <<<"${OSV_SHARDS:-00 01 02 03 04}"
        vq_exec python scripts/osv5m_subset.py --out-dir "$GEO_DATA" --download \
            --shards "${shards[@]}" --val "${GEO_VAL:-1000}" --test "${GEO_TEST:-2000}" --seed 0
        vq_exec python -c 'from huggingface_hub import snapshot_download as s; print(s("openai/clip-vit-large-patch14", allow_patterns=["*.json", "*.safetensors", "*.txt"]))'
    )
fi

if has predict; then
    vq_exec python scripts/geo_eval.py predict "${MANIFESTS[@]}" --out "$PREDS" --top-k 50
fi

if has tune; then
    vq_exec python scripts/geo_eval.py tune "${MANIFESTS[@]}" --preds "$PREDS" --split val \
        --params-out "$PARAMS" --md "$VQ_RESULTS/geo-val.md"
fi

if has test; then
    # Report the test split once: scripts/geo_eval.py refuses to overwrite geo-eval.md.
    vq_exec python scripts/geo_eval.py test "${MANIFESTS[@]}" --preds "$PREDS" --split test \
        --params "$PARAMS" --md "$VQ_RESULTS/geo-eval.md" --json "$VQ_RESULTS/geo-eval.json"
fi

if has mot17; then
    VLM_ARGS=()
    if [[ "${VLM:-1}" == "1" ]]; then
        URL="${VLM_URL:-}"
        if [[ -z "$URL" ]]; then
            trap vq_stop_vllm EXIT
            vq_start_vllm "${VLLM_MEM:-0.45}"
            URL="http://127.0.0.1:$VQ_VLLM_PORT/v1"
        fi
        VLM_ARGS=(--vlm --set "verifier.base_url=$URL" --set "verifier.model=$VQ_VLM_MODEL")
    fi
    PARAM_ARGS=()
    [[ -f "$PARAMS" ]] && PARAM_ARGS=(--params "$PARAMS")
    vq_exec python scripts/geo_mot17.py --videos "${MOT17_VIDEOS:-$VQ_DATA/MOT17/videos}" \
        "${PARAM_ARGS[@]}" "${VLM_ARGS[@]}" "${GEO_SET[@]}" \
        --md "$VQ_RESULTS/geo-mot17.md" --json "$VQ_RESULTS/geo-mot17.json" \
        --maps "$VQ_RESULTS/geo-maps"
fi
echo "geo_eval done: ${STEP_LIST[*]} (results in $VQ_RESULTS)"
