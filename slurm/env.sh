#!/usr/bin/env bash
# Shared settings for vidquery on the Digital Research Alliance "Fir" cluster.
# Source it (do not execute):   source slurm/env.sh
#
# Everything heavy (containers, venv, weights, caches, data, results) lives on $SCRATCH.
# Override any VQ_* variable in your environment before sourcing.

if [[ -z "${SCRATCH:-}" ]]; then
    echo "env.sh: \$SCRATCH is not set (are you on an Alliance cluster?)" >&2
    return 1 2>/dev/null || exit 1
fi

# --- who / where -----------------------------------------------------------------------
export VQ_ACCOUNT="${VQ_ACCOUNT:-def-branzana}"
export VQ_REPO="${VQ_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit jobs from the repo root
export VQ_ROOT="${VQ_ROOT:-$SCRATCH/vidquery}"
export VQ_DATA="${VQ_DATA:-$VQ_ROOT/data}"
export VQ_RESULTS="${VQ_RESULTS:-$VQ_ROOT/results}"
export VQ_LOGS="${VQ_LOGS:-$VQ_ROOT/logs}"
export VQ_WEIGHTS="${VQ_WEIGHTS:-$VQ_ROOT/weights}"
export VQ_CONTAINERS="${VQ_CONTAINERS:-$VQ_ROOT/containers}"

# --- containers (Docker is not available on the cluster; Apptainer pulls OCI images) ---
# 26.03-py3: TensorRT 10.16, CUDA 13.x. Newer tags ship TensorRT 11 (see CLAUDE.md).
export VQ_TRT_TAG="${VQ_TRT_TAG:-26.03-py3}"
export VQ_TRT_IMAGE="${VQ_TRT_IMAGE:-docker://nvcr.io/nvidia/tensorrt:$VQ_TRT_TAG}"
export VQ_TRT_SIF="${VQ_TRT_SIF:-$VQ_CONTAINERS/tensorrt_$VQ_TRT_TAG.sif}"
export VQ_TORCH_INDEX="${VQ_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
# vLLM >= 0.17 is needed for Qwen3.5.
export VQ_VLLM_TAG="${VQ_VLLM_TAG:-v0.17.1}"
export VQ_VLLM_IMAGE="${VQ_VLLM_IMAGE:-docker://vllm/vllm-openai:$VQ_VLLM_TAG}"
export VQ_VLLM_SIF="${VQ_VLLM_SIF:-$VQ_CONTAINERS/vllm-openai_$VQ_VLLM_TAG.sif}"
# Python venv created *inside* the TensorRT container (sees its tensorrt package) and packed
# into one SquashFS image: as plain files it is ~47K entries against a 1M /scratch file quota.
# Jobs mount the image read-only at $VQ_VENV; setup_env.sh venv rebuilds it.
export VQ_VENV_IMG="${VQ_VENV_IMG:-$VQ_ROOT/venv-trt-$VQ_TRT_TAG.sqfs}"
export VQ_VENV="${VQ_VENV:-/opt/venv}"   # path inside the container

# --- models ------------------------------------------------------------------------------
export VQ_CONFIG="${VQ_CONFIG:-$VQ_REPO/configs/fir.yaml}"
export VQ_SIGLIP_MODEL="${VQ_SIGLIP_MODEL:-google/siglip2-so400m-patch16-384}"
export VQ_YOLO_WEIGHTS="${VQ_YOLO_WEIGHTS:-yolo26s.pt}"
export VQ_VLM_MODEL="${VQ_VLM_MODEL:-Qwen/Qwen3.5-4B}"
export VQ_VLLM_PORT="${VQ_VLLM_PORT:-8000}"

# --- caches on $SCRATCH (home quota is small) --------------------------------------------
export HF_HOME="$VQ_ROOT/cache/huggingface"
export TORCH_HOME="$VQ_ROOT/cache/torch"
export XDG_CACHE_HOME="$VQ_ROOT/cache/xdg"
export PIP_CACHE_DIR="$VQ_ROOT/cache/pip"
export YOLO_CONFIG_DIR="$VQ_ROOT/cache/ultralytics"
export VLLM_CACHE_ROOT="$VQ_ROOT/cache/vllm"
export TRITON_CACHE_DIR="$VQ_ROOT/cache/triton"
export APPTAINER_CACHEDIR="$VQ_ROOT/cache/apptainer"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-$VQ_ROOT/cache/apptainer-tmp}"
export VIDQUERY_CONFIG="$VQ_CONFIG"
export YOLO_AUTOINSTALL=false   # never let Ultralytics pip-install at runtime
if [[ -n "${SLURM_JOB_ID:-}" && "${VQ_ONLINE:-0}" != "1" ]]; then
    # Compute nodes do have internet, but runs should only use what is already cached, so a
    # benchmark can't silently fetch something new. VQ_ONLINE=1 lifts this (setup.sbatch).
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

mkdir -p "$VQ_DATA" "$VQ_RESULTS" "$VQ_LOGS" "$VQ_WEIGHTS" "$VQ_CONTAINERS" \
    "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$YOLO_CONFIG_DIR" \
    "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer 2>/dev/null || true
fi

_VQ_ENV_VARS=(HF_HOME TORCH_HOME XDG_CACHE_HOME PIP_CACHE_DIR YOLO_CONFIG_DIR VLLM_CACHE_ROOT
    TRITON_CACHE_DIR VIDQUERY_CONFIG YOLO_AUTOINSTALL HF_HUB_OFFLINE TRANSFORMERS_OFFLINE SCRATCH VQ_ROOT
    VQ_DATA VQ_RESULTS VQ_WEIGHTS VQ_REPO SLURM_JOB_ID SLURM_TMPDIR CUDA_VISIBLE_DEVICES)

# Write the variables containers need to an env file (we run with --cleanenv so host
# modules such as StdEnv do not leak PYTHONPATH & co. into the container).
_vq_env_file() {
    local f="${SLURM_TMPDIR:-$VQ_ROOT/cache}/vq-container-${SLURM_JOB_ID:-login}.env"
    : >"$f"
    local v
    for v in "${_VQ_ENV_VARS[@]}"; do
        [[ -n "${!v:-}" ]] && printf '%s=%s\n' "$v" "${!v}" >>"$f"
    done
    echo "$f"
}

_vq_binds() {
    local binds="$VQ_ROOT,$VQ_REPO"
    [[ -d /scratch ]] && binds="$binds,/scratch"
    [[ -d /project ]] && binds="$binds,/project"
    [[ -n "${SLURM_TMPDIR:-}" ]] && binds="$binds,$SLURM_TMPDIR"
    # The venv: a writable directory while setup_env.sh builds it (VQ_VENV_DIR), else the
    # read-only image (gzip SquashFS; Fir's Apptainer can't read zstd images).
    if [[ -n "${VQ_VENV_DIR:-}" ]]; then
        binds="$binds,$VQ_VENV_DIR:$VQ_VENV"
    elif [[ -f "$VQ_VENV_IMG" ]]; then
        binds="$binds,$VQ_VENV_IMG:$VQ_VENV:image-src=/"
    fi
    echo "$binds"
}

# Run a command inside the TensorRT container with the vidquery venv active.
#   vq_exec vidquery export onnx
vq_exec() {
    apptainer exec --nv --cleanenv --env-file "$(_vq_env_file)" --bind "$(_vq_binds)" \
        --pwd "$VQ_REPO" "$VQ_TRT_SIF" \
        bash -c 'source "$0/bin/activate" && exec "$@"' "$VQ_VENV" "$@"
}

# Run a command inside the vLLM container.
vq_vllm_exec() {
    apptainer exec --nv --cleanenv --env-file "$(_vq_env_file)" --bind "$(_vq_binds)" \
        "$VQ_VLLM_SIF" "$@"
}

# Start vLLM in the background; sets VQ_VLLM_PID and VQ_VLLM_URL. Waits until healthy.
#   vq_start_vllm 0.45   # fraction of GPU memory for vLLM
vq_start_vllm() {
    local mem_frac="${1:-0.85}"
    local log="$VQ_LOGS/vllm-${SLURM_JOB_ID:-local}.log"
    echo "Starting vLLM ($VQ_VLM_MODEL) on port $VQ_VLLM_PORT, log: $log"
    vq_vllm_exec vllm serve "$VQ_VLM_MODEL" \
        --host 0.0.0.0 --port "$VQ_VLLM_PORT" \
        --served-model-name "$VQ_VLM_MODEL" \
        --max-model-len 8192 \
        --max-num-seqs 64 \
        --gpu-memory-utilization "$mem_frac" \
        --limit-mm-per-prompt '{"image": 1}' \
        >"$log" 2>&1 &
    VQ_VLLM_PID=$!
    export VQ_VLLM_PID
    export VQ_VLLM_URL="http://$(hostname):$VQ_VLLM_PORT/v1"
    local waited=0
    until curl -sf "http://127.0.0.1:$VQ_VLLM_PORT/health" >/dev/null 2>&1; do
        if ! kill -0 "$VQ_VLLM_PID" 2>/dev/null; then
            echo "vLLM exited during startup; tail of $log:" >&2
            tail -n 50 "$log" >&2
            return 1
        fi
        if ((waited >= 1200)); then
            echo "vLLM not healthy after ${waited}s" >&2
            return 1
        fi
        sleep 10
        waited=$((waited + 10))
    done
    echo "vLLM ready at $VQ_VLLM_URL after ${waited}s"
}

vq_stop_vllm() {
    if [[ -n "${VQ_VLLM_PID:-}" ]] && kill -0 "$VQ_VLLM_PID" 2>/dev/null; then
        kill "$VQ_VLLM_PID" && wait "$VQ_VLLM_PID" 2>/dev/null || true
    fi
}

vq_banner() {
    echo "== $(date -Is) job=${SLURM_JOB_ID:-none} node=$(hostname) repo=$VQ_REPO"
    echo "== config=$VQ_CONFIG trt=$VQ_TRT_SIF venv=$VQ_VENV_IMG"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader || true
}
