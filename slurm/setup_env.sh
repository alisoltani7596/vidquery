#!/usr/bin/env bash
# One-time setup on a Fir LOGIN node (compute nodes have no internet access).
#   cd ~/vidquery && bash slurm/setup_env.sh            # everything
#   bash slurm/setup_env.sh containers venv              # only some steps
# Steps: containers | venv | weights | check   (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
export VQ_REPO="$PWD"
source slurm/env.sh

steps=("$@")
[[ ${#steps[@]} -eq 0 ]] && steps=(containers venv weights check)

pull() {  # pull <docker-uri> <sif>
    if [[ -f "$2" ]]; then
        echo "exists: $2"
    else
        echo "apptainer pull $2 <- $1 (several GB; takes a while)"
        apptainer pull --force "$2" "$1"
    fi
}

for step in "${steps[@]}"; do
    case "$step" in
    containers)
        pull "$VQ_TRT_IMAGE" "$VQ_TRT_SIF"
        pull "$VQ_VLLM_IMAGE" "$VQ_VLLM_SIF"
        ;;
    venv)
        # A venv inside the TensorRT container. --system-site-packages exposes the container's
        # tensorrt package; torch comes from the CUDA wheel index matching the container.
        apptainer exec --cleanenv --env-file "$(_vq_env_file)" --bind "$(_vq_binds)" \
            "$VQ_TRT_SIF" bash -lc "
            set -e
            python3 -m venv --system-site-packages '$VQ_VENV' \
                || python3 -m venv --without-pip --system-site-packages '$VQ_VENV'
            source '$VQ_VENV/bin/activate'
            python -m pip install --upgrade pip
            python -m pip install --index-url '$VQ_TORCH_INDEX' torch torchvision
            python -m pip install -e '$VQ_REPO[gpu,yolo,dev]'
            python -c 'import tensorrt, torch, onnxruntime; print(\"TensorRT\", tensorrt.__version__, \"| torch\", torch.__version__, \"| ORT\", onnxruntime.__version__)'
        "
        ;;
    weights)
        vq_exec python - <<PY
import os
from huggingface_hub import snapshot_download
for repo, patterns in [
    ("$VQ_SIGLIP_MODEL", ["*.json", "*.safetensors", "*.model", "*.txt"]),
    ("$VQ_VLM_MODEL", None),
]:
    path = snapshot_download(repo_id=repo, allow_patterns=patterns)
    print("downloaded", repo, "->", path)
os.chdir("$VQ_WEIGHTS")
from ultralytics import YOLO
YOLO("$VQ_YOLO_WEIGHTS")  # downloads into the current directory
print("YOLO weights in", os.getcwd(), os.listdir("."))
PY
        ;;
    check)
        vq_exec python - <<'PY'
from vidquery.config import load_config
from vidquery.models.siglip import siglip_config
s = load_config()
print("config:", s.detector.model, "|", s.encoder.model, "|", s.index_dir)
print("siglip image size:", siglip_config(s.encoder.model).vision_config.image_size)
PY
        vq_exec python -m pytest -q -m "not gpu" "$VQ_REPO/tests" -x
        ;;
    *)
        echo "unknown step $step" >&2
        exit 2
        ;;
    esac
done
echo "Setup done. Next: bash scripts/download_mot17.sh, then sbatch slurm/export.sbatch"
