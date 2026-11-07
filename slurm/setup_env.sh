#!/usr/bin/env bash
# One-time setup. Run it through slurm/setup.sbatch: SIF builds and the CPU tests are heavy, and
# torch CPU work aborts on Fir login nodes (AOCL BLIS). Compute nodes have internet access.
#   sbatch slurm/setup.sbatch                            # from the repo root
#   VQ_ONLINE=1 bash slurm/setup_env.sh containers venv  # only some steps, inside a job
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
        # A venv inside the TensorRT container, built on node-local disk and bound at $VQ_VENV
        # (the path jobs see), then packed into one SquashFS image ($VQ_VENV_IMG).
        # --system-site-packages exposes the container's tensorrt package; torch comes from the
        # CUDA wheel index matching the container. slurm/venv-lock-<tag>.txt pins the versions
        # that were validated on the GPU; each build writes its freeze next to the image.
        : "${SLURM_TMPDIR:?the venv step needs node-local disk; run it through setup.sbatch}"
        lock="$VQ_REPO/slurm/venv-lock-$VQ_TRT_TAG.txt"
        pins="" torch_index="--index-url '$VQ_TORCH_INDEX'"
        if [[ -f "$lock" ]]; then
            # torch is pinned to its +cuXXX build (torch index only); other pins may be on PyPI.
            pins="-c '$lock'"
            torch_index="$torch_index --extra-index-url https://pypi.org/simple"
        fi
        export VQ_VENV_DIR="$SLURM_TMPDIR/venv"
        mkdir -p "$VQ_VENV_DIR"
        apptainer exec --cleanenv --env-file "$(_vq_env_file)" --bind "$(_vq_binds)" \
            "$VQ_TRT_SIF" bash -lc "
            set -e
            python3 -m venv --system-site-packages '$VQ_VENV' \
                || python3 -m venv --without-pip --system-site-packages '$VQ_VENV'
            source '$VQ_VENV/bin/activate'
            python -m pip install --upgrade pip
            python -m pip install $pins $torch_index torch torchvision
            python -m pip install $pins -e '$VQ_REPO[gpu,yolo,dev]'
            # Ultralytics pulls opencv-python, whose cv2 needs X11 libraries (libxcb.so.1) that
            # the container lacks. Both wheels write cv2/, so drop it and reinstall the headless
            # build vidquery depends on.
            python -m pip uninstall -y opencv-python
            python -m pip install $pins --force-reinstall --no-deps opencv-python-headless
            python -m pip freeze --exclude-editable > '$VQ_VENV_IMG.freeze.txt'
        "
        unset VQ_VENV_DIR
        # gzip: Fir's Apptainer can't mount zstd SquashFS images.
        mksquashfs "$SLURM_TMPDIR/venv" "$VQ_VENV_IMG.tmp" -comp gzip -noappend -quiet \
            -processors "${SLURM_CPUS_PER_TASK:-4}"
        mv -f "$VQ_VENV_IMG.tmp" "$VQ_VENV_IMG"
        echo "venv image: $VQ_VENV_IMG ($(du -h "$VQ_VENV_IMG" | cut -f1))"
        if [[ -f "$lock" ]]; then
            diff "$lock" "$VQ_VENV_IMG.freeze.txt" && echo "venv matches $lock" \
                || echo "venv differs from $lock (diff above)"
        fi
        # Check the image the way jobs will see it.
        vq_exec python -c 'import cv2, torch, onnxruntime; print("OpenCV", cv2.__version__, "| torch", torch.__version__, "| ORT", onnxruntime.__version__, onnxruntime.get_available_providers())'
        # No GPU in setup.sbatch; export.sbatch rechecks TensorRT on a GPU.
        vq_exec python -c 'import tensorrt; print("TensorRT", tensorrt.__version__)' \
            || echo 'tensorrt import failed without a GPU; recheck on a GPU node'
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
echo "setup_env.sh done: ${steps[*]}"
