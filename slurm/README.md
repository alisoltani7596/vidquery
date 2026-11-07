# vidquery on the Fir cluster (Slurm + Apptainer)

Account `def-branzana`, user `ali7596`, GPUs: H100 80 GB (`--gpus-per-node=h100:1`).
Docker isn't available, so OCI images are pulled into Apptainer SIFs. Compute nodes have internet
access, but jobs default to `HF_HUB_OFFLINE=1` so they only use cached files (`VQ_ONLINE=1` lifts
it).

## Layout on `$SCRATCH/vidquery`

| Path | Contents |
|---|---|
| `containers/` | `tensorrt_<tag>.sif`, `vllm-openai_<tag>.sif` |
| `venv-trt-<tag>/` | venv created inside the TensorRT container (`--system-site-packages`) |
| `cache/` | Hugging Face, torch, pip, Apptainer, vLLM and Triton caches |
| `weights/` | `yolo26s.pt` |
| `data/MOT17/` | MOT17 sequences (`train/`, `test/`) and `videos/*.mp4` |
| `data/queries.jsonl` | hand-written eval queries (you create this) |
| `artifacts/onnx`, `artifacts/trt` | exported ONNX models and TensorRT engines |
| `index/mot17/` | FAISS index + metadata |
| `results/` | parity, benchmark and eval JSON/markdown |
| `logs/` | Slurm and vLLM logs |

`$SCRATCH` is purged after a period of inactivity. Copy `results/` you want to keep to
`/project/def-branzana/...` or into the repo.

## One-time setup (CPU job)

SIF builds and the CPU tests are heavy, and torch CPU work aborts on login nodes, so setup runs
as a job:

```bash
cd ~/vidquery
mkdir -p $SCRATCH/vidquery/logs
sbatch slurm/setup.sbatch            # containers, venv, weights, MOT17 (~5.5 GB + mp4s), CPU tests
```

`setup_env.sh` runs:

```bash
apptainer pull $VQ_TRT_SIF  docker://nvcr.io/nvidia/tensorrt:26.03-py3
apptainer pull $VQ_VLLM_SIF docker://vllm/vllm-openai:v0.17.1
apptainer exec $VQ_TRT_SIF python3 -m venv --system-site-packages $VQ_VENV
# + pip install torch (cu130 wheels) and -e .[gpu,yolo,dev], weight downloads, CPU tests
```

Inside a job, `APPTAINER_TMPDIR` is the node-local `$SLURM_TMPDIR` (env.sh sets it). If a pull
gets killed, resubmit: pulls resume from the layer cache.

## Jobs (submit from the repo root)

| Script | What it does | Useful variables |
|---|---|---|
| `setup.sbatch` | one-time setup: containers, venv, weights, MOT17, CPU tests | `STEPS="venv weights"` |
| `export.sbatch` | ONNX export (3 components), TensorRT FP32/FP16 engines, parity report, `pytest -m gpu` | `INT8=1`, `INT8_METHOD=modelopt`, `EXPORTER=torchscript`, `CALIB=<dir>` |
| `benchmark.sbatch` | latency/throughput/peak memory, batch 1/4/8/16, ingest FPS | `BACKENDS="..."`, `VIDEO=<mp4>` |
| `ingest.sbatch` | ingest every video/sequence in a directory | `DATASET=<dir>`, `EXTRA="--set ..."` |
| `eval.sbatch` | eval with/without verifier; starts vLLM in the same job by default | `QUERIES=`, `VERIFIER=off/on/both`, `VLM_URL=`, `VLLM_MEM=0.45` |
| `vllm.sbatch` | standalone vLLM server; writes its URL to `$SCRATCH/vidquery/vllm_endpoint` | `VQ_VLM_MODEL=Qwen/Qwen3.5-2B`, `VLLM_MEM` |

Every job sources `slurm/env.sh`, which defines `vq_exec` (run in the TensorRT container with
the venv) and `vq_vllm_exec`. For interactive work:

```bash
salloc --account=def-branzana --gpus-per-node=h100:1 --cpus-per-task=8 --mem=64G --time=1:00:00
source slurm/env.sh
vq_exec vidquery query "man in a blue shirt walking left" -k 5
vq_exec python -m pytest -m gpu tests
```

## Changing versions

All of these are in `slurm/env.sh` and can be overridden from the environment:
`VQ_TRT_TAG` (and the matching `VQ_TORCH_INDEX`), `VQ_VLLM_TAG`, `VQ_VLM_MODEL`,
`VQ_SIGLIP_MODEL`, `VQ_YOLO_WEIGHTS`. Changing `VQ_TRT_TAG` creates a new venv path. TensorRT
engines must be rebuilt after any TensorRT or GPU change.
