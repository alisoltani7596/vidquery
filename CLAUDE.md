# CLAUDE.md

Guidance for Claude Code sessions working on **vidquery**, most likely on the Fir cluster with
the first real GPU runs.

## What this repo is

Text-to-video person search, built as a deployment portfolio project:
sample frames → YOLO person detection → SigLIP 2 embeddings of frames and person crops → FAISS →
text query → merge hits into time segments with boxes → optional VLM yes/no verification through
vLLM's OpenAI-compatible API. The detector and image encoder (and the text encoder) run on
**pytorch / onnxruntime / tensorrt** behind one `InferenceBackend` interface. Export, parity,
benchmark and eval tooling are part of the deliverable, as are the README tables they fill.

Model choices (checked October 2026; re-check before big runs):
- Detector: `yolo26s.pt` (Ultralytics YOLO26-S, NMS-free head → `[B, 300, 6]`). `yolo11s.pt` (dense
  `[B, 84, N]`) also works; `models/detector.py` decodes both layouts.
- Encoder: `google/siglip2-so400m-patch16-384` (FixRes, loads as `SiglipModel`, static 384×384 →
  TensorRT friendly). `google/siglip2-giant-opt-patch16-384` is the accuracy upgrade.
  NaFlex checkpoints are rejected on purpose (variable patch count).
- VLM: `Qwen/Qwen3.5-4B` (vLLM ≥ 0.17; `Qwen/Qwen3.5-2B` for low memory). Thinking is disabled per
  request via `chat_template_kwargs={"enable_thinking": false}`.

## Layout

```
src/vidquery/
  config.py        pydantic Settings, YAML + ${ENV} expansion + --set overrides
  types.py         Detection, Record, Hit, Segment, VideoEntry
  video.py         VideoSource (files + MOT image sequences), sampling, prefetch, writers
  backends/        base.py (InferenceBackend, BackendSpec "kind:precision"), torch/onnx/tensorrt
  models/          detector.py (letterbox, decode, NMS, ColorBlobDetector), encoder.py,
                   siglip.py, yolo.py, tiny.py (random tiny models), mock.py (colour models),
                   registry.py (ComponentSpec, artifact paths, build_* on any backend)
  export/          onnx_export.py, tensorrt_build.py (FP32/FP16/INT8, TRT 11 fallback),
                   calibration.py, __init__.build_trt_engines
  index.py         FAISS (frame + crop indexes, shared ids) + npz metadata, atomic save
  segments.py      merge_hits, temporal IoU
  pipeline.py      VidQuery engine: ingest(), query(), load_hit_image()
  verify.py        VLMVerifier (httpx, logprobs → p(yes), rerank/filter)
  parity.py        cross-backend comparison + markdown report
  bench.py         benchmark harness (subprocess isolation, NVML memory)
  evaluation.py    queries.jsonl metrics + eval CLI
  api.py, cli.py, demo.py, synthetic.py
scripts/  slurm/  configs/  docker/  tests/
```

## Conventions

- Python 3.11 target (the NGC container runs 3.12, so keep code compatible with both), src
  layout, `from __future__ import annotations`, full type hints. `mypy` is strict.
- Before committing: `ruff check src tests scripts && ruff format --check src tests scripts && mypy && pytest`.
  CI (`.github/workflows/ci.yml`) runs exactly these plus a CPU Docker build. ruff/mypy are
  pinned in the `dev` extra.
- **Tests never download anything.** Use `tiny` models (random HF `SiglipModel` + `TinyDetectorNet`)
  for export/parity/shape tests and `color-blob` + `color-mock` for semantic end-to-end tests on
  the synthetic video (`vidquery.synthetic`). GPU-only tests get `@pytest.mark.gpu` (auto-skipped
  without CUDA) and should use the tiny models too.
- Backends take and return **numpy**; pre/post-processing lives in `models/`, never in a backend.
  A new component goes through `models/registry.py::component_spec`.
- Config changes: update `config.py`, the YAMLs in `configs/` (tests validate every shipped
  config) and the README if user-facing.
- Never commit weights, ONNX files, engines, SIFs, indexes or datasets (see `.gitignore`).
  Results tables (`results/*.md`) may be committed; JSON stays ignored.
- Commits: small and logical, imperative subject, body explains why.

## The cluster: Fir (Digital Research Alliance of Canada)

- Account `def-branzana`, user `ali7596`. GPU nodes: H100 80 GB. Request
  `--gpus-per-node=h100:1`.
- **No Docker.** Use Apptainer (`module load apptainer`). Images:
  `docker://nvcr.io/nvidia/tensorrt:26.03-py3` and `docker://vllm/vllm-openai:v0.17.1`.
- **Compute nodes are offline.** Downloads (apptainer pull, pip, HF weights, YOLO weights, MOT17)
  happen on a login node. Jobs set `HF_HUB_OFFLINE=1`, and Ultralytics auto-install is disabled.
- Everything lives under `$SCRATCH/vidquery` (see `slurm/README.md` for the layout); `$SCRATCH` is
  purged, so copy keepers to `/project/def-branzana/`.
- `slurm/env.sh` is the single source of paths and versions. It defines `vq_exec` (TensorRT
  container + venv, `--cleanenv`, explicit env file and binds) and `vq_vllm_exec`. Submit jobs
  from the repo root so `SLURM_SUBMIT_DIR` finds it.
- Logs: `$SCRATCH/vidquery/logs/<job-name>-<jobid>.out`; results: `$SCRATCH/vidquery/results/`.

## First GPU session: run in this order

```bash
cd ~/vidquery && module load apptainer
bash slurm/setup_env.sh                       # login node
bash scripts/download_mot17.sh                # login node
sbatch slurm/export.sbatch                    # check results/parity.md + gpu-tests.txt
sbatch slurm/benchmark.sbatch
sbatch slurm/ingest.sbatch
# write $SCRATCH/vidquery/data/queries.jsonl (hand-annotated; see data/queries.example.jsonl)
sbatch slurm/eval.sbatch
```

Nothing below has run on a GPU yet. Verify these first, in order, and fix what breaks:

1. **Driver vs container CUDA.** `nvidia-smi` on a GPU node. `26.03-py3` is CUDA 13, which needs a
   driver ≥ 580 (or forward compatibility). If the driver is older, pick an NGC tag with CUDA 12.x
   and set `VQ_TORCH_INDEX` to the matching `cu12x` index (`slurm/env.sh`).
2. **TensorRT import in the venv:** `vq_exec python -c "import tensorrt; print(tensorrt.__version__)"`.
3. **ONNX Runtime CUDA EP.** PyPI `onnxruntime-gpu` targets CUDA 12. The parity/benchmark tables
   print the provider (`onnxruntime:fp32 (CUDA)` vs `(CPU)`). If it fell back to CPU, install a
   CUDA 13 build of ORT, or report ORT numbers as CPU.
4. **YOLO26 torch wrapper vs ONNX.** `UltralyticsDetectorNet` returns `model(x)[0]` in eval mode.
   The detector parity (detection F1) checks that it matches Ultralytics' ONNX export. If the
   layouts differ, set the Detect head's `export`/`format` attributes the way Ultralytics'
   exporter does.
5. **Dynamo export of SigLIP 2 So400m.** If it fails or TensorRT can't parse it, run
   `EXPORTER=torchscript sbatch slurm/export.sbatch`. If ONNX > 2 GB, external data is written
   automatically for big models.
6. **TensorRT version.** 26.03 = TensorRT 10.16: weakly-typed FP16 flags and the (deprecated)
   entropy calibrator should work. On TensorRT 11 (26.04+), FP16 goes through the strongly
   typed FP16-ONNX fallback (`onnxconverter-common`) and INT8 needs `INT8_METHOD=modelopt`
   (`pip install -e .[int8]` on the login node). Both paths are untested.
7. **vLLM + Qwen3.5.** Check `vllm.sbatch` comes up (`--limit-mm-per-prompt '{"image": 1}'` is the
   JSON form newer vLLM wants) and that a request returns `logprobs`. Quick check from a job:
   `curl $URL/models`. If answers are not "Yes"/"No" tokens, thinking is not disabled. Look at
   `chat_template_kwargs`.
8. **NVML memory.** `GpuMemoryMonitor` uses per-process NVML accounting and falls back to
   device-wide usage, which is fine on an exclusive GPU but not on shared/MIG GPUs.

After good runs: paste `results/benchmark.md`, `results/parity.md` and `results/eval.md` into
the README placeholders, and tune `query.score_margin` / `merge_gap_s` with the eval script.
Note the GPU, driver, container tag and versions next to the tables.

## Ideas not done yet

Tracking (ByteTrack) to merge segments by identity and ground "walking left"; IVF/HNSW FAISS
indexes for large corpora; caching crop thumbnails for faster verification; batching several
images per VLM request; a Triton Inference Server deployment of the TensorRT engines.
