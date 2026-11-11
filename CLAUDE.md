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
  geo/             vidquery locate: geometry.py (haversine, coarsen), backends.py (GeoCLIP, mock),
                   aggregate.py (clusters, radius, alternatives), people.py, vlm.py, locate.py,
                   mapview.py (Leaflet), evaluation.py (predict/tune/test), osv5m.py, mot17.py
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
- No Claude attribution in commits or PRs: no `Co-Authored-By` or `Claude-Session` trailers and
  no "Generated with Claude Code" lines. `.claude/settings.json` turns these off; commits are
  authored with the repo's own `git config user.name/user.email`.

## The cluster: Fir (Digital Research Alliance of Canada)

- Account `def-branzana`, user `ali7596`. GPU nodes: H100 80 GB. Request
  `--gpus-per-node=h100:1`.
- **No Docker.** Use Apptainer (`module load apptainer`). Images:
  `docker://nvcr.io/nvidia/tensorrt:26.03-py3` and `docker://vllm/vllm-openai:v0.17.1`.
- **Compute nodes have internet** (checked November 2026: HF, PyPI, PyTorch, NGC, Docker Hub,
  GitHub). Heavy setup goes through `slurm/setup.sbatch`, because torch CPU work aborts on login
  nodes (AOCL BLIS). Jobs still default to `HF_HUB_OFFLINE=1` so runs only use cached files
  (`VQ_ONLINE=1` lifts it), and Ultralytics auto-install is disabled.
- Everything lives under `$SCRATCH/vidquery` (see `slurm/README.md` for the layout); `$SCRATCH` is
  purged, so copy keepers to `/project/def-branzana/`.
- **File quota is the tight resource** (1M files per user on /scratch, shared with other
  projects). The venv is one SquashFS image (`$VQ_VENV_IMG`, mounted at `/opt/venv`; versions
  pinned in `slurm/venv-lock-<tag>.txt`), MOT17 frames are decoded from the zip and never
  extracted, and artifacts are a few large files. Don't write per-frame or per-hit image files.
- `slurm/env.sh` is the single source of paths and versions. It defines `vq_exec` (TensorRT
  container + venv image, `--cleanenv`, explicit env file and binds) and `vq_vllm_exec`. Submit jobs
  from the repo root so `SLURM_SUBMIT_DIR` finds it.
- Logs: `$SCRATCH/vidquery/logs/<job-name>-<jobid>.out`; results: `$SCRATCH/vidquery/results/`.

## First GPU session: run in this order

```bash
cd ~/vidquery && mkdir -p $SCRATCH/vidquery/logs
sbatch slurm/setup.sbatch                     # containers, venv, weights, MOT17, CPU tests
sbatch slurm/export.sbatch                    # check results/parity.md + gpu-tests.txt
sbatch slurm/benchmark.sbatch
sbatch slurm/ingest.sbatch
# write $SCRATCH/vidquery/data/queries.jsonl (hand-annotated; see data/queries.example.jsonl)
sbatch slurm/eval.sbatch
```

Nothing below has run on a GPU yet. Verify these first, in order, and fix what breaks:

1. **Driver vs container CUDA.** `nvidia-smi` on a GPU node. `26.03-py3` is CUDA 13, which needs a
   driver ≥ 580 (or forward compatibility). If the driver is older, pick an NGC tag with CUDA 12.x
   and set `VQ_TORCH_INDEX` to the matching `cu12x` index (`slurm/env.sh`). Fir had driver
   580.178.04 / CUDA 13.0 in November 2026, so no change was needed.
2. **TensorRT import in the venv:** `vq_exec python -c "import tensorrt; print(tensorrt.__version__)"`.
3. **ONNX Runtime CUDA EP.** PyPI `onnxruntime-gpu` 1.31 is a CUDA 13 build (it links
   `libcublas.so.13`) and ran on the CUDA EP on Fir in November 2026. ORT only warns when the
   CUDA EP fails to load, so `scripts/gpu_check.py` (first step of `export.sbatch`) and the GPU
   test fail on a CPU fallback, and the tables print the provider (`onnxruntime:fp32 (CUDA)`).
4. **YOLO26 torch wrapper vs ONNX.** Ultralytics 8.4 runs the one-to-many head in eval mode and
   `fuse()` deletes the unused branch, so `UltralyticsDetectorNet` sets `head.end2end = True`
   before fusing (as the exporter does with `nms=False`); both then output `[B, 300, 6]`.
   Detector parity is judged by detection F1 only: top-k rows reorder on near-ties, so the raw
   tensor is not compared elementwise. On Fir: ORT F1 0.998, TensorRT FP32 1.000, FP16 0.997.
5. **Dynamo export of SigLIP 2 So400m.** If it fails or TensorRT can't parse it, run
   `EXPORTER=torchscript sbatch slurm/export.sbatch`. If ONNX > 2 GB, external data is written
   automatically for big models.
6. **TensorRT version.** The container is pinned to 26.03 = TensorRT 10.16 (weakly typed FP16
   flags, deprecated entropy calibrator). TensorRT 11 (26.04+) is unsupported: the strongly typed
   FP16-ONNX fallback fails to parse (FP32 tensor into an FP16 Conv; its GPU test is xfail),
   and INT8 would need `INT8_METHOD=modelopt`, whose `[onnx]` extra pins
   `onnxruntime-gpu<1.31` and `onnx~=1.21` (it needs its own venv image). INT8 is experimental
   and not deployed: detector F1 0.76 with entropy calibration, SigLIP INT8 == FP16 (implicit INT8
   leaves transformer matmuls in FP16). `INT8=1` parity is reported but never fails the job.
7. **vLLM + Qwen3.5.** Check `vllm.sbatch` comes up (`--limit-mm-per-prompt '{"image": 1}'` is the
   JSON form newer vLLM wants) and that a request returns `logprobs`. Quick check from a job:
   `curl $URL/models`. If answers are not "Yes"/"No" tokens, thinking is not disabled. Look at
   `chat_template_kwargs`.
8. **NVML memory.** `GpuMemoryMonitor` uses per-process NVML accounting and falls back to
   device-wide usage, which is fine on an exclusive GPU but not on shared/MIG GPUs.

After good runs: paste `results/benchmark.md`, `results/parity.md` and `results/eval.md` into
the README placeholders, and tune `query.score_margin` / `merge_gap_s` with the eval script.
Note the GPU, driver, container tag and versions next to the tables.

## Geolocation (`vidquery locate`)

- Coarse by design: every reported position goes through `geo.geometry.coarsen` (decimals match
  the radius, radius widened by the rounding shift, floor `geo.min_radius_km` = 25). Don't add
  outputs that bypass it, and keep the responsible-use note in the README and in `notice`.
- The VLM opinion is shown next to the GeoCLIP estimate, never fused into it.
- Benchmark protocol: tune only on OSV-5M val (`scripts/geo_eval.py tune`), freeze
  `results/geo-params.yaml`, run `test` once (it refuses to overwrite `geo-eval.md`).
- `data/mot17_locations.json`: `known` only with a source URL + quote; otherwise `unknown` with
  null coordinates. All 14 sequences were unknown when checked (November 2026).
- Fir: geoclip is installed `--no-deps` (its wheel asks for Pillow>=13, the lock pins 12) plus
  pandas/geopy; the CLIP ViT-L/14 backbone is pre-fetched by the `weights` step.
- No copyrighted movie footage or frames in the repo; local tests on such files stay in
  `$SCRATCH`.

## Ideas not done yet

Tracking (ByteTrack) to merge segments by identity and ground "walking left"; IVF/HNSW FAISS
indexes for large corpora; caching crop thumbnails for faster verification; batching several
images per VLM request; a Triton Inference Server deployment of the TensorRT engines; explicit
Q/DQ INT8 with TensorRT Model Optimizer; TensorRT 11 support.
