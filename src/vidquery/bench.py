"""Latency / throughput / memory benchmark across backends, plus end-to-end ingest FPS.

Run via ``scripts/benchmark.py``. With ``--isolate`` (default on GPU) every backend runs in its
own subprocess so peak-memory numbers are not polluted by previously loaded models.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vidquery.backends.base import BackendSpec
from vidquery.config import Settings, load_config, resolve_device
from vidquery.models.registry import COMPONENTS, Component, component_spec, create_backend

log = logging.getLogger(__name__)

DEFAULT_BATCH_SIZES = (1, 4, 8, 16)


class GpuMemoryMonitor:
    """Polls NVML for this process's GPU memory (falls back to device-wide usage)."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.005) -> None:
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml: Any = None
        self._handle: Any = None
        self._baseline = 0
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:  # NVML missing (CPU run, no nvidia-ml-py)
            self._nvml = None

    @property
    def available(self) -> bool:
        return self._nvml is not None

    def _sample(self) -> int:
        nvml, handle = self._nvml, self._handle
        try:
            for proc in nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if proc.pid == os.getpid() and proc.usedGpuMemory:
                    return int(proc.usedGpuMemory)
        except Exception:  # process accounting may be restricted on shared nodes
            pass
        return int(nvml.nvmlDeviceGetMemoryInfo(handle).used) - self._baseline

    def __enter__(self) -> GpuMemoryMonitor:
        if self.available:
            self.peak_bytes = self._sample()

            def loop() -> None:
                while not self._stop.is_set():
                    self.peak_bytes = max(self.peak_bytes, self._sample())
                    time.sleep(self.interval_s)

            self._thread = threading.Thread(target=loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_mb(self) -> float | None:
        return self.peak_bytes / 2**20 if self.available else None


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def bench_component(
    settings: Settings,
    component: Component,
    backend: BackendSpec,
    batch_sizes: Sequence[int],
    *,
    device: str,
    warmup: int = 10,
    iters: int = 50,
) -> list[dict[str, Any]]:
    import torch

    spec = component_spec(settings, component)
    rows: list[dict[str, Any]] = []
    with GpuMemoryMonitor() as mon:
        t0 = time.perf_counter()
        runner = create_backend(settings, component, backend, device)
        load_s = time.perf_counter() - t0
        for bs in batch_sizes:
            row: dict[str, Any] = {
                "component": component,
                "model": spec.model_name,
                "backend": str(backend),
                "description": runner.description,
                "batch_size": bs,
                "load_s": round(load_s, 3),
            }
            if runner.max_batch_size is not None and bs > runner.max_batch_size:
                rows.append({**row, "skipped": f"engine max batch {runner.max_batch_size}"})
                continue
            x = spec.example_input(bs, seed=settings.seed)
            for _ in range(warmup):
                runner(x)
            _sync(device)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(iters):
                t = time.perf_counter()
                runner(x)
                _sync(device)
                times.append(time.perf_counter() - t)
            arr = np.asarray(times) * 1000
            row.update(
                p50_ms=float(np.percentile(arr, 50)),
                p95_ms=float(np.percentile(arr, 95)),
                mean_ms=float(arr.mean()),
                throughput=float(bs * len(arr) / (arr.sum() / 1000)),
                peak_gpu_mem_mb=mon.peak_mb,
                torch_peak_alloc_mb=(
                    torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None
                ),
            )
            rows.append(row)
        runner.close()
    return rows


def synthetic_video(
    path: Path, seconds: float = 20.0, fps: float = 25.0, size: tuple[int, int] = (720, 1280)
) -> Path:
    """A video with moving coloured "people" so detectors and crops have work to do."""
    import cv2

    from vidquery.video import write_video

    rng = np.random.default_rng(0)
    actors = [
        (
            rng.uniform(0, size[1]),
            rng.uniform(0, size[0] - 200),
            rng.uniform(-80, 80),
            tuple(int(c) for c in rng.integers(0, 255, 3)),
        )
        for _ in range(8)
    ]

    def frames() -> Any:
        for i in range(int(seconds * fps)):
            t = i / fps
            img = np.full((*size, 3), 110, np.uint8)
            for x0, y0, vx, color in actors:
                x = int((x0 + vx * t) % size[1])
                cv2.rectangle(img, (x, int(y0)), (x + 60, int(y0) + 180), color, -1)
            yield img

    write_video(frames(), path, fps)
    return path


def bench_ingest(
    settings: Settings, backend: BackendSpec, video: Path, *, device: str
) -> dict[str, Any]:
    from vidquery.pipeline import VidQuery

    s = settings.model_copy(deep=True)
    if s.detector.model != "color-blob":
        s.detector.backend = backend.kind
        s.detector.precision = backend.precision  # type: ignore[assignment]
    s.encoder.backend = backend.kind
    s.encoder.precision = backend.precision  # type: ignore[assignment]
    with tempfile.TemporaryDirectory() as tmp:
        s.index_dir = Path(tmp) / "index"
        engine = VidQuery(s)
        dummy = np.full((480, 640, 3), 127, np.uint8)
        with GpuMemoryMonitor() as mon:
            engine.detector.detect([dummy])  # build + warm up before timing
            engine.image_encoder.encode([dummy])
            stats = engine.ingest(video, save=False)
        engine.close()
    return {
        "backend": str(backend),
        "video": str(video),
        "video_duration_s": stats.duration_s,
        "sampled_frames": stats.sampled_frames,
        "crops": stats.crops,
        "elapsed_s": stats.elapsed_s,
        "ingest_fps": stats.ingest_fps,
        "realtime_factor": stats.realtime_factor,
        "stage_s": stats.stage_s,
        "peak_gpu_mem_mb": mon.peak_mb,
    }


def environment_info(device: str) -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": device,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    for mod in ("onnxruntime", "tensorrt"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def to_markdown(results: dict[str, Any]) -> str:
    env = results.get("environment", {})
    lines = [
        f"Benchmark on {env.get('gpu') or env.get('platform', '?')} "
        f"(torch {env.get('torch')}, onnxruntime {env.get('onnxruntime')}, "
        f"TensorRT {env.get('tensorrt')}), {env.get('time')}",
        "",
        "| component | backend | batch | p50 ms | p95 ms | items/s | peak GPU MB |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results.get("components", []):
        if "skipped" in r or "error" in r:
            lines.append(
                f"| {r['component']} | {r['backend']} | {r.get('batch_size', '-')} | "
                f"{r.get('skipped') or r.get('error')} | | | |"
            )
            continue
        lines.append(
            f"| {r['component']} | {r['description']} | {r['batch_size']} | {_fmt(r['p50_ms'])} | "
            f"{_fmt(r['p95_ms'])} | {_fmt(r['throughput'], 1)} | {_fmt(r['peak_gpu_mem_mb'], 0)} |"
        )
    if results.get("ingest"):
        lines += [
            "",
            "| backend | sampled frames | crops | ingest FPS | x realtime | detect s | embed s | "
            "peak GPU MB |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in results["ingest"]:
            if "error" in r:
                lines.append(f"| {r['backend']} | {r['error']} | | | | | | |")
                continue
            st = r["stage_s"]
            lines.append(
                f"| {r['backend']} | {r['sampled_frames']} | {r['crops']} | "
                f"{_fmt(r['ingest_fps'], 1)} | {_fmt(r['realtime_factor'], 1)} | "
                f"{_fmt(st.get('detect'))} | {_fmt(st.get('embed'))} | "
                f"{_fmt(r['peak_gpu_mem_mb'], 0)} |"
            )
    return "\n".join(lines) + "\n"


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    """Benchmark a single backend (all requested components + optional ingest)."""
    settings = Settings.model_validate(payload["settings"])
    backend = BackendSpec.parse(payload["backend"])
    device = payload["device"]
    out: dict[str, Any] = {"components": [], "ingest": []}
    for comp in payload["components"]:
        try:
            out["components"] += bench_component(
                settings,
                comp,
                backend,
                payload["batch_sizes"],
                device=device,
                warmup=payload["warmup"],
                iters=payload["iters"],
            )
        except Exception as exc:
            log.exception("Benchmark of %s on %s failed", comp, backend)
            out["components"].append(
                {
                    "component": comp,
                    "backend": str(backend),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        gc.collect()
    if payload.get("video"):
        try:
            out["ingest"].append(
                bench_ingest(settings, backend, Path(payload["video"]), device=device)
            )
        except Exception as exc:
            log.exception("Ingest benchmark on %s failed", backend)
            out["ingest"].append({"backend": str(backend), "error": f"{type(exc).__name__}: {exc}"})
    return out


def _run_isolated(payload: dict[str, Any]) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "vidquery.bench", "--worker", path],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-2000:])
        result: dict[str, Any] = json.loads(Path(path + ".out").read_text())
        return result
    finally:
        for p in (path, path + ".out"):
            Path(p).unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", "-c", help="YAML config (default: $VIDQUERY_CONFIG or defaults)")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="key.path=value")
    p.add_argument(
        "--components", nargs="+", default=["detector", "image_encoder"], choices=list(COMPONENTS)
    )
    p.add_argument(
        "--backends",
        nargs="+",
        default=["pytorch:fp32", "pytorch:fp16", "onnxruntime", "tensorrt:fp32", "tensorrt:fp16"],
        help="backend[:precision] specs, e.g. tensorrt:int8",
    )
    p.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--video", help="Video for end-to-end ingest FPS ('synthetic' to generate one)")
    p.add_argument("--device", default=None, help="cpu / cuda (default: settings.device)")
    p.add_argument(
        "--isolate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run each backend in a subprocess (default: on for CUDA)",
    )
    p.add_argument("--out", default="results/benchmark.json")
    p.add_argument("--md", default="results/benchmark.md")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.worker:
        payload = json.loads(Path(args.worker).read_text())
        Path(args.worker + ".out").write_text(json.dumps(run_one(payload)))
        return 0

    settings = load_config(args.config, args.overrides)
    device = resolve_device(args.device or settings.device)
    isolate = device.startswith("cuda") if args.isolate is None else args.isolate
    video = args.video
    tmpdir = None
    if video == "synthetic":
        tmpdir = tempfile.mkdtemp()
        video = str(synthetic_video(Path(tmpdir) / "synthetic.mp4"))

    results: dict[str, Any] = {
        "environment": environment_info(device),
        "config": settings.model_dump(mode="json"),
        "batch_sizes": args.batch_sizes,
        "components": [],
        "ingest": [],
    }
    specs = []
    for b in args.backends:
        spec = BackendSpec.parse(b)
        if device == "cpu" and (spec.kind == "tensorrt" or spec.precision == "fp16"):
            log.warning("Skipping %s on CPU", spec)
            continue
        specs.append(spec)
    for spec in specs:
        payload = {
            "settings": settings.model_dump(mode="json"),
            "backend": str(spec),
            "device": device,
            "components": args.components,
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "iters": args.iters,
            "video": video,
        }
        log.info("Benchmarking %s", spec)
        try:
            out = _run_isolated(payload) if isolate else run_one(payload)
        except Exception as exc:
            log.error("Backend %s failed: %s", spec, exc)
            out = {
                "components": [
                    {"component": c, "backend": str(spec), "error": str(exc)[-300:]}
                    for c in args.components
                ],
                "ingest": [],
            }
        results["components"] += out["components"]
        results["ingest"] += out["ingest"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    md = to_markdown(results)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
