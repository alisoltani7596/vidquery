#!/usr/bin/env python
"""Check that torch, TensorRT and ONNX Runtime really use the GPU, before export/benchmarks.

    vq_exec python scripts/gpu_check.py        # slurm/export.sbatch runs this first

ONNX Runtime falls back to CPU when its CUDA provider can't load (wrong CUDA major, missing
cuDNN). That only logs a warning, so this builds a tiny model, runs it on CUDA and checks the
session's providers. Exits non-zero if any check fails.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def check_torch() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    x = torch.randn(256, 256, device="cuda")
    float((x @ x).sum())
    return (
        f"torch {torch.__version__} (CUDA {torch.version.cuda}, cuDNN "
        f"{torch.backends.cudnn.version()}) on {torch.cuda.get_device_name(0)}; "
        f"container CUDA {os.environ.get('CUDA_VERSION', '?')}"
    )


def check_tensorrt() -> str:
    import tensorrt as trt

    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    if builder is None:
        raise RuntimeError("trt.Builder() returned None (no usable GPU?)")
    return f"TensorRT {trt.__version__} (builder OK, fast FP16: {builder.platform_has_fast_fp16})"


def check_onnxruntime() -> str:
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDAExecutionProvider not in {available} (CPU-only onnxruntime?)")
    w = helper.make_tensor("w", TensorProto.FLOAT, [64, 64], np.eye(64, dtype=np.float32).ravel())
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "gpu_check",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [8, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [8, 64])],
        [w],
    )
    # IR 9 keeps the file loadable by older ORT builds (the onnx package now defaults to newer).
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gpu_check.onnx"
        onnx.save(model, str(path))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = len(os.sched_getaffinity(0))  # the job's CPUs, not the node's
        # The CUDA EP uses TF32 for FP32 MatMul by default (~1e-3 error); turn it off so the
        # result can be checked exactly.
        cuda = ("CUDAExecutionProvider", {"use_tf32": "0"})
        sess = ort.InferenceSession(
            str(path), sess_options=opts, providers=[cuda, "CPUExecutionProvider"]
        )
        x = np.random.default_rng(0).standard_normal((8, 64), dtype=np.float32)
        (y,) = sess.run(None, {"x": x})
    used = sess.get_providers()
    if used[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"session fell back to {used} (CUDA provider failed to load)")
    if not np.allclose(y, x, atol=1e-5):
        raise RuntimeError("CUDA MatMul result is wrong")
    return f"onnxruntime {ort.__version__} runs on {used[0]} (available: {available})"


def main() -> int:
    failed = 0
    for name, check in [
        ("torch", check_torch),
        ("tensorrt", check_tensorrt),
        ("onnxruntime", check_onnxruntime),
    ]:
        try:
            print(f"OK    {name}: {check()}", flush=True)
        except Exception as exc:  # report every check, then fail
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
