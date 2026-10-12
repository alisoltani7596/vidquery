"""Ultralytics YOLO detectors (YOLO26 / YOLO11 / YOLOv8) as plain torch modules."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from torch import nn

DEFAULT_YOLO = "yolo26s.pt"


class UltralyticsDetectorNet(nn.Module):
    """Returns the raw head output: ``[B, N, 6]`` for NMS-free heads, else ``[B, 4+nc, N]``."""

    def __init__(self, weights: str) -> None:
        super().__init__()
        from ultralytics import YOLO

        yolo = YOLO(weights)
        model = yolo.model
        if hasattr(model, "fuse"):
            model = model.fuse(verbose=False)
        self.model = model.float().eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.model(images)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out


def export_ultralytics_onnx(
    weights: str, out_path: Path, input_size: int, opset: int, max_batch: int
) -> Path:
    """Export with Ultralytics' own exporter (dynamic batch, no embedded NMS)."""
    from ultralytics import YOLO

    exported = YOLO(weights).export(
        format="onnx",
        imgsz=input_size,
        dynamic=True,
        opset=opset,
        simplify=True,
        batch=max_batch,
        nms=False,
        device="cpu",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), out_path)
    return out_path
