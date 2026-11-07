"""Ultralytics YOLO detectors (YOLO26 / YOLO11 / YOLOv8) as plain torch modules."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

DEFAULT_YOLO = "yolo26s.pt"


class UltralyticsDetectorNet(nn.Module):
    """Returns the raw head output: ``[B, N, 6]`` for NMS-free heads, else ``[B, 4+nc, N]``."""

    def __init__(self, weights: str) -> None:
        super().__init__()
        from ultralytics.models import YOLO

        model: Any = YOLO(weights).model
        head: Any = model.model[-1] if hasattr(model, "model") else None
        if getattr(head, "one2one_cv2", None) is not None:
            # Select the NMS-free one-to-one head, as Ultralytics' exporter does with nms=False.
            # Recent releases default to one-to-many in eval mode, and fuse() drops the unused
            # branch, so this has to happen before fusing to match the ONNX export ([B, 300, 6]).
            head.end2end = True
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
    from ultralytics.models import YOLO

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
