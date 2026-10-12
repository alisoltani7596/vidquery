"""Person detection: YOLO-style pre/post-processing around any backend.

Two raw output layouts are understood, matching what Ultralytics models export:

* **dense** ``[B, 4 + num_classes, N]``: YOLOv8/YOLO11 heads, boxes as cx, cy, w, h in input
  pixels, followed by per-class scores. Needs NMS (done here, in numpy).
* **end-to-end** ``[B, N, 6]``: NMS-free heads (YOLO26 default) with x1, y1, x2, y2, score, class.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from vidquery.backends.base import InferenceBackend
from vidquery.types import Detection

LETTERBOX_PAD = 114


class PersonDetector(Protocol):
    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        """Return person boxes (original pixel coords) for each BGR frame."""
        ...


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    pad_x: float
    pad_y: float
    orig_w: int
    orig_h: int


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize keeping aspect ratio and pad to ``size x size``. Returns RGB uint8."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    canvas = np.full((size, size, 3), LETTERBOX_PAD, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    canvas[top : top + new_h, left : left + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb, LetterboxMeta(scale, float(left), float(top), w, h)


def preprocess_letterbox(frames: Sequence[np.ndarray], size: int) -> tuple[np.ndarray, list[LetterboxMeta]]:
    """BGR frames -> float32 ``[B, 3, size, size]`` in [0, 1] plus per-frame metadata."""
    batch = np.empty((len(frames), 3, size, size), dtype=np.float32)
    metas: list[LetterboxMeta] = []
    for i, frame in enumerate(frames):
        rgb, meta = letterbox(frame, size)
        batch[i] = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        metas.append(meta)
    return batch, metas


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``[N, 4]`` and ``[M, 4]`` xyxy boxes."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS; returns kept indices sorted by descending score."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def _unletterbox(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    out = boxes.astype(np.float64).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - meta.pad_x) / meta.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - meta.pad_y) / meta.scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, meta.orig_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, meta.orig_h)
    return out


def is_end_to_end(raw: np.ndarray) -> bool:
    return raw.ndim == 3 and raw.shape[-1] == 6 and raw.shape[1] != 6


def decode_detections(
    raw: np.ndarray,
    metas: Sequence[LetterboxMeta],
    *,
    conf_threshold: float,
    iou_threshold: float,
    person_class: int,
    max_detections: int,
) -> list[list[Detection]]:
    """Turn raw network output into per-frame person detections."""
    results: list[list[Detection]] = []
    end_to_end = is_end_to_end(raw)
    for b, meta in enumerate(metas):
        if end_to_end:
            det = raw[b]
            mask = (det[:, 4] >= conf_threshold) & (np.round(det[:, 5]) == person_class)
            boxes, scores = det[mask, :4], det[mask, 4]
            order = np.argsort(-scores, kind="stable")[:max_detections]
        else:
            pred = raw[b].T  # [N, 4 + nc]
            scores = pred[:, 4 + person_class]
            mask = scores >= conf_threshold
            cxcywh, scores = pred[mask, :4], scores[mask]
            boxes = np.empty_like(cxcywh)
            boxes[:, :2] = cxcywh[:, :2] - cxcywh[:, 2:] / 2
            boxes[:, 2:] = cxcywh[:, :2] + cxcywh[:, 2:] / 2
            order = nms(boxes, scores, iou_threshold)[:max_detections]
        boxes = _unletterbox(boxes[order], meta)
        frame_dets = [
            Detection(float(x1), float(y1), float(x2), float(y2), float(s))
            for (x1, y1, x2, y2), s in zip(boxes, scores[order], strict=True)
            if x2 > x1 and y2 > y1
        ]
        results.append(frame_dets)
    return results


class YoloDetector:
    """YOLO-family person detector running on any ``InferenceBackend``."""

    def __init__(
        self,
        backend: InferenceBackend,
        input_size: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        person_class: int = 0,
        max_detections: int = 100,
    ) -> None:
        self.backend = backend
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_class = person_class
        self.max_detections = max_detections

    def raw(self, frames: Sequence[np.ndarray]) -> tuple[np.ndarray, list[LetterboxMeta]]:
        batch, metas = preprocess_letterbox(frames, self.input_size)
        return self.backend(batch)[0], metas

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        max_b = self.backend.max_batch_size or len(frames)
        out: list[list[Detection]] = []
        for start in range(0, len(frames), max_b):
            raw, metas = self.raw(frames[start : start + max_b])
            out.extend(
                decode_detections(
                    raw,
                    metas,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                    person_class=self.person_class,
                    max_detections=self.max_detections,
                )
            )
        return out


class ColorBlobDetector:
    """Deterministic stand-in detector: saturated blobs on a grey background are "people".

    Used by tests and the weight-free demo config; it needs no network and no backend.
    """

    def __init__(self, min_area: int = 30, saturation: int = 80, max_detections: int = 100):
        self.min_area = min_area
        self.saturation = saturation
        self.max_detections = max_detections

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        results: list[list[Detection]] = []
        kernel = np.ones((3, 3), np.uint8)
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = ((hsv[..., 1] > self.saturation) & (hsv[..., 2] > 50)).astype(np.uint8)
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            n, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
            dets = [
                Detection(float(x), float(y), float(x + w), float(y + h), 0.99)
                for x, y, w, h, area in stats[1:n]
                if area >= self.min_area
            ]
            dets.sort(key=lambda d: -(d.width * d.height))
            results.append(dets[: self.max_detections])
        return results
