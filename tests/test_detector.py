from __future__ import annotations

import numpy as np
import pytest

from vidquery.backends.base import InferenceBackend
from vidquery.models.detector import (
    ColorBlobDetector,
    YoloDetector,
    box_iou,
    decode_detections,
    letterbox,
    nms,
    preprocess_letterbox,
)
from vidquery.models.registry import build_detector
from vidquery.synthetic import SCRIPT, actor_box, synthetic_frames


class FakeBackend(InferenceBackend):
    kind = "pytorch"

    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.input_names = ["images"]
        self.output_names = ["output"]
        self.calls: list[tuple[int, ...]] = []

    def run(self, inputs):  # type: ignore[no-untyped-def]
        self.calls.append(inputs["images"].shape)
        return [self.output[: inputs["images"].shape[0]]]


def test_letterbox_geometry() -> None:
    img = np.zeros((100, 200, 3), np.uint8)
    out, meta = letterbox(img, 64)
    assert out.shape == (64, 64, 3)
    assert meta.scale == pytest.approx(0.32)
    assert meta.pad_x == 0 and meta.pad_y == 16
    batch, metas = preprocess_letterbox([img, img], 64)
    assert batch.shape == (2, 3, 64, 64) and batch.dtype == np.float32
    assert batch.min() >= 0.0 and batch.max() <= 1.0
    assert len(metas) == 2


def test_box_iou_and_nms() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], float)
    ious = box_iou(boxes, boxes)
    assert ious[0, 0] == pytest.approx(1.0)
    assert ious[0, 1] == pytest.approx(81 / 119)
    assert ious[0, 2] == 0.0
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), iou_threshold=0.5)
    assert keep.tolist() == [0, 2]


def test_decode_dense_layout_maps_back_to_original_pixels() -> None:
    frame = np.zeros((100, 200, 3), np.uint8)  # letterboxed into 64x64: scale .32, pad_y 16
    _, meta = letterbox(frame, 64)
    # [1, 4 + 2 classes, 3 anchors]: cx, cy, w, h, score_person, score_other
    raw = np.zeros((1, 6, 3), np.float32)
    raw[0, :, 0] = [32, 32, 16, 16, 0.9, 0.1]  # person in the centre
    raw[0, :, 1] = [33, 33, 16, 16, 0.8, 0.1]  # duplicate -> suppressed by NMS
    raw[0, :, 2] = [10, 30, 8, 8, 0.2, 0.95]  # another class
    dets = decode_detections(
        raw, [meta], conf_threshold=0.5, iou_threshold=0.5, person_class=0, max_detections=10
    )
    assert len(dets[0]) == 1
    d = dets[0][0]
    assert d.score == pytest.approx(0.9)
    # centre (32, 32) in letterbox -> ((32 - 0) / .32, (32 - 16) / .32) = (100, 50)
    assert (d.x1 + d.x2) / 2 == pytest.approx(100, abs=0.5)
    assert (d.y1 + d.y2) / 2 == pytest.approx(50, abs=0.5)
    assert d.width == pytest.approx(50, abs=0.5)


def test_decode_end_to_end_layout() -> None:
    frame = np.zeros((64, 64, 3), np.uint8)
    _, meta = letterbox(frame, 64)
    raw = np.zeros((1, 300, 6), np.float32)
    raw[0, 0] = [10, 10, 30, 50, 0.8, 0]
    raw[0, 1] = [5, 5, 20, 20, 0.9, 2]  # not a person
    raw[0, 2] = [1, 1, 3, 3, 0.1, 0]  # below threshold
    dets = decode_detections(
        raw, [meta], conf_threshold=0.35, iou_threshold=0.5, person_class=0, max_detections=10
    )
    assert [d.box for d in dets[0]] == [(10.0, 10.0, 30.0, 50.0)]


def test_yolo_detector_respects_backend_max_batch() -> None:
    class LimitedBackend(FakeBackend):
        @property
        def max_batch_size(self) -> int:
            return 3

    backend = LimitedBackend(np.zeros((8, 300, 6), np.float32))
    det = YoloDetector(backend, input_size=32)
    out = det.detect([np.zeros((40, 40, 3), np.uint8)] * 7)
    assert len(out) == 7
    assert [c[0] for c in backend.calls] == [3, 3, 1]


def test_color_blob_detector_finds_synthetic_people() -> None:
    size = (240, 320)
    frames = list(synthetic_frames(fps=2.0, size=size))
    dets = ColorBlobDetector(min_area=30).detect(frames)
    for i, frame_dets in enumerate(dets):
        t = i / 2.0
        truth = [b for a in SCRIPT if (b := actor_box(a, t, size)) is not None]
        assert len(frame_dets) == len(truth)
        for gt in truth:
            best = max(box_iou(np.array([gt]), np.array([d.box]))[0, 0] for d in frame_dets)
            assert best > 0.8


def test_tiny_detector_runs(tiny_settings) -> None:  # type: ignore[no-untyped-def]
    detector = build_detector(tiny_settings, "cpu")
    out = detector.detect([np.random.default_rng(0).integers(0, 255, (90, 120, 3), np.uint8)] * 2)
    assert len(out) == 2
    for dets in out:
        for d in dets:
            assert 0 <= d.x1 < d.x2 <= 120 and 0 <= d.y1 < d.y2 <= 90
