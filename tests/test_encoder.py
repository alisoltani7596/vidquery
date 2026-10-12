from __future__ import annotations

import numpy as np
import pytest

from vidquery.config import Settings
from vidquery.models.encoder import crop_with_margin, preprocess_images
from vidquery.models.registry import (
    build_image_encoder,
    build_text_encoder,
    component_spec,
    engine_path,
    model_slug,
    onnx_path,
)
from vidquery.models.tiny import HashTokenizer


def _solid(bgr: tuple[int, int, int], size: int = 40) -> np.ndarray:
    img = np.full((size, size, 3), 128, np.uint8)
    img[5:-5, 10:-10] = bgr
    return img


def test_preprocess_images_normalises() -> None:
    batch = preprocess_images(
        [np.zeros((10, 20, 3), np.uint8), np.full((50, 30, 3), 255, np.uint8)], 16
    )
    assert batch.shape == (2, 3, 16, 16)
    assert batch[0].max() == pytest.approx(-1.0)
    assert batch[1].min() == pytest.approx(1.0)


def test_crop_with_margin_clips() -> None:
    img = np.zeros((100, 100, 3), np.uint8)
    assert crop_with_margin(img, (10, 10, 30, 50), 0.1).shape == (48, 24, 3)
    assert crop_with_margin(img, (90, 90, 110, 110), 0.5).shape == (20, 20, 3)
    with pytest.raises(ValueError, match="Degenerate"):
        crop_with_margin(img, (100, 100, 100, 100), 0.0)


def test_tiny_siglip_encoders_shapes(tiny_settings: Settings) -> None:
    image_enc = build_image_encoder(tiny_settings, "cpu")
    text_enc = build_text_encoder(tiny_settings, "cpu")
    imgs = [np.random.default_rng(i).integers(0, 255, (50, 30, 3), np.uint8) for i in range(5)]
    emb = image_enc.encode(imgs, batch_size=2)
    temb = text_enc.encode(["a man walking left", "red jacket"])
    assert emb.shape == (5, 32) and temb.shape == (2, 32)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(temb, axis=1), 1.0, atol=1e-5)
    # Deterministic across rebuilds (seeded), which ONNX parity relies on.
    again = build_image_encoder(tiny_settings, "cpu").encode(imgs[:2])
    np.testing.assert_allclose(again, emb[:2], atol=1e-6)


def test_hash_tokenizer() -> None:
    tok = HashTokenizer(vocab_size=64, max_length=4)
    ids = tok(["Hello world", "hello  WORLD again and more"])
    assert ids.shape == (2, 4) and ids.dtype == np.int64
    assert ids[0, :2].tolist() == ids[1, :2].tolist()
    assert ids[0, 2:].tolist() == [0, 0]
    assert ids.max() < 64


def test_color_mock_has_colour_semantics(mock_settings: Settings) -> None:
    image_enc = build_image_encoder(mock_settings, "cpu")
    text_enc = build_text_encoder(mock_settings, "cpu")
    imgs = [_solid((0, 0, 230)), _solid((230, 0, 0)), _solid((0, 200, 0))]  # red, blue, green
    sims = (
        text_enc.encode(["a person in red", "someone in blue", "green shirt"])
        @ image_enc.encode(imgs).T
    )
    assert np.argmax(sims, axis=1).tolist() == [0, 1, 2]
    assert sims[0, 0] > 0.95 and sims[0, 1] < 0.05


def test_artifact_paths(tiny_settings: Settings) -> None:
    assert model_slug("google/siglip2-so400m-patch16-384") == "google--siglip2-so400m-patch16-384"
    assert model_slug("/scratch/w/yolo26s.pt") == "yolo26s"
    p = onnx_path(tiny_settings, "image_encoder")
    assert p.name == "image_encoder__tiny.onnx"
    assert engine_path(tiny_settings, "detector", "fp16").name == "detector__tiny__fp16.engine"
    spec = component_spec(tiny_settings, "text_encoder")
    x = spec.example_input(3)
    assert x.shape == (3, 16) and x.dtype == np.int64


def test_color_blob_has_no_network(mock_settings: Settings) -> None:
    with pytest.raises(ValueError, match="no network"):
        component_spec(mock_settings, "detector")
