from __future__ import annotations

import os
from pathlib import Path

import pytest

# Never touch the network from tests: tiny/mock models only.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vidquery.config import Settings
from vidquery.synthetic import make_synthetic_video


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _cuda_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """12 s, 10 fps, 240x320 video with four coloured walkers (see vidquery.synthetic)."""
    return make_synthetic_video(tmp_path_factory.mktemp("video") / "synthetic.mp4")


@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    return Settings.model_validate(
        {
            "device": "cpu",
            "index_dir": str(tmp_path / "index"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "detector": {"model": "color-blob"},
            "encoder": {"model": "color-mock"},
            "ingest": {"sample_fps": 2.0, "min_box_size": 6, "prefetch_frames": 4},
        }
    )


@pytest.fixture
def tiny_settings(tmp_path: Path) -> Settings:
    return Settings.model_validate(
        {
            "device": "cpu",
            "index_dir": str(tmp_path / "index"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "detector": {"model": "tiny", "input_size": 64, "conf_threshold": 0.5},
            "encoder": {"model": "tiny"},
            "ingest": {"sample_fps": 1.0, "min_box_size": 4, "max_crops_per_frame": 5},
        }
    )
