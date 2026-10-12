from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vidquery.config import load_config, parse_overrides


def test_defaults_without_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDQUERY_CONFIG", raising=False)
    s = load_config()
    assert s.ingest.sample_fps == 2.0
    assert s.encoder.model.startswith("google/siglip2")
    assert s.detector.backend == "pytorch"


def test_yaml_env_expansion_and_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRATCH", "/scratch/me")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("index_dir: ${SCRATCH}/idx\ndetector:\n  backend: onnxruntime\n")
    s = load_config(
        cfg, ["detector.precision=fp16", "ingest.sample_fps=4", "verifier.enabled=true"]
    )
    assert s.index_dir == Path("/scratch/me/idx")
    assert s.detector.backend == "onnxruntime"
    assert s.detector.precision == "fp16"
    assert s.ingest.sample_fps == 4.0
    assert s.verifier.enabled is True


def test_parse_overrides_nested() -> None:
    assert parse_overrides(["a.b.c=1", "a.d=x"]) == {"a": {"b": {"c": 1}, "d": "x"}}
    with pytest.raises(ValueError, match=r"key\.path=value"):
        parse_overrides(["nope"])


def test_invalid_values_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(None, ["detector.backend=openvino"])
    with pytest.raises(ValidationError):
        load_config(None, ["unknown_key=1"])
    with pytest.raises(ValidationError):
        load_config(None, ["ingest.sample_fps=0"])


def test_shipped_configs_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRATCH", "/scratch/test")
    root = Path(__file__).resolve().parents[1] / "configs"
    for path in sorted(root.glob("*.yaml")):
        load_config(path)
