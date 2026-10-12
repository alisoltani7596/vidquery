"""CLI, retrieval evaluation, benchmark and demo rendering on CPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from vidquery.bench import main as bench_main
from vidquery.cli import main as cli_main
from vidquery.config import Settings
from vidquery.evaluation import QuerySpec, evaluate, load_queries, results_markdown, score_query
from vidquery.pipeline import VidQuery
from vidquery.synthetic import SCRIPT
from vidquery.types import Segment


def _write_config(settings: Settings, path: Path) -> Path:
    path.write_text(yaml.safe_dump(settings.model_dump(mode="json")))
    return path


def _queries(path: Path) -> Path:
    with open(path, "w") as fh:
        for a in SCRIPT:
            fh.write(
                json.dumps(
                    {
                        "query": f"a person in {a.color_name}",
                        "video_id": "synthetic",
                        "segments": [[a.start, a.end]],
                    }
                )
                + "\n"
            )
        fh.write("# comment lines are ignored\n")
    return path


def test_cli_ingest_query_videos_demo(
    mock_settings: Settings,
    synthetic_video: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _write_config(mock_settings, tmp_path / "cfg.yaml")
    assert cli_main(["ingest", str(synthetic_video), "-c", str(cfg)]) == 0
    assert "synthetic: 24 frames" in capsys.readouterr().out

    assert cli_main(["query", "person in yellow", "-c", str(cfg), "-k", "2"]) == 0
    out = capsys.readouterr().out
    assert "synthetic" in out and "box=[" in out

    assert cli_main(["query", "green", "-c", str(cfg), "--json", "--set", "query.kind=crop"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["segments"][0]["hits"][0]["kind"] == "crop"

    assert cli_main(["videos", "-c", str(cfg)]) == 0
    assert "synthetic" in capsys.readouterr().out

    gif = tmp_path / "demo.gif"
    assert cli_main(["demo", "red", "-c", str(cfg), "--out", str(gif)]) == 0
    with Image.open(gif) as im:
        assert im.n_frames > 5
    mp4 = tmp_path / "demo.mp4"
    assert cli_main(["demo", "blue", "-c", str(cfg), "--out", str(mp4), "--segments", "2"]) == 0
    assert mp4.stat().st_size > 1000


def test_cli_export_and_parity(
    tiny_settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _write_config(tiny_settings, tmp_path / "cfg.yaml")
    assert cli_main(["export", "onnx", "-c", str(cfg), "--exporter", "torchscript"]) == 0
    out = tmp_path / "parity.json"
    code = cli_main(
        [
            "parity",
            "-c",
            str(cfg),
            "--components",
            "detector",
            "image_encoder",
            "text_encoder",
            "--backends",
            "onnxruntime",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert "PARITY OK" in capsys.readouterr().out
    assert len(json.loads(out.read_text())) == 3


def test_score_query_metrics() -> None:
    spec = QuerySpec(query="q", video_id="v", gt=[(10.0, 12.0), (30.0, 31.0)])
    segs = [
        Segment(video_id="v", start=0, end=1, score=0.9, retrieval_score=0.9, num_hits=1),
        Segment(video_id="w", start=10, end=12, score=0.8, retrieval_score=0.8, num_hits=1),
        Segment(video_id="v", start=11, end=13, score=0.7, retrieval_score=0.7, num_hits=1),
    ]
    row = score_query(segs, spec, ks=(1, 5), tiou=0.0)
    assert row["recall@1"] == 0.0 and row["recall@5"] == 1.0
    assert row["first_hit_rank"] == 3 and row["rr"] == pytest.approx(1 / 3)
    assert row["covered@5"] == 1 and row["precision@5"] == pytest.approx(1 / 3)
    strict = score_query(segs, spec, ks=(5,), tiou=0.5)
    assert strict["recall@5"] == 0.0  # tIoU of (11, 13) vs (10, 12) is 1/3


def test_evaluate_on_synthetic(
    mock_settings: Settings, synthetic_video: Path, tmp_path: Path
) -> None:
    engine = VidQuery(mock_settings)
    engine.ingest(synthetic_video)
    queries = load_queries(_queries(tmp_path / "q.jsonl"))
    assert len(queries) == 4 and queries[0].id == "q0"
    run = evaluate(engine, queries, ks=(1, 5, 10))
    s = run["summary"]
    assert s["recall@1"] == 1.0 and s["segment_hit_rate@10"] == 1.0 and s["mrr"] == 1.0
    assert "embedding only" in results_markdown([run])


def test_eval_script_main(mock_settings: Settings, synthetic_video: Path, tmp_path: Path) -> None:
    VidQuery(mock_settings).ingest(synthetic_video)
    cfg = _write_config(mock_settings, tmp_path / "cfg.yaml")
    out, md = tmp_path / "eval.json", tmp_path / "eval.md"
    code = cli_main(
        [
            "eval",
            "--config",
            str(cfg),
            "--queries",
            str(_queries(tmp_path / "q.jsonl")),
            "--out",
            str(out),
            "--md",
            str(md),
        ]
    )
    assert code == 0
    runs = json.loads(out.read_text())
    assert len(runs) == 1  # verifier disabled/unreachable -> only the embedding-only run
    assert runs[0]["summary"]["recall@1"] == 1.0
    assert "R@1" in md.read_text()


def test_benchmark_cpu_smoke(tiny_settings: Settings, tmp_path: Path) -> None:
    cfg = _write_config(tiny_settings, tmp_path / "cfg.yaml")
    assert (
        cli_main(
            [
                "export",
                "onnx",
                "-c",
                str(cfg),
                "--exporter",
                "torchscript",
                "--components",
                "detector",
                "image_encoder",
            ]
        )
        == 0
    )
    out, md = tmp_path / "bench.json", tmp_path / "bench.md"
    code = bench_main(
        [
            "--config",
            str(cfg),
            "--backends",
            "pytorch",
            "onnxruntime",
            "tensorrt:fp16",
            "--batch-sizes",
            "1",
            "4",
            "--warmup",
            "1",
            "--iters",
            "3",
            "--video",
            "synthetic",
            "--out",
            str(out),
            "--md",
            str(md),
            "--no-isolate",
        ]
    )
    assert code == 0
    results = json.loads(out.read_text())
    rows = [r for r in results["components"] if "p50_ms" in r]
    assert {(r["component"], r["backend"], r["batch_size"]) for r in rows} == {
        (c, b, n)
        for c in ("detector", "image_encoder")
        for b in ("pytorch:fp32", "onnxruntime:fp32")
        for n in (1, 4)
    }
    assert all(r["p95_ms"] >= r["p50_ms"] > 0 and r["throughput"] > 0 for r in rows)
    assert len(results["ingest"]) == 2 and all(r["ingest_fps"] > 0 for r in results["ingest"])
    text = md.read_text()
    assert "| detector |" in text and "ingest FPS" in text
