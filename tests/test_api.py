from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vidquery.api import create_app
from vidquery.config import Settings


@pytest.fixture
def client(mock_settings: Settings) -> TestClient:
    with TestClient(create_app(mock_settings)) as c:
        yield c  # type: ignore[misc]


def test_health_empty(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["vectors"] == 0 and body["device"] == "cpu"
    assert body["verifier_enabled"] is False and body["verifier_reachable"] is None


def test_ingest_then_query(client: TestClient, synthetic_video: Path) -> None:
    r = client.post("/ingest", json={"video_path": str(synthetic_video), "video_id": "cam1"})
    assert r.status_code == 200, r.text
    assert r.json()["video_id"] == "cam1" and r.json()["vectors"] > 0

    r = client.post("/query", json={"text": "person in blue", "max_segments": 2})
    assert r.status_code == 200
    body = r.json()
    seg = body["segments"][0]
    assert seg["video_id"] == "cam1"
    assert 4.5 <= seg["start"] <= 5.5 and 8.5 <= seg["end"] <= 9.5
    assert any(h["box"] for h in seg["hits"])
    assert body["verified"] is False and len(body["segments"]) <= 2

    assert client.get("/health").json()["videos"] == 1
    assert [v["video_id"] for v in client.get("/videos").json()] == ["cam1"]


def test_background_ingest_job(client: TestClient, synthetic_video: Path) -> None:
    r = client.post("/ingest", json={"video_path": str(synthetic_video), "background": True})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    for _ in range(200):
        job = client.get(f"/jobs/{job_id}").json()
        if job["state"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert job["state"] == "done", job
    assert job["result"]["sampled_frames"] == 24
    assert client.get("/jobs/unknown").status_code == 404


def test_validation_and_errors(client: TestClient, tmp_path: Path) -> None:
    assert client.post("/query", json={"text": ""}).status_code == 422
    assert client.post("/query", json={"text": "x", "kind": "pixels"}).status_code == 422
    r = client.post("/ingest", json={"video_path": str(tmp_path / "missing.mp4")})
    assert r.status_code == 404
    assert client.post("/query", json={"text": "red"}).json()["segments"] == []


def test_allowed_roots(mock_settings: Settings, synthetic_video: Path, tmp_path: Path) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.server.allowed_roots = [tmp_path / "only-here"]
    with TestClient(create_app(settings)) as c:
        r = c.post("/ingest", json={"video_path": str(synthetic_video)})
        assert r.status_code == 403
