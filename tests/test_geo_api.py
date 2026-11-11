"""vidquery locate end to end with mocked backends: GeoLocator, POST /locate, the map, the VLM
second opinion and the CLI. Nothing here downloads a model or touches the network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from vidquery.api import create_app
from vidquery.cli import main as cli_main
from vidquery.config import Settings
from vidquery.geo.backends import MOCK_ANCHORS, MockGeoBackend, mock_anchor
from vidquery.geo.geometry import haversine_km
from vidquery.geo.locate import RESPONSIBLE_USE, GeoLocator
from vidquery.geo.mapview import LEAFLET_JS_SRI, render_map_html
from vidquery.geo.people import person_coverage
from vidquery.geo.vlm import VLMFrameAnswer, VLMGeolocator, consensus, parse_geo_answer
from vidquery.synthetic import synthetic_frames
from vidquery.types import Detection
from vidquery.video import write_video


@pytest.fixture
def geo_settings(mock_settings: Settings) -> Settings:
    s = mock_settings.model_copy(deep=True)
    s.geo.backend = "mock"
    s.geo.batch_size = 4
    return s


@pytest.fixture
def crowded_video(tmp_path: Path) -> Path:
    """10 s at 5 fps, brightness 200 (mock -> Sydney). From t = 5 s a saturated blob (which the
    colour-blob 'person' detector picks up) covers ~70% of the frame."""

    def frames() -> Any:
        for i in range(50):
            img = np.full((120, 160, 3), 200, np.uint8)
            if i >= 25:
                img[10:110, 10:150] = (0, 0, 255)
            yield img

    path = tmp_path / "crowded.mp4"
    write_video(frames(), path, 5.0)
    return path


class CrowdDetector:
    """Every frame is wall-to-wall people."""

    def detect(self, frames: Any) -> list[list[Detection]]:
        return [[Detection(0, 0, f.shape[1], f.shape[0], 0.9)] for f in frames]


def expected_anchor() -> tuple[float, float]:
    return MOCK_ANCHORS[mock_anchor(next(iter(synthetic_frames())))]


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_locate_synthetic_video(geo_settings: Settings, synthetic_video: Path) -> None:
    result = GeoLocator(geo_settings).locate(synthetic_video, include_frames=True)
    assert result.backend == "mock" and result.frames_sampled == 12  # 12 s at 1 fps
    assert result.frames_used + result.frames_skipped_people == result.frames_sampled
    est = result.estimate
    assert est is not None
    assert float(haversine_km(est.lat, est.lon, *expected_anchor())) <= est.radius_km
    assert est.radius_km >= geo_settings.geo.min_radius_km
    # Coordinates are rounded to the radius: 25 km -> one decimal.
    assert est.lat == round(est.lat, 1) and est.lon == round(est.lon, 1)
    # top_k=10: the mock puts 8 points at the anchor and 2 at two other cities.
    assert len(result.alternatives) == 2
    assert all(a.share < est.share for a in result.alternatives)
    assert result.frames is not None and len(result.frames) == 12
    assert result.vlm is None and RESPONSIBLE_USE in result.notice
    payload = json.loads(result.model_dump_json())
    assert payload["coverage_mass"] == pytest.approx(0.68)


def test_person_frames_are_skipped(geo_settings: Settings, crowded_video: Path) -> None:
    result = GeoLocator(geo_settings).locate(crowded_video, sample_fps=2.0, include_frames=True)
    assert result.frames_sampled == 20
    assert result.frames_skipped_people == 10 and result.frames_used == 10
    assert result.frames is not None
    skipped = [f for f in result.frames if f.skipped]
    assert all(f.timestamp >= 5.0 and (f.person_fraction or 0) > 0.5 for f in skipped)
    assert result.estimate is not None
    assert float(haversine_km(result.estimate.lat, result.estimate.lon, *MOCK_ANCHORS[3])) < 40

    geo_settings.geo.skip_person_frames = False
    assert GeoLocator(geo_settings).locate(crowded_video, sample_fps=2.0).frames_used == 20


def test_all_frames_skipped_gives_no_estimate(
    geo_settings: Settings, synthetic_video: Path, tmp_path: Path
) -> None:
    locator = GeoLocator(geo_settings, detector=CrowdDetector())
    result = locator.locate(synthetic_video)
    assert result.frames_sampled == 12 and result.frames_used == 0
    assert result.estimate is None and result.alternatives == []
    assert "No usable frames" in result.notice
    # max_frames caps long videos by lowering the sampling rate.
    geo_settings.geo.max_frames = 6
    capped = GeoLocator(geo_settings).locate(synthetic_video)
    assert capped.frames_sampled == 6 and capped.sample_fps == pytest.approx(0.5)


def test_person_coverage() -> None:
    assert person_coverage([], 100, 100) == 0.0
    full = [Detection(0, 0, 100, 100, 0.9)]
    assert person_coverage(full, 100, 100) == pytest.approx(1.0)
    overlap = [Detection(0, 0, 50, 100, 0.9), Detection(25, 0, 50, 100, 0.8)]
    assert person_coverage(overlap, 100, 100) == pytest.approx(0.5, abs=0.02)
    outside = [Detection(-50, -50, 10, 10, 0.5)]
    assert person_coverage(outside, 100, 200) == pytest.approx(0.005, abs=0.003)


def test_api_locate_and_map(geo_settings: Settings, synthetic_video: Path, tmp_path: Path) -> None:
    geo_settings.server.allowed_roots = [synthetic_video.parent]
    app = create_app(geo_settings, locator=GeoLocator(geo_settings, backend=MockGeoBackend()))
    with TestClient(app) as client:
        r = client.post("/locate", json={"video_path": str(synthetic_video), "sample_fps": 2})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["frames_sampled"] == 24 and body["estimate"]["radius_km"] >= 25
        assert body["frames"] is None and body["vlm"] is None
        assert "not intended for locating individuals" in body["notice"]

        r = client.post("/locate/map", json={"video_path": str(synthetic_video)})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        assert "leaflet@1.9.4" in r.text and LEAFLET_JS_SRI in r.text
        assert "tile.openstreetmap.org" in r.text and "OpenStreetMap" in r.text

        missing = synthetic_video.parent / "nope.mp4"
        assert client.post("/locate", json={"video_path": str(missing)}).status_code == 404
        outside = tmp_path / "x.mp4"
        assert client.post("/locate", json={"video_path": str(outside)}).status_code == 403
        assert client.post("/locate", json={"video_path": "a", "sample_fps": 0}).status_code == 422


def test_api_builds_locator_lazily(geo_settings: Settings, synthetic_video: Path) -> None:
    with TestClient(create_app(geo_settings)) as client:
        r = client.post("/locate", json={"video_path": str(synthetic_video)})
        assert r.status_code == 200 and r.json()["backend"] == "mock"


def test_vlm_second_opinion(geo_settings: Settings, synthetic_video: Path) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        content = body["messages"][0]["content"]
        assert [c["type"] for c in content] == ["image_url", "text"]
        assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        return chat_response(
            '```json\n{"country": "France", "city": "Paris", '
            '"clues": ["French street signs", "Haussmann facades"]}\n```'
        )

    geo_settings.geo.vlm.frames = 2
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://vlm/v1")
    vlm = VLMGeolocator(geo_settings.verifier, geo_settings.geo.vlm, client=client)
    locator = GeoLocator(geo_settings, vlm=vlm)
    plain = locator.locate(synthetic_video, vlm=False)
    result = locator.locate(synthetic_video, vlm=True)
    assert len(seen) == 2  # two most confident frames, one image per request
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen[0]["model"] == geo_settings.verifier.model
    op = result.vlm
    assert op is not None and op.country == "France" and op.city == "Paris"
    assert op.frames_asked == op.frames_answered == 2 and op.agreement == 1.0
    assert op.clues == ["French street signs", "Haussmann facades"]
    # Shown alongside, never mixed in: the GeoCLIP estimate is unchanged.
    assert result.estimate == plain.estimate


def test_vlm_failure_is_reported_not_fatal(geo_settings: Settings, synthetic_video: Path) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(503)), base_url="http://vlm/v1"
    )
    vlm = VLMGeolocator(geo_settings.verifier, geo_settings.geo.vlm, client=client)
    result = GeoLocator(geo_settings, vlm=vlm).locate(synthetic_video, vlm=True)
    assert result.estimate is not None
    assert result.vlm is not None and result.vlm.frames_answered == 0 and result.vlm.error


def test_parse_geo_answer_and_consensus() -> None:
    assert parse_geo_answer('{"country": "Japan", "city": null, "clues": ["kanji"]}') == (
        "Japan",
        None,
        ["kanji"],
    )
    country, city, clues = parse_geo_answer(
        'Sure! <think>hmm</think> {"country": "unknown", "city": "Lyon", "clues": "a sign"}'
    )
    assert country is None and city is None and clues == ["a sign"]
    with pytest.raises(ValueError):
        parse_geo_answer("I cannot tell.")

    def ans(country: str | None, city: str | None = None, *clues: str) -> VLMFrameAnswer:
        return VLMFrameAnswer(
            frame_idx=0, timestamp=0, country=country, city=city, clues=list(clues)
        )

    answers = [
        ans("Germany", None, "German signs"),
        ans("germany", "Berlin", "TV tower", "german signs"),
        ans("Austria", "Vienna"),
        VLMFrameAnswer(frame_idx=3, timestamp=3, error="timeout"),
    ]
    country, city, clues, agreement = consensus(answers)
    assert (country, city) == ("Germany", "Berlin")
    assert clues == ["German signs", "TV tower"] and agreement == pytest.approx(2 / 3)
    assert consensus([ans(None)]) == (None, None, [], 0.0)
    assert consensus([]) == (None, None, [], None)


def test_map_escapes_untrusted_text(geo_settings: Settings, synthetic_video: Path) -> None:
    result = GeoLocator(geo_settings).locate(synthetic_video)
    result = result.model_copy(update={"video": "/data/</script><script>alert(1)</script>.mp4"})
    page = render_map_html(result)
    assert "</script><script>alert(1)" not in page
    assert page.count("</script>") == 3  # leaflet.js, the JSON block and the page script
    assert "maxZoom" in page and "L.circle" in page and "L.marker" not in page


def test_cli_locate(
    geo_settings: Settings,
    synthetic_video: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(geo_settings.model_dump(mode="json")))
    out_map, out_json = tmp_path / "map.html", tmp_path / "loc.json"
    code = cli_main(
        [
            "locate",
            str(synthetic_video),
            "-c",
            str(cfg),
            "--fps",
            "2",
            "--frames",
            "--map",
            str(out_map),
            "--out",
            str(out_json),
            "--no-vlm",
        ]
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(out_json.read_text())
    assert printed["frames_sampled"] == 24 and len(printed["frames"]) == 24
    assert "L.circle" in out_map.read_text()
