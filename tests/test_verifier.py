"""VLM verifier against a mocked OpenAI-compatible server (no network, no vLLM)."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from vidquery.config import Settings, VerifierConfig
from vidquery.pipeline import VidQuery
from vidquery.types import Hit, Segment
from vidquery.verify import VLMVerifier, parse_p_yes


def _response(token: str, p: float, alt: str | None = None) -> dict:  # type: ignore[type-arg]
    top = [{"token": token, "logprob": math.log(p)}]
    if alt:
        top.append({"token": alt, "logprob": math.log(1 - p)})
    return {
        "choices": [
            {
                "message": {"content": token},
                "logprobs": {
                    "content": [{"token": token, "logprob": math.log(p), "top_logprobs": top}]
                },
            }
        ]
    }


def _dominant_red(data_url: str) -> bool:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    b, g, r = img.reshape(-1, 3).mean(axis=0)
    return bool(r > b + 20 and r > g + 20)


def red_judge(requests: list[dict]) -> httpx.MockTransport:  # type: ignore[type-arg]
    """A fake VLM that answers 'yes' only for reddish images."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "fake"}]})
        body = json.loads(request.content)
        requests.append(body)
        image = body["messages"][0]["content"][0]["image_url"]["url"]
        if _dominant_red(image):
            return httpx.Response(200, json=_response("Yes", 0.9, "No"))
        return httpx.Response(200, json=_response("No", 0.8, "Yes"))

    return httpx.MockTransport(handler)


def test_parse_p_yes_variants() -> None:
    assert parse_p_yes(_response("Yes", 0.9, "No")) == pytest.approx(0.9)
    assert parse_p_yes(_response(" no", 0.75, " yes")) == pytest.approx(0.25)
    text_only = {"choices": [{"message": {"content": "Yes."}}]}
    assert parse_p_yes(text_only) == 1.0
    assert parse_p_yes({"choices": [{"message": {"content": "No"}}]}) == 0.0
    assert parse_p_yes({"choices": [{"message": {"content": "<think>"}}]}) == 0.5


def test_request_shape() -> None:
    v = VLMVerifier(
        VerifierConfig(model="Qwen/Qwen3.5-4B", max_image_side=64),
        client=httpx.Client(transport=red_judge([]), base_url="http://x/v1"),
    )
    body = v.build_request("a man in red", np.zeros((300, 100, 3), np.uint8))
    assert body["model"] == "Qwen/Qwen3.5-4B"
    assert body["max_tokens"] == 1 and body["logprobs"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "a man in red" in body["messages"][0]["content"][1]["text"]
    assert v.health()


def _seg(score: float, t: float) -> Segment:
    hit = Hit(
        id=int(t),
        video_id="v",
        frame_idx=int(t),
        timestamp=t,
        kind="crop",
        score=score,
        box=[0, 0, 4, 4],
    )
    return Segment(
        video_id="v", start=t, end=t + 1, score=score, retrieval_score=score, num_hits=1, hits=[hit]
    )


def test_rerank_promotes_verified_segment() -> None:
    cfg = VerifierConfig(top_n=3, alpha=0.8)
    v = VLMVerifier(cfg, client=httpx.Client(transport=red_judge([]), base_url="http://x/v1"))
    images = {0: (255, 0, 0), 10: (0, 0, 255), 20: (0, 255, 0)}  # BGR: blue, red, green
    segs = [_seg(0.30, 0), _seg(0.29, 10), _seg(0.28, 20), _seg(0.1, 30)]
    out = v.rerank(
        "red", segs, lambda h: np.full((32, 32, 3), images.get(int(h.timestamp), 0), np.uint8)
    )
    assert out[0].start == 10 and out[0].verifier_score == pytest.approx(0.9)
    assert out[-1].start == 30 and out[-1].verifier_score is None  # beyond top_n untouched
    assert segs[0].verifier_score is None  # inputs not mutated

    cfg_filter = VerifierConfig(top_n=3, mode="filter", threshold=0.5)
    vf = VLMVerifier(
        cfg_filter, client=httpx.Client(transport=red_judge([]), base_url="http://x/v1")
    )
    out = vf.rerank(
        "red", segs, lambda h: np.full((32, 32, 3), images.get(int(h.timestamp), 0), np.uint8)
    )
    assert [s.start for s in out] == [10, 30]


def test_server_errors_keep_retrieval_order() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(503, text="overloaded"))
    v = VLMVerifier(
        VerifierConfig(top_n=2), client=httpx.Client(transport=transport, base_url="http://x/v1")
    )
    segs = [_seg(0.3, 0), _seg(0.2, 10)]
    out = v.rerank("red", segs, lambda h: np.zeros((8, 8, 3), np.uint8))
    assert [s.start for s in out] == [0, 10]
    assert not v.health()


def test_pipeline_with_verifier(mock_settings: Settings, synthetic_video: Path) -> None:
    """'magenta' scores red and blue crops equally; the VLM breaks the tie toward red."""
    requests: list[dict] = []  # type: ignore[type-arg]
    settings = mock_settings.model_copy(deep=True)
    settings.verifier.enabled = True
    settings.verifier.top_n = 5
    settings.query.score_margin = 0.25
    settings.query.kind = "crop"
    settings.query.merge_gap_s = 0.6
    verifier = VLMVerifier(
        settings.verifier,
        client=httpx.Client(transport=red_judge(requests), base_url="http://x/v1"),
    )
    engine = VidQuery(settings, verifier=verifier)
    engine.ingest(synthetic_video)
    plain = engine.query("magenta", verify=False)
    verified = engine.query("magenta")
    assert verified.verified and requests
    assert verified.segments[0].verifier_score == pytest.approx(0.9)
    assert verified.segments[0].start < 4.0  # the red actor's time range
    assert {round(s.start, 2) for s in plain.segments} == {
        round(s.start, 2) for s in verified.segments
    }
