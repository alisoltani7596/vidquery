"""VLM second opinion for ``locate``: ask Qwen (via vLLM's OpenAI-compatible API) for a country,
an optional city and the visual clues it used.

The answer is reported next to the GeoCLIP estimate and never mixed into it: the two are
different kinds of evidence (a calibrated-ish distribution over a gallery vs. free text), and
keeping them apart lets a reader see when they disagree.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel, Field

from vidquery.config import GeoVLMConfig, VerifierConfig
from vidquery.verify import encode_jpeg_data_url

log = logging.getLogger(__name__)

_NULLS = {"", "null", "none", "unknown", "n/a", "na", "not sure", "unclear"}
MAX_CLUES = 8


class VLMFrameAnswer(BaseModel):
    frame_idx: int
    timestamp: float
    country: str | None = None
    city: str | None = None
    clues: list[str] = Field(default_factory=list)
    error: str | None = None


class VLMOpinion(BaseModel):
    model: str
    country: str | None = Field(None, description="Most common country among the answers.")
    city: str | None = Field(None, description="Most common city among answers for that country.")
    clues: list[str] = Field(default_factory=list)
    agreement: float | None = Field(
        None, description="Share of answered frames that named the majority country."
    )
    frames_asked: int = 0
    frames_answered: int = 0
    answers: list[VLMFrameAnswer] = Field(default_factory=list)
    error: str | None = None
    note: str = "Second opinion from a VLM; shown next to the GeoCLIP estimate, not combined."


def _clean(value: Any, max_len: int = 80) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip(" .")
    return None if text.lower() in _NULLS else text[:max_len]


def parse_geo_answer(text: str) -> tuple[str | None, str | None, list[str]]:
    """Read ``{"country", "city", "clues"}`` from a model reply (code fences and chatter are
    tolerated). Raises ValueError when no JSON object can be found."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in answer {text[:60]!r}")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("answer is not a JSON object")
    raw_clues = data.get("clues") or []
    if isinstance(raw_clues, str):
        raw_clues = [raw_clues]
    clues = [c for c in (_clean(x, 120) for x in raw_clues if isinstance(x, str)) if c]
    country = _clean(data.get("country"))
    city = _clean(data.get("city")) if country else None
    return country, city, clues[:MAX_CLUES]


def consensus(
    answers: Sequence[VLMFrameAnswer],
) -> tuple[str | None, str | None, list[str], float | None]:
    """Majority country (ties go to the earlier, more confident frame), the most common city
    among answers for that country, and the de-duplicated clues of those answers."""
    answered = [a for a in answers if a.error is None]
    named = [a for a in answered if a.country]
    if not named:
        return None, None, [], (0.0 if answered else None)
    counts = Counter(a.country.casefold() for a in named if a.country)
    top = max(counts.values())
    key = next(
        a.country.casefold() for a in named if a.country and counts[a.country.casefold()] == top
    )
    agree = [a for a in named if a.country and a.country.casefold() == key]
    cities = Counter(a.city for a in agree if a.city)
    city = cities.most_common(1)[0][0] if cities else None
    clues: list[str] = []
    seen: set[str] = set()
    for a in agree:
        for c in a.clues:
            if c.casefold() not in seen:
                seen.add(c.casefold())
                clues.append(c)
    return agree[0].country, city, clues[:MAX_CLUES], len(agree) / len(answered)


class VLMGeolocator:
    def __init__(
        self,
        endpoint: VerifierConfig,
        cfg: GeoVLMConfig,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.cfg = cfg
        self.client = client or httpx.Client(
            base_url=endpoint.base_url.rstrip("/"),
            timeout=endpoint.timeout_s,
            headers={"Authorization": f"Bearer {endpoint.api_key}"},
        )

    def close(self) -> None:
        self.client.close()

    def build_request(self, image: np.ndarray) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_jpeg_data_url(image, self.cfg.max_image_side)
                            },
                        },
                        {"type": "text", "text": self.cfg.prompt},
                    ],
                }
            ],
            "max_tokens": self.cfg.max_tokens,
            "temperature": 0.0,
        }
        if self.endpoint.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def ask(self, frame_idx: int, timestamp: float, image: np.ndarray) -> VLMFrameAnswer:
        try:
            resp = self.client.post("/chat/completions", json=self.build_request(image))
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"] or ""
            country, city, clues = parse_geo_answer(text)
            return VLMFrameAnswer(
                frame_idx=frame_idx, timestamp=timestamp, country=country, city=city, clues=clues
            )
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            log.warning("VLM geolocation request for frame %d failed: %s", frame_idx, exc)
            return VLMFrameAnswer(frame_idx=frame_idx, timestamp=timestamp, error=str(exc)[:200])

    def opinion(self, frames: Sequence[tuple[int, float, np.ndarray]]) -> VLMOpinion:
        """Ask about each (frame_idx, timestamp, BGR image), in parallel, and summarise."""
        if not frames:
            return VLMOpinion(model=self.endpoint.model, error="no usable frames")
        workers = min(self.endpoint.max_concurrency, len(frames))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answers = list(pool.map(lambda f: self.ask(*f), frames))
        country, city, clues, agreement = consensus(answers)
        answered = sum(a.error is None for a in answers)
        return VLMOpinion(
            model=self.endpoint.model,
            country=country,
            city=city,
            clues=clues,
            agreement=agreement,
            frames_asked=len(answers),
            frames_answered=answered,
            answers=answers,
            error=None if answered else (answers[0].error or "all requests failed"),
        )
