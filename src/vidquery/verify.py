"""Optional VLM verification over an OpenAI-compatible chat API (vLLM).

For each of the top segments, the best hit's crop (or frame) is sent to the VLM with a yes/no
prompt built from the query. ``p(yes)`` is read from the first token's log-probabilities when the
server returns them, falling back to parsing the text. Segments are then re-ranked by
``alpha * p_yes + (1 - alpha) * normalised_retrieval_score`` (``rerank`` mode) or dropped when
``p_yes < threshold`` (``filter`` mode).
"""

from __future__ import annotations

import base64
import logging
import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import httpx
import numpy as np

from vidquery.config import VerifierConfig
from vidquery.types import Hit, Segment

log = logging.getLogger(__name__)

ImageLoader = Callable[[Hit], np.ndarray]
NEUTRAL = 0.5


def encode_jpeg_data_url(image: np.ndarray, max_side: int, quality: int = 90) -> str:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        image = cv2.resize(
            image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _norm_token(token: str) -> str:
    return token.strip().strip(".,!:\"'").lower()


def parse_p_yes(response: dict[str, Any]) -> float:
    """Extract p(yes) from a chat completion response."""
    choice = response["choices"][0]
    logprobs = (choice.get("logprobs") or {}).get("content") or []
    if logprobs:
        first = logprobs[0]
        candidates = first.get("top_logprobs") or [first]
        p_yes = sum(math.exp(c["logprob"]) for c in candidates if _norm_token(c["token"]) == "yes")
        p_no = sum(math.exp(c["logprob"]) for c in candidates if _norm_token(c["token"]) == "no")
        if p_yes + p_no > 0:
            return p_yes / (p_yes + p_no)
    text = _norm_token((choice.get("message") or {}).get("content") or "")
    if text.startswith("yes"):
        return 1.0
    if text.startswith("no"):
        return 0.0
    log.warning("Unparseable verifier answer %r; using neutral score", text[:40])
    return NEUTRAL


class VLMVerifier:
    def __init__(self, cfg: VerifierConfig, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self.client = client or httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            timeout=cfg.timeout_s,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
        )

    def close(self) -> None:
        self.client.close()

    def health(self) -> bool:
        try:
            return self.client.get("/models", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    def build_request(self, query: str, image: np.ndarray) -> dict[str, Any]:
        prompt = self.cfg.prompt_template.format(query=query.strip())
        body: dict[str, Any] = {
            "model": self.cfg.model,
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
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.cfg.max_tokens,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 10,
        }
        if self.cfg.disable_thinking:
            # Qwen3.x chat templates think by default; a yes/no check must not.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def score_image(self, query: str, image: np.ndarray) -> float:
        resp = self.client.post("/chat/completions", json=self.build_request(query, image))
        resp.raise_for_status()
        return parse_p_yes(resp.json())

    def score_images(self, query: str, images: Sequence[np.ndarray]) -> list[float | None]:
        """p(yes) per image; None where the request failed."""

        def one(img: np.ndarray) -> float | None:
            try:
                return self.score_image(query, img)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                log.warning("Verifier request failed: %s", exc)
                return None

        if not images:
            return []
        with ThreadPoolExecutor(max_workers=min(self.cfg.max_concurrency, len(images))) as pool:
            return list(pool.map(one, images))

    def rerank(
        self, query: str, segments: Sequence[Segment], load_image: ImageLoader
    ) -> list[Segment]:
        """Verify the top ``top_n`` segments and return the re-ordered list."""
        head = [s.model_copy(deep=True) for s in segments[: self.cfg.top_n]]
        tail = [s.model_copy(deep=True) for s in segments[self.cfg.top_n :]]
        if not head:
            return []
        jobs: list[tuple[int, Hit]] = []
        for i, seg in enumerate(head):
            best = sorted(seg.hits, key=lambda h: -h.score)[: self.cfg.hits_per_segment]
            jobs.extend((i, h) for h in best)
        images: list[np.ndarray] = []
        for _, hit in jobs:
            images.append(load_image(hit))
        scores = self.score_images(query, images)

        per_seg: dict[int, list[float]] = {}
        for (i, _), p in zip(jobs, scores, strict=True):
            if p is not None:
                per_seg.setdefault(i, []).append(p)
        if not per_seg:
            log.warning("All verifier requests failed; keeping retrieval order")
            return [*head, *tail]

        r = np.array([s.retrieval_score for s in head])
        r_norm = (r - r.min()) / (r.max() - r.min()) if r.max() > r.min() else np.ones_like(r)
        for i, seg in enumerate(head):
            p_yes = max(per_seg[i]) if i in per_seg else None
            seg.verifier_score = p_yes
            p = NEUTRAL if p_yes is None else p_yes
            seg.score = float(self.cfg.alpha * p + (1 - self.cfg.alpha) * r_norm[i])

        if self.cfg.mode == "filter":
            kept = [
                s
                for s in head
                if s.verifier_score is None or s.verifier_score >= self.cfg.threshold
            ]
            # Keep retrieval order among survivors, but put verified-yes first.
            kept.sort(
                key=lambda s: -(s.verifier_score if s.verifier_score is not None else NEUTRAL)
            )
            return [*kept, *tail]
        head.sort(key=lambda s: -s.score)
        return [*head, *tail]
