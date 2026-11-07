"""Retrieval evaluation against time-range ground truth.

``queries.jsonl`` holds one object per line::

    {"id": "q1", "query": "man in a blue shirt walking left",
     "video_id": "MOT17-04", "gt": [[12.0, 18.5], [40.2, 44.0]]}

Metrics (a predicted segment *hits* a ground-truth range when it is in the same video and their
temporal IoU is > ``tiou`` (default 0, i.e. any overlap)):

* ``recall@k``: fraction of queries with at least one hit among the top-k segments.
* ``segment_hit_rate@k``: fraction of all ground-truth ranges covered by a hit in the top-k.
* ``precision@k``: fraction of the top-k predicted segments that hit some ground truth.
* ``mrr``: mean reciprocal rank of the first hit (0 when none in the returned list).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from vidquery.pipeline import VidQuery
from vidquery.segments import temporal_iou, temporal_overlap
from vidquery.types import Segment

log = logging.getLogger(__name__)


Range = tuple[float, float]


class QuerySpec(BaseModel):
    """One eval query. ``gt`` is a list of [start, end] seconds in ``video_id``, or, when the
    description matches people in several videos, a mapping {video_id: [[start, end], ...]}."""

    id: str | None = None
    query: str
    video_id: str | None = None
    gt: list[Range] | dict[str, list[Range]] = Field(default_factory=list)
    split: str | None = Field(default=None, description="'val' (tuning) or 'test'")

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if isinstance(data, dict) and "gt" not in data:
            for key in ("segments", "ranges", "gt_ranges"):
                if key in data:
                    data = {**data, "gt": data.pop(key)}
                    break
        return data

    @model_validator(mode="after")
    def _check(self) -> QuerySpec:
        if isinstance(self.gt, list) and not self.video_id:
            raise ValueError("video_id is required when gt is a list of ranges")
        for _, start, end in self.ranges:
            if end < start:
                raise ValueError(f"gt range [{start}, {end}] ends before it starts")
        return self

    @property
    def ranges(self) -> list[tuple[str, float, float]]:
        """Every ground-truth range as (video_id, start, end)."""
        if isinstance(self.gt, dict):
            return [(vid, s, e) for vid, rs in self.gt.items() for s, e in rs]
        assert self.video_id is not None
        return [(self.video_id, s, e) for s, e in self.gt]

    @property
    def video_ids(self) -> list[str]:
        ids = {vid for vid, _, _ in self.ranges}
        if self.video_id:
            ids.add(self.video_id)
        return sorted(ids)


def load_queries(path: str | Path) -> list[QuerySpec]:
    specs = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            spec = QuerySpec.model_validate_json(line)
            specs.append(spec if spec.id else spec.model_copy(update={"id": f"q{i}"}))
    return specs


def _hits_range(seg: Segment, rng: tuple[str, float, float], tiou: float) -> bool:
    vid, start, end = rng
    span = (seg.start, seg.end)
    return (
        seg.video_id == vid
        and temporal_overlap(span, (start, end)) > 0
        and temporal_iou(span, (start, end)) >= tiou
    )


def is_hit(seg: Segment, spec: QuerySpec, tiou: float = 0.0) -> bool:
    return any(_hits_range(seg, rng, tiou) for rng in spec.ranges)


def score_query(
    segments: Sequence[Segment], spec: QuerySpec, ks: Sequence[int], tiou: float
) -> dict[str, Any]:
    hits = [is_hit(s, spec, tiou) for s in segments]
    first = next((i for i, h in enumerate(hits) if h), None)
    out: dict[str, Any] = {
        "id": spec.id,
        "query": spec.query,
        "first_hit_rank": None if first is None else first + 1,
        "rr": 0.0 if first is None else 1.0 / (first + 1),
    }
    for k in ks:
        top = segments[:k]
        out[f"recall@{k}"] = float(any(hits[:k]))
        out[f"precision@{k}"] = sum(hits[:k]) / len(top) if top else 0.0
        out[f"covered@{k}"] = sum(
            any(_hits_range(s, rng, tiou) for s in top) for rng in spec.ranges
        )
    out["num_gt"] = len(spec.ranges)
    return out


def evaluate(
    engine: VidQuery,
    queries: Sequence[QuerySpec],
    *,
    ks: Sequence[int] = (1, 5, 10),
    verify: bool = False,
    tiou: float = 0.0,
    per_video: bool = False,
) -> dict[str, Any]:
    """Run every query and aggregate the metrics."""
    max_k = max(ks)
    per_query = []
    latencies = []
    for spec in queries:
        t0 = time.perf_counter()
        result = engine.query(
            spec.query,
            max_segments=max_k,
            verify=verify,
            video_ids=spec.video_ids if per_video else None,
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        row = score_query(result.segments, spec, ks, tiou)
        row["latency_ms"] = latencies[-1]
        row["top_segments"] = [
            {
                "video_id": s.video_id,
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "score": round(s.score, 4),
                "verifier": s.verifier_score,
            }
            for s in result.segments[:max_k]
        ]
        per_query.append(row)
    n = max(1, len(per_query))
    total_gt = max(1, sum(r["num_gt"] for r in per_query))
    summary: dict[str, Any] = {"num_queries": len(per_query), "verify": verify, "tiou": tiou}
    for k in ks:
        summary[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in per_query) / n
        summary[f"precision@{k}"] = sum(r[f"precision@{k}"] for r in per_query) / n
        summary[f"segment_hit_rate@{k}"] = sum(r[f"covered@{k}"] for r in per_query) / total_gt
    summary["mrr"] = sum(r["rr"] for r in per_query) / n
    if latencies:
        lat = sorted(latencies)
        summary["latency_p50_ms"] = lat[len(lat) // 2]
        summary["latency_p95_ms"] = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    return {"summary": summary, "per_query": per_query}


def results_markdown(runs: Sequence[dict[str, Any]], ks: Sequence[int] = (1, 5, 10)) -> str:
    head = (
        ["setting"]
        + [f"R@{k}" for k in ks]
        + [f"SegHit@{max(ks)}", f"P@{max(ks)}", "MRR", "p50 ms"]
    )
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for run in runs:
        s = run["summary"]
        name = "with VLM verifier" if s["verify"] else "embedding only"
        cells = (
            [name]
            + [f"{s[f'recall@{k}']:.3f}" for k in ks]
            + [
                f"{s[f'segment_hit_rate@{max(ks)}']:.3f}",
                f"{s[f'precision@{max(ks)}']:.3f}",
                f"{s['mrr']:.3f}",
                f"{s.get('latency_p50_ms', 0):.1f}",
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_results(
    runs: Sequence[dict[str, Any]],
    json_path: Path,
    md_path: Path | None,
    ks: Sequence[int] = (1, 5, 10),
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(list(runs), indent=2))
    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(results_markdown(runs, ks) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate retrieval against queries.jsonl")
    p.add_argument("--queries", required=True, help="queries.jsonl (query, video_id, gt ranges)")
    p.add_argument("--config", "-c", help="YAML config (default: $VIDQUERY_CONFIG)")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="key.path=value")
    p.add_argument(
        "--verifier",
        choices=["off", "on", "both"],
        default="both",
        help="Evaluate without / with the VLM verifier, or both (default; skips 'with' when the "
        "verifier endpoint is unreachable)",
    )
    p.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--tiou", type=float, default=0.0, help="Temporal IoU needed for a hit")
    p.add_argument("--per-video", action="store_true", help="Search only the query's video(s)")
    p.add_argument("--split", help="Only queries with this split (e.g. val or test)")
    p.add_argument("--out", default="results/eval.json")
    p.add_argument("--md", default="results/eval.md")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    from vidquery.config import load_config

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_config(args.config, args.overrides)
    engine = VidQuery(settings)
    if engine.index is None:
        raise SystemExit(f"No index at {settings.index_dir}; ingest videos first")
    queries = load_queries(args.queries)
    if args.split:
        queries = [q for q in queries if q.split == args.split]
        if not queries:
            raise SystemExit(f"No queries with split={args.split!r} in {args.queries}")
    missing = sorted({v for q in queries for v in q.video_ids} - set(engine.index.videos))
    if missing:
        log.warning("Queries reference videos that are not indexed: %s", missing)
    modes = {"off": [False], "on": [True], "both": [False, True]}[args.verifier]
    runs = []
    for verify in modes:
        if verify and not engine.verifier.health():
            msg = f"Verifier at {settings.verifier.base_url} is unreachable"
            if args.verifier == "on":
                raise SystemExit(msg)
            log.warning("%s; skipping the verified run", msg)
            continue
        log.info("Evaluating %d queries (verify=%s)", len(queries), verify)
        runs.append(
            evaluate(
                engine, queries, ks=args.ks, verify=verify, tiou=args.tiou, per_video=args.per_video
            )
        )
    save_results(runs, Path(args.out), Path(args.md) if args.md else None, args.ks)
    print(results_markdown(runs, args.ks))
    engine.close()
    return 0
