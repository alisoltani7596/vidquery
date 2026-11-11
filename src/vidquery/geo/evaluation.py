"""Geolocation benchmark: cache per-image predictions once, tune aggregation on val, test once.

    python scripts/geo_eval.py predict --manifest m.jsonl --out preds.npz      # GPU, slow
    python scripts/geo_eval.py tune    --manifest m.jsonl --preds preds.npz \
        --params-out results/geo-params.yaml --md results/geo-val.md          # val split only
    python scripts/geo_eval.py test    --manifest m.jsonl --preds preds.npz \
        --params results/geo-params.yaml --md results/geo-eval.md             # refuses to rerun

A manifest is JSON lines, one image per line::

    {"id": "547473234108938", "zip": "images/test/00.zip", "member": "00/547473234108938.jpg",
     "lat": -16.336, "lon": 45.628, "split": "val", "clip": "547473234108938", "mode": "image"}

(``path`` may replace ``zip``/``member``; relative paths are relative to the manifest.) Images
that share a ``clip`` are aggregated together like the frames of one video; the clip's truth is
the spherical mean of its images' positions. Errors are measured from the raw (unrounded)
estimate; the coarsened output adds at most the rounding step on top.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import statistics
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from vidquery.geo.aggregate import AggregateParams, aggregate
from vidquery.geo.backends import FramePrediction
from vidquery.geo.geometry import haversine_km, spherical_mean

log = logging.getLogger(__name__)

THRESHOLDS_KM = (1, 25, 200, 750, 2500)
DEFAULT_GRID: dict[str, list[Any]] = {
    "top_k": [1, 3, 5, 10, 20, 50],
    "temperature": [0.25, 0.5, 1.0, 2.0, 4.0],
    "frame_weighting": ["uniform", "mass"],
    "cluster_km": [25.0, 100.0, 300.0, 750.0],
}
DEFAULT_MASS_GRID = [0.5, 0.6, 0.68, 0.75, 0.8, 0.9, 0.95]
TUNED_FIELDS = ("top_k", "temperature", "frame_weighting", "cluster_km", "confidence_mass")


# --- data -------------------------------------------------------------------------------------


@dataclass
class Clip:
    clip_id: str
    split: str
    mode: str
    lat: float
    lon: float
    ids: list[str]
    preds: list[FramePrediction]


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            for key in ("zip", "path"):
                if key in row and not Path(row[key]).is_absolute():
                    row[key] = str(path.parent / row[key])
            row.setdefault("clip", row["id"])
            row.setdefault("mode", "image")
            rows.append(row)
    return rows


def file_digest(path: str | Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


class ImageReader:
    """Decodes manifest images straight from their zip files (nothing is extracted)."""

    def __init__(self) -> None:
        self._zips: dict[str, zipfile.ZipFile] = {}

    def read(self, row: dict[str, Any]) -> np.ndarray | None:
        if "path" in row:
            return cv2.imread(row["path"], cv2.IMREAD_COLOR)
        zf = self._zips.get(row["zip"])
        if zf is None:
            zf = self._zips[row["zip"]] = zipfile.ZipFile(row["zip"])
        buf = np.frombuffer(zf.read(row["member"]), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def close(self) -> None:
        for zf in self._zips.values():
            zf.close()
        self._zips.clear()


def save_predictions(
    path: str | Path, ids: Sequence[str], preds: Sequence[FramePrediction]
) -> None:
    k = min(len(p.lats) for p in preds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=str),
        lats=np.stack([p.lats[:k] for p in preds]).astype(np.float32),
        lons=np.stack([p.lons[:k] for p in preds]).astype(np.float32),
        logits=np.stack([p.logits[:k] for p in preds]).astype(np.float32),
        log_norm=np.asarray([p.log_norm for p in preds], np.float64),
    )


def load_predictions(path: str | Path) -> dict[str, FramePrediction]:
    with np.load(path) as z:
        return {
            str(i): FramePrediction(la.astype(np.float64), lo.astype(np.float64), lg, float(n))
            for i, la, lo, lg, n in zip(
                z["ids"], z["lats"], z["lons"], z["logits"], z["log_norm"], strict=True
            )
        }


def build_clips(
    rows: Iterable[dict[str, Any]],
    preds: dict[str, FramePrediction],
    split: str | None = None,
) -> list[Clip]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    missing = 0
    for row in rows:
        if split is not None and row["split"] != split:
            continue
        if row["id"] not in preds:
            missing += 1
            continue
        groups[(row["mode"], row["split"], str(row["clip"]))].append(row)
    if missing:
        log.warning("%d manifest images have no prediction and are left out", missing)
    clips = []
    for (mode, sp, clip_id), members in sorted(groups.items()):
        members.sort(key=lambda r: str(r["id"]))
        lat, lon = spherical_mean([r["lat"] for r in members], [r["lon"] for r in members])
        clips.append(
            Clip(
                clip_id,
                sp,
                mode,
                lat,
                lon,
                [str(r["id"]) for r in members],
                [preds[str(r["id"])] for r in members],
            )
        )
    return clips


# --- metrics ----------------------------------------------------------------------------------


def evaluate_clips(clips: Sequence[Clip], params: AggregateParams) -> dict[str, Any]:
    """Accuracy within each threshold, median/mean error, and how often the truth falls inside
    the reported confidence radius (calibration of the radius)."""
    errors, radii, inside = [], [], []
    for clip in clips:
        agg = aggregate(clip.preds, params)
        if agg is None:
            continue
        err = float(haversine_km(agg.best.lat, agg.best.lon, clip.lat, clip.lon))
        errors.append(err)
        radii.append(agg.best.radius_km)
        inside.append(err <= agg.best.radius_km)
    n = len(errors)
    out: dict[str, Any] = {"n": n}
    for t in THRESHOLDS_KM:
        out[f"acc@{t}km"] = sum(e <= t for e in errors) / n if n else 0.0
    out["median_km"] = statistics.median(errors) if n else float("nan")
    out["mean_km"] = statistics.fmean(errors) if n else float("nan")
    out["coverage"] = sum(inside) / n if n else 0.0
    out["median_radius_km"] = statistics.median(radii) if n else float("nan")
    return out


def _grid_points(
    grid: dict[str, list[Any]], single_frame: bool, max_k: int
) -> Iterator[dict[str, Any]]:
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        point = dict(zip(keys, values, strict=True))
        if point.get("top_k", 1) > max_k:
            continue
        if single_frame and point.get("frame_weighting", "uniform") != "uniform":
            continue  # frame weights are irrelevant when every clip is one image
        yield point


@dataclass
class TuneResult:
    best: AggregateParams
    rows: list[dict[str, Any]]
    coverage_rows: list[dict[str, Any]]
    target_coverage: float


def tune(
    clips: Sequence[Clip],
    base: AggregateParams,
    grid: dict[str, list[Any]] | None = None,
    mass_grid: Sequence[float] = DEFAULT_MASS_GRID,
    target_coverage: float = 0.68,
) -> TuneResult:
    """Grid search on validation clips only.

    Stage 1 picks top_k / temperature / frame weighting / cluster bandwidth by the lowest median
    error (ties: higher accuracy at 25 km, then smaller top_k). Stage 2 keeps those and picks the
    ``confidence_mass`` whose radius covers the truth on val closest to ``target_coverage``.
    """
    if not clips:
        raise ValueError("no validation clips to tune on")
    grid = grid or DEFAULT_GRID
    single = all(len(c.preds) == 1 for c in clips)
    max_k = min(len(p.lats) for c in clips for p in c.preds)
    rows = []
    for point in _grid_points(grid, single, max_k):
        params = base.with_overrides(**point)
        rows.append({**point, **evaluate_clips(clips, params)})
    if not rows:
        raise ValueError("the grid is empty for these predictions")
    rows.sort(key=lambda r: (r["median_km"], -r["acc@25km"], r.get("top_k", 0)))
    best = base.with_overrides(**{k: rows[0][k] for k in grid if k in rows[0]})
    cov_rows = []
    for mass in mass_grid:
        m = evaluate_clips(clips, best.with_overrides(confidence_mass=mass))
        cov_rows.append({"confidence_mass": mass, **m})
    chosen = min(
        cov_rows, key=lambda r: (abs(r["coverage"] - target_coverage), r["confidence_mass"])
    )
    best = best.with_overrides(confidence_mass=chosen["confidence_mass"])
    return TuneResult(best, rows, cov_rows, target_coverage)


# --- markdown ---------------------------------------------------------------------------------


def _metric_cells(m: dict[str, Any]) -> list[str]:
    return (
        [str(m["n"])]
        + [f"{100 * m[f'acc@{t}km']:.1f}" for t in THRESHOLDS_KM]
        + [f"{m['median_km']:.0f}", f"{m['mean_km']:.0f}"]
        + [f"{100 * m['coverage']:.1f}", f"{m['median_radius_km']:.0f}"]
    )


METRIC_HEAD = (
    ["n"]
    + [f"<{t} km %" for t in THRESHOLDS_KM]
    + ["median km", "mean km", "in radius %", "median radius km"]
)


def metrics_table(rows: Sequence[tuple[str, dict[str, Any]]], first: str = "setting") -> str:
    head = [first, *METRIC_HEAD]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, m in rows:
        lines.append("| " + " | ".join([name, *_metric_cells(m)]) + " |")
    return "\n".join(lines)


def params_line(p: AggregateParams) -> str:
    return ", ".join(f"{k}={getattr(p, k)}" for k in TUNED_FIELDS) + (
        f", min_radius_km={p.min_radius_km}"
    )


def tune_markdown(res: TuneResult, clips: Sequence[Clip], top: int = 15) -> str:
    keys = [k for k in res.rows[0] if k in DEFAULT_GRID or k in TUNED_FIELDS]
    head = [*keys, "n", "<25 km %", "<200 km %", "median km", "in radius %"]
    lines = [
        f"Validation clips: {len(clips)} ({_mode_counts(clips)}). Selected by median error.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for r in res.rows[:top]:
        cells = [str(r[k]) for k in keys] + [
            str(r["n"]),
            f"{100 * r['acc@25km']:.1f}",
            f"{100 * r['acc@200km']:.1f}",
            f"{r['median_km']:.0f}",
            f"{100 * r['coverage']:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Radius calibration (target: truth inside the radius on {100 * res.target_coverage:.0f}% "
        "of val clips):",
        "",
        "| confidence_mass | in radius % | median radius km |",
        "|---|---|---|",
    ]
    for r in res.coverage_rows:
        lines.append(
            f"| {r['confidence_mass']} | {100 * r['coverage']:.1f} | {r['median_radius_km']:.0f} |"
        )
    lines += ["", f"Frozen: {params_line(res.best)}"]
    return "\n".join(lines)


def _mode_counts(clips: Sequence[Clip]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for c in clips:
        counts[c.mode] += 1
    return ", ".join(f"{n} {mode}" for mode, n in sorted(counts.items()))


# --- params file ------------------------------------------------------------------------------


def write_params(path: str | Path, params: AggregateParams, header: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump({"geo": params.to_dict()}, sort_keys=False)
    path.write_text("".join(f"# {h}\n" for h in header) + body)


def read_params(path: str | Path, base: AggregateParams | None = None) -> AggregateParams:
    data = yaml.safe_load(Path(path).read_text()) or {}
    geo = data.get("geo") or data
    if not isinstance(geo, dict):
        raise ValueError(f"{path}: expected a 'geo' mapping")
    base = base or AggregateParams()
    known = set(base.to_dict())
    return base.with_overrides(**{k: v for k, v in geo.items() if k in known})


# --- CLI --------------------------------------------------------------------------------------


def _load_rows(paths: Sequence[str]) -> list[dict[str, Any]]:
    return [row for p in paths for row in load_manifest(p)]


def cmd_predict(args: argparse.Namespace) -> int:
    from vidquery.config import load_config, resolve_device
    from vidquery.geo.backends import build_geo_backend

    settings = load_config(args.config, args.overrides)
    device = resolve_device(settings.geo.device or settings.device)
    backend = build_geo_backend(settings.geo, device)
    rows = list({r["id"]: r for r in _load_rows(args.manifest)}.values())
    if args.limit:
        rows = rows[: args.limit]
    reader = ImageReader()
    ids: list[str] = []
    preds: list[FramePrediction] = []
    batch_rows: list[dict[str, Any]] = []
    batch_imgs: list[np.ndarray] = []

    def flush() -> None:
        if batch_imgs:
            preds.extend(backend.predict(batch_imgs, args.top_k))
            ids.extend(str(r["id"]) for r in batch_rows)
            batch_rows.clear()
            batch_imgs.clear()

    for i, row in enumerate(rows):
        img = reader.read(row)
        if img is None:
            log.warning("Could not decode %s", row["id"])
            continue
        batch_rows.append(row)
        batch_imgs.append(img)
        if len(batch_imgs) >= settings.geo.batch_size:
            flush()
        if (i + 1) % 500 == 0:
            log.info("%d / %d images", i + 1, len(rows))
    flush()
    reader.close()
    save_predictions(args.out, ids, preds)
    print(f"{len(ids)} predictions (top-{args.top_k}, {backend.name}) -> {args.out}")
    return 0


def _base_params(args: argparse.Namespace) -> AggregateParams:
    from vidquery.config import load_config

    return AggregateParams.from_config(load_config(args.config, args.overrides).geo)


def cmd_tune(args: argparse.Namespace) -> int:
    rows = _load_rows(args.manifest)
    clips = build_clips(rows, load_predictions(args.preds), split=args.split)
    grid = json.loads(Path(args.grid).read_text()) if args.grid else None
    res = tune(clips, _base_params(args), grid, target_coverage=args.target_coverage)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    shas = ",".join(file_digest(m) for m in args.manifest)
    md = f"# Geolocation aggregation tuned on `{args.split}` ({stamp})\n\n" + tune_markdown(
        res, clips
    )
    write_params(
        args.params_out,
        res.best,
        [
            f"Frozen geolocation aggregation, tuned on split={args.split} ({len(clips)} clips)",
            f"preds={Path(args.preds).name} manifest_sha={shas}",
            f"tuned {stamp}; report the test split once with: scripts/geo_eval.py test",
        ],
    )
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md + "\n")
    print(md)
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    out = Path(args.md)
    if out.exists() and not args.force:
        print(
            f"{out} already exists: the test split is reported once. Pass --force only to "
            "regenerate the same frozen run (e.g. after a crash), never to try new settings."
        )
        return 2
    params = read_params(args.params, _base_params(args))
    rows = _load_rows(args.manifest)
    preds = load_predictions(args.preds)
    clips = build_clips(rows, preds, split=args.split)
    if not clips:
        raise SystemExit(f"No clips with split={args.split!r}")
    table: list[tuple[str, dict[str, Any]]] = []
    results: dict[str, Any] = {"params": params.to_dict(), "split": args.split, "modes": {}}
    for mode in sorted({c.mode for c in clips}):
        sub = [c for c in clips if c.mode == mode]
        base = evaluate_clips(sub, params.with_overrides(top_k=1))
        tuned = evaluate_clips(sub, params)
        table.append((f"{mode}: GeoCLIP top-1 (no aggregation)", base))
        table.append((f"{mode}: aggregation frozen on val", tuned))
        results["modes"][mode] = {"top1": base, "tuned": tuned}
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    md = "\n".join(
        [
            f"# Geolocation test results ({stamp})",
            "",
            f"Split `{args.split}`, {len(clips)} clips ({_mode_counts(clips)}), run once with "
            f"parameters frozen on val: {params_line(params)}.",
            f"Predictions `{Path(args.preds).name}`, manifest sha256 "
            f"{', '.join(file_digest(m) for m in args.manifest)}.",
            "",
            metrics_table(table),
            "",
            "Errors are great-circle distances from the raw estimate to the truth. 'in radius' is "
            "how often the truth falls inside the reported confidence radius.",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    print(md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Geolocation benchmark (predict / tune / test)")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", "-c", help="YAML config (default: $VIDQUERY_CONFIG)")
        sp.add_argument("--set", dest="overrides", action="append", default=[])
        sp.add_argument("--manifest", action="append", required=True, help="repeatable")

    sp = sub.add_parser("predict", help="Cache per-image top-k predictions")
    common(sp)
    sp.add_argument("--out", required=True)
    sp.add_argument("--top-k", type=int, default=50)
    sp.add_argument("--limit", type=int, default=0)
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("tune", help="Grid-search aggregation settings on the val split")
    common(sp)
    sp.add_argument("--preds", required=True)
    sp.add_argument("--split", default="val")
    sp.add_argument("--grid", help="JSON file overriding the stage-1 grid")
    sp.add_argument("--target-coverage", type=float, default=0.68)
    sp.add_argument("--params-out", default="results/geo-params.yaml")
    sp.add_argument("--md", default="results/geo-val.md")
    sp.set_defaults(func=cmd_tune)

    sp = sub.add_parser("test", help="Report the test split once with frozen settings")
    common(sp)
    sp.add_argument("--preds", required=True)
    sp.add_argument("--params", required=True)
    sp.add_argument("--split", default="test")
    sp.add_argument("--md", default="results/geo-eval.md")
    sp.add_argument("--json")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_test)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result: int = args.func(args)
    return result
