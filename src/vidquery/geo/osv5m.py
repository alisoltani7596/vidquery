"""Build a small, fixed val/test benchmark from the OpenStreetView-5M test set.

OSV-5M (Astruc et al., CVPR 2024; https://huggingface.co/datasets/osv5m/osv5m) is released under
CC BY-SA 4.0. Only its ``test.csv`` and test image zips are used; images are read from the zips
and never extracted or redistributed, and nothing from the dataset is committed to the repo.

The split is made by Mapillary *sequence* so near-duplicate frames of one drive never end up in
both val and test. Two manifests are written:

* ``manifest-images.jsonl``: one image per sequence, each image its own clip (the headline);
* ``manifest-sequences.jsonl``: several images of one sequence as a clip (a stand-in for the
  frames of a video), when the downloaded shards contain enough of them.

    python scripts/osv5m_subset.py --out-dir $SCRATCH/vidquery/data/osv5m --download
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from vidquery.geo.geometry import haversine_km, spherical_mean

log = logging.getLogger(__name__)

REPO_ID = "osv5m/osv5m"
LICENSE = "CC BY-SA 4.0"
TEST_SHARDS = ("00", "01", "02", "03", "04")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def download(out_dir: Path, shards: Sequence[str]) -> tuple[Path, list[Path]]:
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    kw: dict[str, Any] = {"repo_id": REPO_ID, "repo_type": "dataset", "local_dir": out_dir}
    csv_path = Path(hf_hub_download(filename="test.csv", **kw))
    zips = [
        Path(hf_hub_download(filename=f"{s}.zip", subfolder="images/test", **kw)) for s in shards
    ]
    return csv_path, zips


def index_zip_members(zip_paths: Iterable[Path]) -> dict[str, tuple[Path, str]]:
    """Image id (file stem) -> (zip path, member name)."""
    index: dict[str, tuple[Path, str]] = {}
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                pp = PurePosixPath(name)
                if pp.suffix.lower() in IMAGE_SUFFIXES:
                    index[pp.stem] = (zp, name)
    return index


def read_rows(csv_path: Path, ids: set[str] | None = None) -> list[dict[str, Any]]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if ids is not None and r["id"] not in ids:
                continue
            rows.append(
                {
                    "id": r["id"],
                    "lat": float(r["latitude"]),
                    "lon": float(r["longitude"]),
                    "sequence": r.get("sequence") or r["id"],
                    "country": r.get("country") or "",
                    "captured_at": r.get("captured_at") or "",
                }
            )
    return rows


def split_sequences(sequences: Iterable[str], val_fraction: float, seed: int) -> dict[str, str]:
    seqs = sorted(set(sequences))
    order = np.random.default_rng(seed).permutation(len(seqs))
    n_val = round(val_fraction * len(seqs))
    return {seqs[j]: ("val" if rank < n_val else "test") for rank, j in enumerate(order)}


def make_subset(
    rows: Sequence[dict[str, Any]],
    n_val: int,
    n_test: int,
    seed: int = 0,
    seq_clips: int = 200,
    seq_min_frames: int = 3,
    seq_max_frames: int = 8,
    seq_max_spread_km: float = 5.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Returns (image rows, sequence rows, summary). Rows carry split, clip and mode."""
    rng = np.random.default_rng(seed + 1)
    split_of = split_sequences((r["sequence"] for r in rows), n_val / (n_val + n_test), seed)
    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        by_seq[r["sequence"]].append(r)

    images: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    for split, n in (("val", n_val), ("test", n_test)):
        seqs = sorted(s for s in by_seq if split_of[s] == split)
        # One image per sequence, then a fixed-size sample of those.
        picks = [by_seq[s][int(rng.integers(len(by_seq[s])))] for s in seqs]
        chosen = rng.choice(len(picks), size=min(n, len(picks)), replace=False)
        for j in sorted(chosen):
            images.append({**picks[j], "split": split, "clip": picks[j]["id"], "mode": "image"})

        clips = []
        for s in seqs:
            members = by_seq[s]
            if len(members) < seq_min_frames:
                continue
            if len(members) > seq_max_frames:
                idx = rng.choice(len(members), size=seq_max_frames, replace=False)
                members = [members[i] for i in sorted(idx)]
            lat, lon = spherical_mean([m["lat"] for m in members], [m["lon"] for m in members])
            spread = float(
                haversine_km(
                    lat, lon, [m["lat"] for m in members], [m["lon"] for m in members]
                ).max()
            )
            if spread <= seq_max_spread_km:
                clips.append((s, members))
        if clips:
            chosen = rng.choice(len(clips), size=min(seq_clips, len(clips)), replace=False)
            for j in sorted(chosen):
                s, members = clips[j]
                sequences += [{**m, "split": split, "clip": s, "mode": "sequence"} for m in members]

    summary = {
        "source": REPO_ID,
        "license": LICENSE,
        "seed": seed,
        "rows_available": len(rows),
        "sequences_available": len(by_seq),
        "images": dict(Counter(r["split"] for r in images)),
        "sequence_clips": {
            sp: len({r["clip"] for r in sequences if r["split"] == sp}) for sp in ("val", "test")
        },
        "sequence_images": len(sequences),
        "countries": {
            sp: len({r["country"] for r in images if r["split"] == sp}) for sp in ("val", "test")
        },
    }
    return images, sequences, summary


def write_manifest(
    path: Path, rows: Sequence[dict[str, Any]], index: dict[str, tuple[Path, str]]
) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            zp, member = index[r["id"]]
            rec = {
                "id": r["id"],
                "zip": os.path.relpath(Path(zp).resolve(), path.parent.resolve()),
                "member": member,
                "lat": r["lat"],
                "lon": r["lon"],
                "split": r["split"],
                "clip": r["clip"],
                "mode": r["mode"],
                "country": r["country"],
            }
            fh.write(json.dumps(rec) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fixed val/test subset of the OSV-5M test set")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--download", action="store_true", help="Fetch test.csv and the zips first")
    p.add_argument("--shards", nargs="+", default=list(TEST_SHARDS))
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seq-clips", type=int, default=200, help="Sequence clips per split")
    p.add_argument("--seq-min-frames", type=int, default=3)
    p.add_argument("--seq-max-frames", type=int, default=8)
    p.add_argument("--seq-max-spread-km", type=float, default=5.0)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out: Path = args.out_dir
    if args.download:
        csv_path, zips = download(out, args.shards)
    else:
        csv_path = out / "test.csv"
        zips = [out / "images" / "test" / f"{s}.zip" for s in args.shards]
    zips = [z for z in zips if z.exists()]
    if not csv_path.exists() or not zips:
        raise SystemExit(f"Need {csv_path} and at least one test zip (use --download)")
    index = index_zip_members(zips)
    rows = read_rows(csv_path, set(index))
    log.info("%d images in %d zips match test.csv", len(rows), len(zips))
    images, sequences, summary = make_subset(
        rows,
        args.val,
        args.test,
        args.seed,
        args.seq_clips,
        args.seq_min_frames,
        args.seq_max_frames,
        args.seq_max_spread_km,
    )
    summary["shards"] = [z.name for z in zips]
    write_manifest(out / "manifest-images.jsonl", images, index)
    write_manifest(out / "manifest-sequences.jsonl", sequences, index)
    (out / "subset.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0
