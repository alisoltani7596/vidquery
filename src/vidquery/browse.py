"""Browse query results while writing ``queries.jsonl``: print the top segments, save sheets.

    vidquery browse "a woman pushing a stroller" "a man in a red jacket" --out results/browse
    vidquery browse --queries my_queries.txt -k 10
    vidquery browse --timeline MOT17-09 --every 1      # one frame per second, with timestamps

Each query gives one JPEG contact sheet (the best frame of each top segment, boxes drawn), and
each ``--timeline`` video one sheet of frames at fixed steps. Never one file per hit: the
/scratch file quota is tight. The printed JSONL template leaves ``video_id`` and ``gt`` empty on
purpose: take them from watching the video (the timeline sheets help), not from the retrieved
segments, or the evaluation would be graded against the system's own output.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from vidquery.pipeline import QueryResult, VidQuery
from vidquery.types import Hit, Segment

BOX_COLOR = (0, 220, 255)  # BGR amber, as in the demo renderer
BEST_COLOR = (60, 60, 255)  # BGR red: the segment's best-scoring box
BG = (24, 24, 24)
FG = (240, 240, 240)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "query"


def read_queries(path: str | Path) -> list[str]:
    """One query per line (``#`` comments allowed), or JSONL lines with a ``query`` field."""
    queries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(str(json.loads(line)["query"]) if line.startswith("{") else line)
    return queries


def _caption_bar(text: str, width: int, height: int = 24, scale: float = 0.5) -> np.ndarray:
    bar = np.full((height, width, 3), BG, np.uint8)
    cv2.putText(bar, text, (6, height - 7), FONT, scale, FG, 1, cv2.LINE_AA)
    return bar


def _boxes_at(segment: Segment, t: float, window: float = 0.6) -> list[Hit]:
    """Person boxes from the segment's sampled frame closest to ``t``."""
    boxed = segment.boxes()
    if not boxed:
        return []
    nearest = min(boxed, key=lambda h: abs(h.timestamp - t))
    if abs(nearest.timestamp - t) > window:
        return []
    return [h for h in boxed if h.frame_idx == nearest.frame_idx]


def make_tile(
    frame: np.ndarray, caption: str, width: int, boxes: Sequence[Hit] = (), best_id: int = -1
) -> np.ndarray:
    """Downscaled frame with boxes drawn at tile resolution and a caption bar on top."""
    scale = width / frame.shape[1]
    tile = cv2.resize(
        frame, (width, max(1, round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA
    )
    for hit in boxes:
        assert hit.box is not None
        x1, y1, x2, y2 = (round(v * scale) for v in hit.box)
        color = BEST_COLOR if hit.id == best_id else BOX_COLOR
        cv2.rectangle(tile, (x1, y1), (x2, y2), color, 2 if hit.id == best_id else 1)
    return np.vstack([_caption_bar(caption, width), tile])


def contact_sheet(tiles: Sequence[np.ndarray], title: str, cols: int) -> np.ndarray:
    """Grid of tiles (padded to the tallest) under a title bar."""
    if not tiles:
        return _caption_bar(f"{title}: no results", 640, 32, 0.6)
    cols = max(1, min(cols, len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    cell_w = max(t.shape[1] for t in tiles)
    cell_h = max(t.shape[0] for t in tiles)
    gap = 4
    sheet = np.full((rows * (cell_h + gap) + gap, cols * (cell_w + gap) + gap, 3), BG, np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = gap + r * (cell_h + gap), gap + c * (cell_w + gap)
        sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return np.vstack([_caption_bar(title, sheet.shape[1], 32, 0.6), sheet])


def _save(image: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise OSError(f"Cannot write {path}")
    return path


def render_query_sheet(
    engine: VidQuery, result: QueryResult, path: Path, tile_width: int = 480, cols: int = 5
) -> Path:
    tiles = []
    for rank, seg in enumerate(result.segments, 1):
        best = seg.best_hit
        frame = engine.source_for(seg.video_id).read_frame(best.frame_idx)
        caption = (
            f"#{rank} {seg.video_id} {seg.start:.1f}-{seg.end:.1f}s "
            f"s={seg.score:.3f} n={seg.num_hits}"
        )
        boxes = _boxes_at(seg, best.timestamp)
        tiles.append(make_tile(frame, caption, tile_width, boxes, best.id))
    title = f'"{result.query}"  top {len(tiles)} segments'
    return _save(contact_sheet(tiles, title, cols), path)


def render_timeline(
    engine: VidQuery, video_id: str, path: Path, every_s: float = 1.0, tile_width: int = 320
) -> Path:
    """Frames every ``every_s`` seconds of one ingested video, captioned with timestamps."""
    src = engine.source_for(video_id)
    info = src.info
    tiles = []
    for t in np.arange(0.0, info.duration, every_s):
        idx = min(info.num_frames - 1, round(float(t) * info.fps))
        tiles.append(make_tile(src.read_frame(idx), f"{video_id} t={t:.1f}s", tile_width))
    title = f"{video_id}: every {every_s:g}s of {info.duration:.1f}s ({info.fps:g} fps)"
    return _save(contact_sheet(tiles, title, cols=8), path)


def format_result(result: QueryResult, sheet: Path) -> str:
    lines = [f'== "{result.query}" -> {sheet}']
    if not result.segments:
        lines.append("   no results (is the index empty?)")
    for rank, seg in enumerate(result.segments, 1):
        best = seg.best_hit
        box = "" if best.box is None else " box=[" + ", ".join(f"{v:.0f}" for v in best.box) + "]"
        lines.append(
            f"{rank:3d}. {seg.video_id}  {seg.start:7.2f}s - {seg.end:7.2f}s  "
            f"score={seg.score:.4f}  hits={seg.num_hits}  best t={best.timestamp:.2f}s "
            f"{best.kind}{box}"
        )
    template = {"id": slugify(result.query), "query": result.query, "video_id": "", "gt": []}
    lines.append(f"   template: {json.dumps({**template, 'split': 'val'})}")
    return "\n".join(lines)


def browse(
    engine: VidQuery,
    queries: Sequence[str],
    out_dir: Path,
    k: int = 10,
    video_ids: list[str] | None = None,
    timelines: Sequence[str] = (),
    every_s: float = 1.0,
) -> list[Path]:
    """Print results for each query and write the sheets; returns the written paths."""
    written: list[Path] = []
    for i, text in enumerate(queries, 1):
        result = engine.query(text, k, verify=False, video_ids=video_ids)
        sheet = render_query_sheet(engine, result, out_dir / f"{i:02d}-{slugify(text)}.jpg")
        print(format_result(result, sheet), flush=True)
        written.append(sheet)
    if timelines:
        index = engine.index
        known = sorted(index.videos) if index is not None else []
        wanted = known if list(timelines) == ["all"] else list(timelines)
        for vid in wanted:
            if vid not in known:
                raise SystemExit(f"Unknown video {vid!r}; ingested: {', '.join(known)}")
            path = render_timeline(engine, vid, out_dir / f"timeline-{vid}.jpg", every_s)
            print(f"timeline {vid} -> {path}", flush=True)
            written.append(path)
    return written
