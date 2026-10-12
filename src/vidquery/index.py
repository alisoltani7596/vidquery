"""FAISS vector index with per-vector metadata.

Frames and person crops live in two FAISS indexes that share one id space and one metadata
table, so a query can target crops, frames or both without post-filtering. Vectors are
L2-normalised, so inner product equals cosine similarity.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from vidquery.types import Hit, Record, RecordKind, VideoEntry

INDEX_FORMAT_VERSION = 1
KINDS: tuple[RecordKind, ...] = ("frame", "crop")


def _new_faiss(dim: int, factory: str) -> Any:
    base = faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)
    return faiss.IndexIDMap2(base)


class VectorIndex:
    def __init__(self, dim: int, factory: str = "Flat", encoder_name: str = "") -> None:
        self.dim = dim
        self.factory = factory
        self.encoder_name = encoder_name
        self._indexes: dict[RecordKind, Any] = {k: _new_faiss(dim, factory) for k in KINDS}
        self.records: dict[int, Record] = {}
        self.videos: dict[str, VideoEntry] = {}
        self._next_id = 0

    # ------------------------------------------------------------------ mutation
    def add(self, vectors: np.ndarray, records: Sequence[Record]) -> None:
        """Add vectors with records whose ``id`` must come from :meth:`allocate_ids`."""
        if len(vectors) != len(records):
            raise ValueError("vectors and records must have the same length")
        if len(records) == 0:
            return
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors of shape [N, {self.dim}], got {vectors.shape}")
        for kind in KINDS:
            sel = [i for i, r in enumerate(records) if r.kind == kind]
            if sel:
                ids = np.asarray([records[i].id for i in sel], dtype=np.int64)
                self._indexes[kind].add_with_ids(vectors[sel], ids)
        for r in records:
            self.records[r.id] = r

    def allocate_ids(self, n: int) -> list[int]:
        start = self._next_id
        self._next_id += n
        return list(range(start, start + n))

    def register_video(self, entry: VideoEntry) -> None:
        self.videos[entry.video_id] = entry

    def remove_video(self, video_id: str) -> int:
        ids = [rid for rid, r in self.records.items() if r.video_id == video_id]
        if ids:
            sel = faiss.IDSelectorBatch(np.asarray(ids, dtype=np.int64))
            for idx in self._indexes.values():
                idx.remove_ids(sel)
            for rid in ids:
                del self.records[rid]
        self.videos.pop(video_id, None)
        return len(ids)

    # ------------------------------------------------------------------ queries
    def __len__(self) -> int:
        return len(self.records)

    def count(self, kind: RecordKind | None = None) -> int:
        if kind is None:
            return len(self.records)
        return int(self._indexes[kind].ntotal)

    def search(
        self,
        query: np.ndarray,
        k: int,
        kind: str = "both",
        video_ids: Iterable[str] | None = None,
        frame_weight: float = 1.0,
    ) -> list[Hit]:
        """Top-``k`` hits for one normalised query vector, best first."""
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"Query has dim {q.shape[1]}, index has dim {self.dim}")
        kinds: tuple[RecordKind, ...] = KINDS if kind == "both" else (kind,)  # type: ignore[assignment]
        wanted = set(video_ids) if video_ids is not None else None
        # Over-fetch when filtering by video so k results survive the filter.
        fetch = k if wanted is None else k * 8
        scored: list[tuple[float, Record]] = []
        for kd in kinds:
            idx = self._indexes[kd]
            n = int(idx.ntotal)
            if n == 0:
                continue
            scores, ids = idx.search(q, min(fetch, n))
            weight = frame_weight if kd == "frame" else 1.0
            for s, i in zip(scores[0], ids[0], strict=True):
                if i < 0:
                    continue
                rec = self.records[int(i)]
                if wanted is not None and rec.video_id not in wanted:
                    continue
                scored.append((float(s) * weight, rec))
        scored.sort(key=lambda t: -t[0])
        return [Hit.from_record(r, s) for s, r in scored[:k]]

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        """Atomically write the index to ``directory``."""
        directory = Path(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".tmp-index-", dir=directory.parent))
        try:
            for kind, idx in self._indexes.items():
                faiss.write_index(idx, str(tmp / f"{kind}.faiss"))
            recs = sorted(self.records.values(), key=lambda r: r.id)
            boxes = np.array(
                [r.box if r.box is not None else (np.nan,) * 4 for r in recs], dtype=np.float64
            ).reshape(-1, 4)
            video_ids = sorted({r.video_id for r in recs})
            vid_lookup = {v: i for i, v in enumerate(video_ids)}
            np.savez_compressed(
                tmp / "records.npz",
                id=np.array([r.id for r in recs], dtype=np.int64),
                video=np.array([vid_lookup[r.video_id] for r in recs], dtype=np.int32),
                frame_idx=np.array([r.frame_idx for r in recs], dtype=np.int64),
                timestamp=np.array([r.timestamp for r in recs], dtype=np.float64),
                kind=np.array([KINDS.index(r.kind) for r in recs], dtype=np.uint8),
                box=boxes,
                det_score=np.array(
                    [np.nan if r.det_score is None else r.det_score for r in recs],
                    dtype=np.float64,
                ),
            )
            info = {
                "format_version": INDEX_FORMAT_VERSION,
                "dim": self.dim,
                "factory": self.factory,
                "encoder": self.encoder_name,
                "next_id": self._next_id,
                "record_video_ids": video_ids,
                "videos": [v.model_dump() for v in self.videos.values()],
                "saved_at": time.time(),
            }
            (tmp / "info.json").write_text(json.dumps(info, indent=2))
            if directory.exists():
                backup = directory.with_name(directory.name + ".old")
                shutil.rmtree(backup, ignore_errors=True)
                os.replace(directory, backup)
                os.replace(tmp, directory)
                shutil.rmtree(backup, ignore_errors=True)
            else:
                os.replace(tmp, directory)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @classmethod
    def load(cls, directory: str | Path) -> VectorIndex:
        directory = Path(directory)
        info = json.loads((directory / "info.json").read_text())
        if info.get("format_version") != INDEX_FORMAT_VERSION:
            raise ValueError(f"Unsupported index format in {directory}")
        index = cls(int(info["dim"]), info["factory"], info.get("encoder", ""))
        for kind in KINDS:
            index._indexes[kind] = faiss.read_index(str(directory / f"{kind}.faiss"))
        data = np.load(directory / "records.npz")
        video_ids: list[str] = info["record_video_ids"]
        for i in range(len(data["id"])):
            box = data["box"][i]
            det = float(data["det_score"][i])
            rec = Record(
                id=int(data["id"][i]),
                video_id=video_ids[int(data["video"][i])],
                frame_idx=int(data["frame_idx"][i]),
                timestamp=float(data["timestamp"][i]),
                kind=KINDS[int(data["kind"][i])],
                box=None if np.isnan(box).any() else tuple(float(v) for v in box),  # type: ignore[arg-type]
                det_score=None if np.isnan(det) else det,
            )
            index.records[rec.id] = rec
        index.videos = {v["video_id"]: VideoEntry(**v) for v in info["videos"]}
        index._next_id = int(info["next_id"])
        return index

    @staticmethod
    def exists(directory: str | Path) -> bool:
        return (Path(directory) / "info.json").exists()
