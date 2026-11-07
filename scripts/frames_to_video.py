#!/usr/bin/env python
"""Convert MOTChallenge image sequences (seq/img1/*.jpg + seqinfo.ini) to mp4 files.

    # straight from the archive: frames are decoded in memory, nothing is extracted
    python scripts/frames_to_video.py --zip $SCRATCH/vidquery/data/MOT17/MOT17.zip \
        --out $SCRATCH/vidquery/data/MOT17/videos --meta $SCRATCH/vidquery/data/MOT17
    # or from extracted sequence directories
    python scripts/frames_to_video.py --mot-root <dir with train/ test/> --out <videos dir>

Only one copy of each sequence is converted (MOT17 ships every sequence three times, once per
public detector; the frames are identical). ``--meta`` keeps the small per-sequence files
(seqinfo.ini, gt/gt.txt) next to the videos, so the annotations stay available.
"""

from __future__ import annotations

import argparse
import configparser
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from vidquery.video import frames_to_video, write_video

SEQ_RE = re.compile(r"(MOT\d+-\d+)(?:-(\w+))?$")
META_FILES = ("seqinfo.ini", "gt/gt.txt")


def _base_name(seq: str, detector: str) -> str | None:
    """``MOT17-02-FRCNN`` -> ``MOT17-02``; None for the other detector copies."""
    m = SEQ_RE.match(seq)
    if m and m.group(2) and m.group(2) != detector:
        return None
    return m.group(1) if m else seq


def convert_dirs(args: argparse.Namespace, out: Path) -> None:
    seen: set[str] = set()
    for split in args.splits:
        split_dir = Path(args.mot_root) / split
        if not split_dir.is_dir():
            continue
        for seq in sorted(split_dir.iterdir()):
            base = _base_name(seq.name, args.detector)
            if not (seq / "img1").is_dir() or base is None or base in seen:
                continue
            seen.add(base)
            target = out / f"{base}.mp4"
            if target.exists() and not args.overwrite:
                print(f"skip {target} (exists)")
                continue
            n = frames_to_video(seq, target)
            print(f"{seq} -> {target} ({n} frames)")


def convert_zip(args: argparse.Namespace, out: Path) -> None:
    with zipfile.ZipFile(args.zip) as zf:
        # <prefix>/<split>/<seq>/img1/<frame>.jpg, with or without a top-level MOT17/ dir
        frames: dict[tuple[str, str], list[str]] = {}
        for name in zf.namelist():
            parts = PurePosixPath(name).parts
            if len(parts) >= 4 and parts[-2] == "img1" and parts[-1].lower().endswith(".jpg"):
                split, seq = parts[-4], parts[-3]
                if split in args.splits:
                    frames.setdefault((split, seq), []).append(name)
        seen: set[str] = set()
        for (split, seq), names in sorted(frames.items()):
            base = _base_name(seq, args.detector)
            if base is None or base in seen:
                continue
            seen.add(base)
            seq_dir = PurePosixPath(names[0]).parent.parent
            if args.meta:
                for rel in META_FILES:
                    member = str(seq_dir / rel)
                    if member in zf.NameToInfo:
                        dest = Path(args.meta) / split / seq / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(zf.read(member))
            target = out / f"{base}.mp4"
            if target.exists() and not args.overwrite:
                print(f"skip {target} (exists)")
                continue
            fps = _seqinfo_fps(zf, str(seq_dir / "seqinfo.ini"))

            def decode(members: list[str]) -> Iterator[np.ndarray]:
                for member in sorted(members):
                    img = cv2.imdecode(np.frombuffer(zf.read(member), np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError(f"cannot decode {member}")
                    yield img

            n = write_video(decode(names), target, fps)
            print(f"{args.zip}:{seq_dir} -> {target} ({n} frames, {fps:g} fps)")


def _seqinfo_fps(zf: zipfile.ZipFile, member: str, default: float = 30.0) -> float:
    if member not in zf.NameToInfo:
        return default
    ini = configparser.ConfigParser()
    ini.read_string(zf.read(member).decode())
    return ini.getfloat("Sequence", "frameRate", fallback=default)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", help="MOT17.zip (frames are read without extracting)")
    src.add_argument("--mot-root", help="Directory containing train/ and/or test/")
    p.add_argument("--out", required=True)
    p.add_argument("--meta", help="With --zip: copy seqinfo.ini and gt/gt.txt under this dir")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--detector", default="FRCNN", help="Which duplicate to read (DPM/FRCNN/SDP)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.zip:
        convert_zip(args, out)
    else:
        convert_dirs(args, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
