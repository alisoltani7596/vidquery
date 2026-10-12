#!/usr/bin/env python
"""Convert MOTChallenge image sequences (seq/img1/*.jpg + seqinfo.ini) to mp4 files.

    python scripts/frames_to_video.py --mot-root $SCRATCH/vidquery/data/MOT17 \
        --out $SCRATCH/vidquery/data/MOT17/videos

Only one copy of each sequence is converted (MOT17 ships every sequence three times, once per
public detector; the frames are identical).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from vidquery.video import frames_to_video


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mot-root", required=True, help="Directory containing train/ and/or test/")
    p.add_argument("--out", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--detector", default="FRCNN", help="Which duplicate to read (DPM/FRCNN/SDP)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for split in args.splits:
        split_dir = Path(args.mot_root) / split
        if not split_dir.is_dir():
            continue
        for seq in sorted(split_dir.iterdir()):
            if not (seq / "img1").is_dir():
                continue
            m = re.match(r"(MOT\d+-\d+)(?:-(\w+))?$", seq.name)
            base = m.group(1) if m else seq.name
            if m and m.group(2) and m.group(2) != args.detector:
                continue
            if base in seen:
                continue
            seen.add(base)
            target = out / f"{base}.mp4"
            if target.exists() and not args.overwrite:
                print(f"skip {target} (exists)")
                continue
            n = frames_to_video(seq, target)
            print(f"{seq} -> {target} ({n} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
