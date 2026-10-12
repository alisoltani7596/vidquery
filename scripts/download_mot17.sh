#!/usr/bin/env bash
# Download MOT17 (frames + labels) into $SCRATCH and convert each sequence to an mp4.
# Run on a LOGIN node (compute nodes have no internet):
#   bash scripts/download_mot17.sh                 # -> $SCRATCH/vidquery/data/MOT17
#   bash scripts/download_mot17.sh /some/other/dir
#   NO_CONVERT=1 bash scripts/download_mot17.sh    # keep image sequences only
#
# MOT17.zip is ~5.5 GB. Downloads resume if interrupted. The MOTChallenge data is licensed
# CC BY-NC-SA 3.0. Cite the MOT16/17 paper when you publish results.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-${SCRATCH:?set SCRATCH or pass a destination}/vidquery/data}/MOT17"
URL="${MOT17_URL:-https://motchallenge.net/data/MOT17.zip}"
ZIP="$DEST/MOT17.zip"

mkdir -p "$DEST"
echo "Downloading $URL -> $ZIP"
curl -L --fail --retry 5 --retry-delay 10 -C - -o "$ZIP" "$URL"

if [[ ! -d "$DEST/train" ]]; then
    echo "Unzipping (this takes a few minutes on Lustre)"
    tmp="$DEST/.unzip"
    rm -rf "$tmp" && mkdir -p "$tmp"
    unzip -q "$ZIP" -d "$tmp"
    # The archive has a top-level MOT17/ directory; flatten it.
    src="$tmp"
    [[ -d "$tmp/MOT17" ]] && src="$tmp/MOT17"
    mv "$src"/* "$DEST"/
    rm -rf "$tmp"
fi
echo "Sequences:"
find "$DEST" -mindepth 2 -maxdepth 2 -type d -name 'MOT17-*-FRCNN' | sort

if [[ "${NO_CONVERT:-0}" == "1" ]]; then
    echo "Skipping mp4 conversion (ingest can read $DEST/train/<seq> directly)."
    exit 0
fi

# Convert one copy of each sequence (the DPM/FRCNN/SDP copies have identical frames).
CONVERT=(python "$REPO/scripts/frames_to_video.py" --mot-root "$DEST" --out "$DEST/videos")
if [[ -f "$REPO/slurm/env.sh" && -n "${SCRATCH:-}" ]]; then
    export VQ_REPO="$REPO"
    # shellcheck source=/dev/null
    source "$REPO/slurm/env.sh"
    if [[ -f "$VQ_TRT_SIF" && -d "$VQ_VENV" ]]; then
        vq_exec "${CONVERT[@]}"
        echo "Videos in $DEST/videos"
        exit 0
    fi
fi
if python -c "import vidquery" 2>/dev/null; then
    "${CONVERT[@]}"
    echo "Videos in $DEST/videos"
else
    echo "vidquery is not importable here; convert later with:" >&2
    echo "  ${CONVERT[*]}" >&2
fi
