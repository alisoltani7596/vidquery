#!/usr/bin/env bash
# Download MOT17 into $SCRATCH and convert each sequence to an mp4. Frames are decoded straight
# from MOT17.zip, never extracted: loose JPEGs (~11K per copy) would eat the /scratch file
# quota. Only the mp4s and the small seqinfo.ini / gt.txt files land on disk.
# slurm/setup.sbatch runs this in a job, because converting is heavy:
#   bash scripts/download_mot17.sh                 # -> $SCRATCH/vidquery/data/MOT17
#   bash scripts/download_mot17.sh /some/other/dir
#
# MOT17.zip is ~5.5 GB and is kept, so videos can be rebuilt. Downloads resume if interrupted.
# The MOTChallenge data is licensed CC BY-NC-SA 3.0. Cite the MOT16/17 paper when you publish.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-${SCRATCH:?set SCRATCH or pass a destination}/vidquery/data}/MOT17"
URL="${MOT17_URL:-https://motchallenge.net/data/MOT17.zip}"
ZIP="$DEST/MOT17.zip"

mkdir -p "$DEST"
echo "Downloading $URL -> $ZIP"
curl -L --fail --retry 5 --retry-delay 10 -C - -o "$ZIP" "$URL"

# One copy of each sequence (the DPM/FRCNN/SDP copies have identical frames).
CONVERT=(python "$REPO/scripts/frames_to_video.py" --zip "$ZIP" --out "$DEST/videos"
    --meta "$DEST")
if [[ -f "$REPO/slurm/env.sh" && -n "${SCRATCH:-}" ]]; then
    export VQ_REPO="$REPO"
    # shellcheck source=/dev/null
    source "$REPO/slurm/env.sh"
    if [[ -f "$VQ_TRT_SIF" && -f "$VQ_VENV_IMG" ]]; then
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
