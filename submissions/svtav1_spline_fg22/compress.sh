#!/usr/bin/env bash
# svtav1_spline_fg22 — score 2.16
# SVT-AV1 with 45% spline downscale, CRF 32, preset 0, GOP 180,
# film-grain=22 (denoise+synth)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--in-dir <dir>] [--video-names-file <file>]" >&2
      exit 2 ;;
  esac
done

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  IN="${IN_DIR}/${line}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"

  echo "→ ${IN}  →  ${OUT}"

  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$IN" \
    -vf "scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=spline" \
    -c:v libsvtav1 -crf 32 -preset 0 \
    -svtav1-params "tune=0:film-grain=22:film-grain-denoise=1" \
    -g 180 -an \
    -r 20 "$OUT"
done < "$VIDEO_NAMES_FILE"

cd "$ARCHIVE_DIR"
rm -f "${HERE}/archive.zip"
zip -r "${HERE}/archive.zip" .
