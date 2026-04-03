#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue
  IN="${IN_DIR}/${rel}"
  BASE="${rel%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"

  echo "→ ${IN}  →  ${OUT}"

  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$IN" \
    -vf "scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=lanczos" \
    -c:v libsvtav1 -preset 0 -crf 33 \
    -g 180 \
    -svtav1-params "film-grain=22:film-grain-denoise=1" \
    -r 20 "$OUT"
done < "$VIDEO_NAMES_FILE"

cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip"
