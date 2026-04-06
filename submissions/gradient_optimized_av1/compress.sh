#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

# Compress video with SVT-AV1 (NO label generation - labels computed on-the-fly from original)
while IFS= read -r line; do
  [ -z "$line" ] && continue
  IN="${IN_DIR}/${line}"
  BASE="${line%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"

  echo "Compressing ${IN} -> ${OUT}"
  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$IN" \
    -vf "scale=trunc(iw*0.42/2)*2:trunc(ih*0.42/2)*2:flags=lanczos" \
    -pix_fmt yuv420p -c:v libsvtav1 -preset 0 -crf 31 \
    -svtav1-params "film-grain=22:keyint=180:scd=0:enable-qm=1:qm-min=0" \
    -r 20 "$OUT"

done < "$VIDEO_NAMES_FILE"

# Zip archive (video only, no labels)
cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
echo "Archive size: $(wc -c < "${HERE}/archive.zip") bytes"
echo "Compressed to ${HERE}/archive.zip"
