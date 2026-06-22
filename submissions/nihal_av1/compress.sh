#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"
IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
JOBS="1"
CRF="${CRF:-33}"
SCALE="${SCALE:-0.45}"

rm -rf "$ARCHIVE_DIR"; mkdir -p "$ARCHIVE_DIR"
export IN_DIR ARCHIVE_DIR CRF SCALE

head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | xargs -P"$JOBS" -I{} bash -lc '
  rel="$1"; [[ -z "$rel" ]] && exit 0
  IN="${IN_DIR}/${rel}"; BASE="${rel%.*}"; OUT="${ARCHIVE_DIR}/${BASE}.mkv"
  echo "-> ${IN} -> ${OUT} (CRF=${CRF} SCALE=${SCALE})"
  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$IN" \
    -vf "scale=trunc(iw*${SCALE}/2)*2:trunc(ih*${SCALE}/2)*2:flags=lanczos" \
    -pix_fmt yuv420p -c:v libsvtav1 -preset 0 -crf ${CRF} \
    -svtav1-params "film-grain=22:keyint=180:scd=0" \
    -r 20 "$OUT"
' _ {}

cd "$ARCHIVE_DIR"; zip -r "${HERE}/archive.zip" . >/dev/null
echo "Compressed (CRF=${CRF} SCALE=${SCALE}) -> ${HERE}/archive.zip"
