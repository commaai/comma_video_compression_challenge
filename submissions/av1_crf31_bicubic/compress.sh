#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"
TMP_DIR="${PD}/tmp/av1_crf31_bicubic"
IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
rm -rf "$ARCHIVE_DIR"; mkdir -p "$ARCHIVE_DIR" "$TMP_DIR"
export IN_DIR ARCHIVE_DIR PD
head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | xargs -P1 -I{} bash -lc '
  rel="$1"; [[ -z "$rel" ]] && exit 0
  IN="${IN_DIR}/${rel}"; BASE="${rel%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"; PRE_IN="'"${TMP_DIR}"'/${BASE}.pre.mkv"
  rm -f "$PRE_IN"
  cd "'"${PD}"'"
  .venv/bin/python -m submissions.av1_crf31_bicubic.preprocess \
    --input "$IN" --output "$PRE_IN" \
    --outside-luma-denoise 2.5 --outside-chroma-mode medium \
    --feather-radius 24 --outside-blend 0.50
  FFMPEG="'"${HERE}"'/ffmpeg-new"
  [ ! -x "$FFMPEG" ] && FFMPEG="ffmpeg"
  [ -d "'"${HERE}"'/lib" ] && export LD_LIBRARY_PATH="'"${HERE}"'/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  "$FFMPEG" -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$PRE_IN" \
    -vf "scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=lanczos" \
    -pix_fmt yuv420p -c:v libsvtav1 -preset 0 -crf 31 \
    -svtav1-params "film-grain=22:keyint=180:scd=0" \
    -r 20 "$OUT"
  rm -f "$PRE_IN"
' _ {}
cd "$ARCHIVE_DIR"; zip -r "${HERE}/archive.zip" .
