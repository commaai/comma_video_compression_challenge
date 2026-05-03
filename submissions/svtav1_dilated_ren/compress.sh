#!/usr/bin/env bash
# Encode with SVT-AV1 v2.3.0 and bundle with ren.bz2 into archive.zip.
# ren.bz2 must exist at $HERE/ren.bz2 (trained via this notebook).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
JOBS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)   IN_DIR="${2%/}"; shift 2 ;;
    --jobs)              JOBS="$2"; shift 2 ;;
    --video-names-file|--video_names_file) VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ ! -f "${HERE}/ren.bz2" ]; then
  echo "ERROR: ${HERE}/ren.bz2 not found." >&2
  exit 1
fi

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

export IN_DIR ARCHIVE_DIR

head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | xargs -P"$JOBS" -I{} bash -lc '
  rel="$1"
  [[ -z "$rel" ]] && exit 0
  IN="${IN_DIR}/${rel}"
  BASE="${rel%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"
  echo "→ ${IN}  →  ${OUT}"
  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$IN" \
    -vf "scale=544:408:flags=lanczos" \
    -pix_fmt yuv420p -c:v libsvtav1 -preset 0 -crf 36 \
    -svtav1-params "film-grain=22:keyint=180:scd=0" \
    -r 20 "$OUT"
' _ {}

cp "${HERE}/ren.bz2" "${ARCHIVE_DIR}/ren.bz2"
cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip"
