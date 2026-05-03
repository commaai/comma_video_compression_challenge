#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"
TMP_DIR="${PD}/tmp/roi_gop300_c34"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir) IN_DIR="${2%/}"; shift 2 ;;
    --video-names-file|--video_names_file) VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR" "$TMP_DIR"
rm -f "${HERE}/archive.zip"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  IN="${IN_DIR}/${line}"
  BASE="${line%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"
  PRE="${TMP_DIR}/${BASE}.pre.mkv"

  echo "→ ${IN} → ${OUT}"

  # ROI preprocessing: denoise outside driving corridor
  python "${HERE}/roi_preprocess.py" \
    --input "$IN" \
    --output "$PRE" \
    --outside-luma-denoise 2.5 \
    --outside-chroma-mode medium \
    --feather-radius 48 \
    --outside-blend 0.60

  # Encode with SVT-AV1: CRF 34, scale 0.45, GOP 300, film-grain 22
  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$PRE" \
    -vf "scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=lanczos" \
    -pix_fmt yuv420p -c:v libsvtav1 -preset 0 -crf 34 \
    -svtav1-params "film-grain=22:keyint=300:scd=0:enable-qm=1:qm-min=0" \
    -r 20 "$OUT"

  rm -f "$PRE"
done < "$VIDEO_NAMES_FILE"

cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip"
