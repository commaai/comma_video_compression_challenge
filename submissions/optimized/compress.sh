#!/usr/bin/env bash
# Score: 2.05 — SVT-AV1 with quantization matrices + film-grain denoise
# Key: 50% lanczos, CRF 34, GOP 240, enable-qm=1:qm-min=0, fg=22
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)     IN_DIR="${2%/}"; shift 2 ;;
    --video-names-file|--video_names_file) VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

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
    -vf "scale=trunc(iw*0.50/2)*2:trunc(ih*0.50/2)*2:flags=lanczos" \
    -c:v libsvtav1 -preset 0 -crf 34 -g 240 \
    -svtav1-params "film-grain=22:film-grain-denoise=1:enable-qm=1:qm-min=0" \
    -pix_fmt yuv420p \
    -r 20 "$OUT"
done < "$VIDEO_NAMES_FILE"

cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip"
