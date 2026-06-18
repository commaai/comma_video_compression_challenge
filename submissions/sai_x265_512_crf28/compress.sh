#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
JOBS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir) IN_DIR="${2%/}"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --video-names-file|--video_names_file) VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

rm -rf "$ARCHIVE_DIR"
rm -f "${HERE}/archive.zip"
mkdir -p "$ARCHIVE_DIR"

export IN_DIR ARCHIVE_DIR

cat "$VIDEO_NAMES_FILE" | xargs -P"$JOBS" -I{} bash -lc '
  rel="$1"
  [[ -z "$rel" ]] && exit 0

  input_file="${IN_DIR}/${rel}"
  base="${rel%.*}"
  output_file="${ARCHIVE_DIR}/${base}.mkv"

  echo "Encoding ${input_file} -> ${output_file}"

  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$input_file" \
    -vf "hqdn3d=1.5:1.5:6:6,scale=512:384:flags=lanczos" \
    -c:v libx265 -preset slow -crf 28 \
    -g 60 \
    -x265-params "keyint=60:min-keyint=30:bframes=4:log-level=warning" \
    -r 20 "$output_file"
' _ {}

cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip"