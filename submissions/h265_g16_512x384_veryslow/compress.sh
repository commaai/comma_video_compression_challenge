#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PD="$(cd "${HERE}/../.." && pwd)"
IN_DIR="${PD}/videos"; VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"; ARCHIVE_DIR="${HERE}/archive"; JOBS="1"
while [[ $# -gt 0 ]]; do case "$1" in --in-dir|--in_dir) IN_DIR="${2%/}"; shift 2 ;; --jobs) JOBS="$2"; shift 2 ;; --video-names-file|--video_names_file) VIDEO_NAMES_FILE="$2"; shift 2 ;; *) echo "Unknown arg: $1" >&2; exit 2 ;; esac; done
rm -rf "$ARCHIVE_DIR"; mkdir -p "$ARCHIVE_DIR"; export IN_DIR ARCHIVE_DIR
head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | xargs -P"$JOBS" -I{} bash -lc '
  rel="$1"; [[ -z "$rel" ]] && exit 0; IN="${IN_DIR}/${rel}"; BASE="${rel%.*}"; OUT="${ARCHIVE_DIR}/${BASE}.mkv"
  ffmpeg -nostdin -y -hide_banner -loglevel warning -r 20 -fflags +genpts -i "$IN" \
    -vf "scale=512:384:flags=lanczos" \
    -c:v libx265 -preset veryslow -crf 27 -g 16 -bf 1 -x265-params "log-level=warning" -r 20 "$OUT"
' _ {}
cd "$ARCHIVE_DIR"; zip -r "${HERE}/archive.zip" .; echo "Compressed to ${HERE}/archive.zip"
