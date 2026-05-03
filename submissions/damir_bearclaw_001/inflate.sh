#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"
SHARPEN_MODE_FILE="${HERE}/sharpen_mode.txt"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.mkv"
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Decoding + resizing %s ... " "$line"
  cd "$ROOT"
  if [ -f "$SHARPEN_MODE_FILE" ]; then
    MY_SUBMISSION_SHARPEN="$(cat "$SHARPEN_MODE_FILE")" python -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"
  else
    python -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"
  fi
done < "$FILE_LIST"

