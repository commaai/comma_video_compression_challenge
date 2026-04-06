#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"
INFLATE_CONFIG_FILE="${HERE}/inflate_config.env"

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
  if [ -f "$INFLATE_CONFIG_FILE" ]; then
    set -a
    . "$INFLATE_CONFIG_FILE"
    set +a
    python -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"
  else
    python -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"
  fi
done < "$FILE_LIST"

