#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  SRC="${DATA_DIR}/$(dirname "$line")/video.mkv"
  DST="${OUTPUT_DIR}/$(dirname "$line")/video.raw"
  mkdir -p "$(dirname "$DST")"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Decoding + resizing %s ... " "$line"
  cd "$ROOT"
  python -m submissions.optimized.inflate "$SRC" "$DST"
done < "$FILE_LIST"
