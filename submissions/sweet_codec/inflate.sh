#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

cd "$ROOT"
source .venv/bin/activate

while IFS= read -r line; do
  [ -z "$line" ] && continue
  SRC="${DATA_DIR}/$(dirname "$line")/video.mkv"
  DST="${OUTPUT_DIR}/$(dirname "$line")/video.raw"
  ARCHIVE_DIR="${DATA_DIR}/$(dirname "$line")"
  mkdir -p "$(dirname "$DST")"
  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1
  printf "Inflating %s … " "$line"
  python -m submissions.sweet_codec.inflate "$SRC" "$DST" "$ARCHIVE_DIR"
done < "$FILE_LIST"
