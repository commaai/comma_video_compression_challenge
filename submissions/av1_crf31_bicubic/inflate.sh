#!/usr/bin/env bash
# Emits <output_dir>/<base>.raw: flat uint8 RGB dump, shape (N, H, W, 3), no header.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

DATA_DIR="$1"    # unzipped archive/
OUTPUT_DIR="$2"  # where .raw files go
FILE_LIST="$3"   # text file, one video name per line

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.mkv"
  DST="${OUTPUT_DIR}/${BASE}.raw"
  [ ! -f "$SRC" ] && echo "ERROR: missing ${SRC}" >&2 && exit 1
  cd "$ROOT"
  python -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"
done < "$FILE_LIST"
