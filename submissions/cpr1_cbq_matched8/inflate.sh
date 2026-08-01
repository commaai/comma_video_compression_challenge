#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  base="${line%.*}"
  python "${HERE}/inflate.py" "$DATA_DIR" "$base" "${OUTPUT_DIR}/${base}.raw"
done < "$FILE_LIST"
