#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: inflate.sh <archive-dir> <output-dir> <file-list>" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

"${CC:-cc}" -O3 -std=c11 -shared -fPIC \
  "$HERE/runtime/entropy/rc64_backend.c" \
  -lm -o "$BUILD_DIR/rc64_backend.so"
export CPR1_RC64_LIBRARY="$BUILD_DIR/rc64_backend.so"

mkdir -p "$OUTPUT_DIR"
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  base="${line%.*}"
  if [[ "$base" != "0" ]]; then
    echo "unsupported public video: $line" >&2
    exit 2
  fi
  python "$HERE/inflate.py" "$DATA_DIR" "$base" "$OUTPUT_DIR/$base.raw"
done < "$FILE_LIST"
