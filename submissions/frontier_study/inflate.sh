#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Converts the extracted archive/ into raw frames <output_dir>/<base>.raw
# (flat uint8 RGB, shape (N, 874, 1164, 3), no header).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

if [ -n "${FRONTIER_PYTHON_BIN:-}" ]; then PY="$FRONTIER_PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then PY=python
else PY=python3; fi

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/payload.bin"
  [ ! -f "$SRC" ] && SRC="${DATA_DIR}/${BASE}.bin"
  DST="${OUTPUT_DIR}/${BASE}.raw"
  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1
  printf "Inflating %s ... " "$line"
  "$PY" "$HERE/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
