#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

# Detect Python — prefer UV-managed, fallback to venv, fallback to system
UV_PYTHON="/home/amogh/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11"
if [ -x "$UV_PYTHON" ]; then
  PYTHON="$UV_PYTHON"
  export PYTHONPATH="${ROOT}:${ROOT}/.venv/lib/python3.11/site-packages"
elif [ -x "${ROOT}/.venv/bin/python3" ]; then
  PYTHON="${ROOT}/.venv/bin/python3"
  export PYTHONPATH="${ROOT}"
else
  PYTHON="python"
  export PYTHONPATH="${ROOT}"
fi

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC_DIR="${DATA_DIR}/${BASE}"
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -d "$SRC_DIR" ] && echo "ERROR: ${SRC_DIR} not found" >&2 && exit 1

  printf "Inflating %s via adversarial decode ... " "$line"
  cd "$ROOT"
  "$PYTHON" -m "submissions.${SUB_NAME}.inflate" "$SRC_DIR" "$DST"
done < "$FILE_LIST"
