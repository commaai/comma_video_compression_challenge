#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE"
while [ "$ROOT" != "/" ] && [ ! -f "${ROOT}/evaluate.py" ]; do
  ROOT="$(dirname "$ROOT")"
done

if [ ! -f "${ROOT}/evaluate.py" ]; then
  ROOT="$(pwd)"
fi

if [ -x "${ROOT}/.venv/bin/python" ]; then
  PYTHON_BIN="${ROOT}/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "ERROR: no Python interpreter found" >&2
  exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"
