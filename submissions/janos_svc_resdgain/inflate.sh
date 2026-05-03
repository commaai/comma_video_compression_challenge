#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

PYTHON_BIN="python3"
if [ -x "$ROOT/.venv/bin/python3" ]; then
  PYTHON_BIN="$ROOT/.venv/bin/python3"
fi

"$PYTHON_BIN" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"
