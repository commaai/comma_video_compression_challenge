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

INNER_PAYLOAD="$DATA_DIR/payload.tar.xz"
if [ -f "$INNER_PAYLOAD" ] && [ ! -f "$DATA_DIR/segmap_inference.pt" ]; then
  "$PYTHON_BIN" - <<'PY' "$INNER_PAYLOAD" "$DATA_DIR"
import sys
import tarfile
from pathlib import Path

payload_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
with tarfile.open(payload_path, mode='r:xz') as tf:
    tf.extractall(output_dir)
PY
fi

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  DST="${OUTPUT_DIR}/${BASE}.raw"
  "$PYTHON_BIN" "$HERE/inflate.py" "$DATA_DIR" "$line" "$DST"
done < "$FILE_LIST"
