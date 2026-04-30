#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"
PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" && -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT}/.venv/bin/python"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue
  base="${rel%.*}"
  src="${DATA_DIR}/${base}"
  dst="${OUTPUT_DIR}/${base}.raw"

  [[ ! -d "$src" ]] && echo "ERROR: ${src} not found" >&2 && exit 1

  cd "$ROOT"
  "$PYTHON_BIN" -m "submissions.${SUB_NAME}.inflate" "$src" "$dst"
done < "$FILE_LIST"
