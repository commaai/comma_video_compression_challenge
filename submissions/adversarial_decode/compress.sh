#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

IN_DIR="${ROOT}/videos"
VIDEO_NAMES_FILE="${ROOT}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--in-dir <dir>] [--video-names-file <file>]" >&2
      exit 2 ;;
  esac
done

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

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  VIDEO_PATH="${IN_DIR}/${line}"
  OUTPUT_SUBDIR="${ARCHIVE_DIR}/${BASE}"

  echo "→ Encoding ${line} → ${OUTPUT_SUBDIR}"
  cd "$ROOT"
  "$PYTHON" -m submissions.adversarial_decode.encode "$VIDEO_PATH" "$OUTPUT_SUBDIR"
done < "$VIDEO_NAMES_FILE"

cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip"
ls -lh "${HERE}/archive.zip"
