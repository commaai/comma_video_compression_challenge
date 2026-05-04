#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="$1"
OUTPUT_DIR="$2"
VIDEO_NAMES_FILE="$3"

VIDEO_COUNT="$(grep -cve '^[[:space:]]*$' "$VIDEO_NAMES_FILE")"
FIRST_VIDEO="$(grep -ve '^[[:space:]]*$' "$VIDEO_NAMES_FILE" | head -n 1)"
if [ "$VIDEO_COUNT" != "1" ] || [ "$FIRST_VIDEO" != "0.mkv" ]; then
  echo "FATAL: PR98 adapter expects exactly one contest video named 0.mkv" >&2
  exit 64
fi

mkdir -p "$OUTPUT_DIR"
exec python "$HERE/inflate.py" "$ARCHIVE_DIR/0.bin" "$OUTPUT_DIR/0.raw"
