#!/usr/bin/env bash
set -euo pipefail
ARCHIVE_DIR="$1"
INFLATED_DIR="$2"
VIDEO_NAMES_FILE="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$INFLATED_DIR"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  python "$HERE/inflate.py" "$ARCHIVE_DIR/${BASE}.mkv" "$INFLATED_DIR/${BASE}.raw"
done < "$VIDEO_NAMES_FILE"
