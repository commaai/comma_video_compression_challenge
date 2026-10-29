#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo 'Usage: inflate.sh <archive_dir> <output_dir> <video_names_file>' >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="$(cd "$1" && pwd)"
mkdir -p "$2"
OUTPUT_DIR="$(cd "$2" && pwd)"
NAMES="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

"${PYTHON:-python3}" "$HERE/inflate.py" "$ARCHIVE_DIR" "$SCRATCH"
bash "$HERE/vendor/inflate.sh" "$SCRATCH" "$OUTPUT_DIR" "$NAMES"
