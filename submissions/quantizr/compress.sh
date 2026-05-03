#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="${HERE}/archive"

echo "Starting full end-to-end compression pipeline..."

# pass along any arguments (e.g., --crf 50)
python3 "${HERE}/compress.py" "$@"

echo "Pipeline complete. Packaging artifacts..."

mkdir -p "$ARCHIVE_DIR"
cd "$ARCHIVE_DIR"

zip -0 "${HERE}/archive.zip" *.br

echo "Done! Final payload saved to: ${HERE}/archive.zip"