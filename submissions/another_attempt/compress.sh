#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="${HERE}/archive"

echo "Starting another_attempt compression pipeline..."
python3 "${HERE}/compress.py" "$@"

echo "Packaging archive..."
mkdir -p "$ARCHIVE_DIR"
cd "$ARCHIVE_DIR"
zip -0 "${HERE}/archive.zip" *.br

echo "Done. Archive: ${HERE}/archive.zip"
