#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# Usage: compress.sh [video_dir] [output_archive]
VIDEO_DIR="${1:-$ROOT/videos}"
DEVICE="${2:-cpu}"

echo "=== pixel_oracle compression ==="
echo "Video dir: $VIDEO_DIR"
echo "Device: $DEVICE"

cd "$ROOT"
python "$HERE/compress.py" \
  --video "$VIDEO_DIR/0.mkv" \
  --output-dir "$HERE/archive_contents" \
  --scale 0.35 \
  --crf 50 \
  --device "$DEVICE"

echo "=== Compression complete ==="
