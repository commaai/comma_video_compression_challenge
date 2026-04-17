#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

ARCHIVE_DIR="$1"
OUTPUT_DIR="$2"
VIDEO_NAMES_FILE="$3"

echo "=== pixel_oracle inflation ==="

# Detect device
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  DEVICE="cuda"
elif python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
  DEVICE="mps"
else
  DEVICE="cpu"
fi
echo "Device: $DEVICE"

cd "$ROOT"
python "$HERE/inflate.py" \
  "$ARCHIVE_DIR" \
  "$OUTPUT_DIR" \
  "$VIDEO_NAMES_FILE" \
  --device "$DEVICE" \
  --n-steps 80 \
  --lr 2.0 \
  --batch-size 4 \
  --seg-weight 60.0 \
  --pose-weight 2.0

echo "=== Inflation complete ==="
