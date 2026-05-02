#!/usr/bin/env bash
# compress.sh — produce archive.zip from the original videos.
#
# Run from the root of the comma_video_compression_challenge repo, e.g.:
#   bash submissions/my_submission/compress.sh
#
# Edits below: point CHECKPOINT to your trained weights.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# --- configure these ---------------------------------------------------------
CHECKPOINT="${CHECKPOINT:-$HERE/checkpoints/compressor_final.pt}"
INPUT_DIR="${INPUT_DIR:-$REPO_ROOT/videos}"
VIDEO_NAMES="${VIDEO_NAMES:-$REPO_ROOT/public_test_video_names.txt}"
DEVICE="${DEVICE:-cuda}"
BATCH="${BATCH:-4}"
# -----------------------------------------------------------------------------

WORK_DIR="$HERE/_archive_build"
ARCHIVE_DIR="$WORK_DIR/archive"
ZIP_PATH="$HERE/archive.zip"

rm -rf "$WORK_DIR"
mkdir -p "$ARCHIVE_DIR"

echo "==> Compressing videos with $CHECKPOINT"
python "$HERE/compress.py" \
    --input_dir "$INPUT_DIR" \
    --video_names "$VIDEO_NAMES" \
    --checkpoint "$CHECKPOINT" \
    --out_dir "$ARCHIVE_DIR" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --save_dtype fp16

# Also include the inference-side python files in the archive so inflate.sh
# can run standalone from the extracted archive (no external repo needed).
cp "$HERE/compressor.py" "$ARCHIVE_DIR/"
cp "$HERE/decompress.py" "$ARCHIVE_DIR/"

echo "==> Zipping archive"
( cd "$WORK_DIR" && zip -r -9 -q "$ZIP_PATH" archive )

SIZE=$(stat -c%s "$ZIP_PATH" 2>/dev/null || stat -f%z "$ZIP_PATH")
echo "Done. archive.zip = $((SIZE / 1024)) KB at $ZIP_PATH"