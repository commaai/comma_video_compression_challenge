#!/usr/bin/env bash
# inflate.sh — convert the extracted archive/ into reconstructed video frames.
#
# The comma evaluator extracts archive.zip into a directory and then runs this.
# The first argument (or $ARCHIVE_DIR) is the path to the extracted archive,
# and the second argument (or $OUT_DIR) is where to write reconstructed videos.
#
# We write lossless HEVC files named <video_basename>.hevc — the same basenames
# as the original test videos — so the evaluator can match them up.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCHIVE_DIR="${1:-${ARCHIVE_DIR:-$HERE/archive}}"
OUT_DIR="${2:-${OUT_DIR:-$HERE/inflated}}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_DIR"

# decompress.py is shipped INSIDE the archive (compress.sh copies it there)
# so this script doesn't depend on anything outside the extracted archive.
SCRIPT="$ARCHIVE_DIR/decompress.py"
if [[ ! -f "$SCRIPT" ]]; then
    # fall back to the script next to this file (dev mode)
    SCRIPT="$HERE/decompress.py"
fi

echo "==> Inflating $ARCHIVE_DIR -> $OUT_DIR  (device=$DEVICE)"
python "$SCRIPT" \
    --archive_dir "$ARCHIVE_DIR" \
    --out_dir "$OUT_DIR" \
    --device "$DEVICE" \
    --ext raw

echo "Done."