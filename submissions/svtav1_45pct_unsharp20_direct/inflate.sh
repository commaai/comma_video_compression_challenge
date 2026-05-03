#!/usr/bin/env bash
# Must produce a raw video file at `<output_dir>/<base_name>.raw`.
# A `.raw` file is a flat binary dump of uint8 RGB frames with shape `(N, H, W, 3)`
# where N is the number of frames, H and W match the original video dimensions, no header.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  if [ -f "${DATA_DIR}/${BASE}.ivf" ]; then
    SRC="${DATA_DIR}/${BASE}.ivf"
  else
    SRC="${DATA_DIR}/${BASE}.mkv"
  fi
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Decoding + resizing %s ... " "$line"
  cd "$ROOT"
  python -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"
done < "$FILE_LIST"
