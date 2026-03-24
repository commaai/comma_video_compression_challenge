#!/usr/bin/env bash
# Must produce a raw video file at `<output_dir>/<segment_id>/video.raw`.
# A `.raw` file is a flat binary dump of uint8 RGB frames with shape `(N, H, W, 3)`
# where N is the number of frames, H and W match the original video dimensions, no header.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  SRC="${DATA_DIR}/$(dirname "$line")/video.mkv"
  DST="${OUTPUT_DIR}/$(dirname "$line")/video.raw"
  mkdir -p "$(dirname "$DST")"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Decoding + resizing %s ... " "$line"
  cd "$ROOT"
  python -m submissions.baseline_fast.inflate "$SRC" "$DST"
done < "$FILE_LIST"
