#!/usr/bin/env bash
# Must produce a raw video file at `<output_dir>/<base_name>.raw`:
# a flat binary dump of uint8 RGB frames, shape (N, 874, 1164, 3), no header.
#
# This submission is a neural representation (HNeRV): the extracted archive/ contains a trained
# decoder (meta.json + weights.br) + a per-pair selector (selector.bin) that render the frames.
# The SRC video is not used. Self-contained under src/. Runs on CPU (~1 min) or GPU.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="$1"     # extracted archive/ dir (meta.json + weights.br + selector.bin)
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  DST="${OUTPUT_DIR}/${BASE}.raw"
  printf "Rendering %s from neural decoder ... " "$line"
  python "${HERE}/inflate.py" "$DATA_DIR" "$DST"
done < "$FILE_LIST"
