#!/usr/bin/env bash
# SVT-AV1 v2.3.0 | scale=45% Lanczos | CRF=33 | preset=0 | GOP=240 | scd=0 | film-grain=22
# Inflate: bicubic upscale + binomial 9x9 unsharp mask (amount=0.85)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IN_DIR="${HERE}/../../videos"
OUT_DIR="${HERE}/archive"
mkdir -p "$OUT_DIR"
ffmpeg -r 20 -fflags +genpts -i "$IN_DIR/0.mkv" \
  -vf "scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=lanczos" \
  -pix_fmt yuv420p -c:v libsvtav1 -preset 0 -crf 33 -g 240 \
  -svtav1-params "film-grain=22:film-grain-denoise=1:scd=0" \
  -r 20 "$OUT_DIR/0.mkv"
cd "$OUT_DIR" && zip -r "${HERE}/archive.zip" .
