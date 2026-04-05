#!/usr/bin/env bash
# SVT-AV1 v2.3.0, scale=45% Lanczos, CRF=33, preset=0, GOP=240, scd=0, film-grain=22
ffmpeg -r 20 -fflags +genpts -i "$1" \
  -vf "scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=lanczos" \
  -pix_fmt yuv420p -c:v libsvtav1 -preset 0 -crf 33 -g 240 \
  -svtav1-params "film-grain=22:film-grain-denoise=1:scd=0" \
  -r 20 "$2"
