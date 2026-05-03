#!/usr/bin/env bash
set -euo pipefail

# This submission uses a custom-built upstream SVT-AV1 v4 encoder with
# SegNet-guided per-block QP offset maps (--roi-map-file).
#
# The QP map allocates more bits to class boundary regions and road surfaces
# (identified by running SegNet on the source video during compression),
# and fewer bits to uniform sky/dashboard regions.
#
# Requirements:
# - SVT-AV1 v4.0.1 built from source (https://gitlab.com/AOMediaCodec/SVT-AV1)
# - Python with torch, safetensors, segmentation-models-pytorch
#
# The SvtAv1EncApp binary is NOT included in the archive (not needed for inflate).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"
TMP_DIR="${PD}/tmp/v4_qp_aq2_roi"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"

SVTAV1_ENC="${SVTAV1_ENC:-SvtAv1EncApp}"

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR" "$TMP_DIR"
rm -f "${HERE}/archive.zip"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  IN="${IN_DIR}/${line}"
  BASE="${line%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"
  PRE="${TMP_DIR}/${BASE}.pre.mkv"
  Y4M="${TMP_DIR}/${BASE}.y4m"
  IVF="${TMP_DIR}/${BASE}.ivf"
  QPMAP="${TMP_DIR}/${BASE}_qpmap.txt"

  echo "→ ${IN} → ${OUT}"

  # Step 1: ROI preprocessing
  python "${HERE}/roi_preprocess.py" \
    --input "$IN" --output "$PRE" \
    --outside-luma-denoise 2.5 --outside-chroma-mode medium \
    --feather-radius 48 --outside-blend 0.60

  # Step 2: Downscale to Y4M
  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -i "$PRE" \
    -vf "scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=lanczos" \
    -pix_fmt yuv420p -r 20 "$Y4M"

  # Step 3: Generate SegNet-guided QP map
  python "${HERE}/generate_qpmap.py" --video "$IN" --output "$QPMAP"

  # Step 4: Encode with SVT-AV1 + QP map
  "$SVTAV1_ENC" -i "$Y4M" -b "$IVF" \
    --preset 0 --crf 34 --keyint 180 --scd 0 \
    --enable-qm 1 --qm-min 0 --film-grain 22 \
    --roi-map-file "$QPMAP" --aq-mode 2

  # Step 5: Remux to MKV
  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -i "$IVF" -c copy "$OUT"

  rm -f "$PRE" "$Y4M" "$IVF" "$QPMAP"
done < "$VIDEO_NAMES_FILE"

cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip"
