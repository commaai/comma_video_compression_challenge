#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"
TMP_DIR="${PD}/tmp/neural_inflate"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
JOBS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --jobs)
      JOBS="$2"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--in-dir <dir>] [--jobs <n>] [--video-names-file <file>]" >&2
      exit 2 ;;
  esac
done

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"
mkdir -p "$TMP_DIR"

export IN_DIR ARCHIVE_DIR PD

head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | xargs -P"$JOBS" -I{} bash -lc '
  rel="$1"
  [[ -z "$rel" ]] && exit 0

  IN="${IN_DIR}/${rel}"
  BASE="${rel%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"
  PRE_IN="'"${TMP_DIR}"'/${BASE}.pre.mkv"

  echo "→ ${IN}  →  ${OUT}"

  # Step 1: ROI preprocess — denoise outside driving corridor
  rm -f "$PRE_IN"
  python "'"${HERE}"'/preprocess.py" \
    --input "$IN" \
    --output "$PRE_IN" \
    --outside-luma-denoise 2.5 \
    --outside-chroma-mode medium \
    --feather-radius 24 \
    --outside-blend 0.50

  # Step 2: Downscale + AV1 encode
  FFMPEG="${PD}/ffmpeg-new"
  [ ! -x "$FFMPEG" ] && FFMPEG="ffmpeg"
  export LD_LIBRARY_PATH="${PD}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  "$FFMPEG" -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$PRE_IN" \
    -vf "scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=lanczos" \
    -pix_fmt yuv420p -c:v libsvtav1 -preset 0 -crf 33 \
    -svtav1-params "film-grain=22:keyint=180:scd=0" \
    -r 20 "$OUT"

  rm -f "$PRE_IN"
' _ {}

# zip archive
cd "$ARCHIVE_DIR"
if command -v zip &>/dev/null; then
  zip -r "${HERE}/archive.zip" .
else
  python3 -c "
import zipfile, os
with zipfile.ZipFile('${HERE}/archive.zip', 'w', zipfile.ZIP_STORED) as zf:
    for f in os.listdir('.'):
        zf.write(f)
"
fi
echo "Compressed to ${HERE}/archive.zip"
