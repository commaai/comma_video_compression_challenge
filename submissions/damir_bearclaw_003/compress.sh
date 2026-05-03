#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
TMP_DIR="${ROOT}/tmp/damir_bearclaw_003"
PYTHON_BIN="${ROOT}/.venv/bin/python"

IN_DIR="${ROOT}/videos"
VIDEO_NAMES_FILE="${ROOT}/public_test_video_names.txt"
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
mkdir -p "$ARCHIVE_DIR" "$TMP_DIR"
rm -f "${HERE}/archive.zip"

export IN_DIR ARCHIVE_DIR TMP_DIR HERE PYTHON_BIN
echo "Middle-band single-stream: BPS medium middle region, synthetic top/bottom bands on inflate"

head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | xargs -P"$JOBS" -I{} bash -lc '
  rel="$1"
  [[ -z "$rel" ]] && exit 0

  IN="${IN_DIR}/${rel}"
  BASE="${rel%.*}"
  PRE_IN="${TMP_DIR}/${BASE}.mid.mkv"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"

  echo "→ ${IN}  →  ${OUT}"

  rm -f "$PRE_IN"
  "${PYTHON_BIN}" -m "submissions.damir_bearclaw_003.seg_middle_preprocess" \
    --input "$IN" \
    --output "$PRE_IN"

  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$PRE_IN" \
    -vf "scale=trunc(iw*0.45/2)*2:trunc(ih*1.0/2)*2:flags=lanczos" \
    -pix_fmt yuv420p -c:v libsvtav1 -preset 0 -crf 33 \
    -svtav1-params "film-grain=0:keyint=180:scd=0" \
    -r 20 "$OUT"

  rm -f "$PRE_IN"
' _ {}

cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip"
