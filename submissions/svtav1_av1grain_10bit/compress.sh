#!/usr/bin/env bash
# Compression script for svtav1_av1grain_10bit submission.
#
# Requires SVT-AV1 v2.3.0 built from source:
#   git clone --depth 1 --branch v2.3.0 https://gitlab.com/AOMediaCodec/SVT-AV1.git /tmp/svt-av1-build
#   mkdir -p /tmp/svt-av1-build/Build
#   cmake -S /tmp/svt-av1-build -B /tmp/svt-av1-build/Build -DCMAKE_BUILD_TYPE=Release
#   make -C /tmp/svt-av1-build/Build -j$(nproc) SvtAv1EncApp

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

SVT_BIN="/tmp/svt-av1-build/Bin/Release/SvtAv1EncApp"
SVT_LIB="/tmp/svt-av1-build/Bin/Release"

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

if [ ! -x "$SVT_BIN" ]; then
  echo "ERROR: SVT-AV1 v2.3.0 not found at $SVT_BIN" >&2
  echo "Build it with the commands in the header of this script." >&2
  exit 1
fi

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

export IN_DIR ARCHIVE_DIR SVT_BIN SVT_LIB

head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | xargs -P"$JOBS" -I{} bash -lc '
  rel="$1"
  [[ -z "$rel" ]] && exit 0

  IN="${IN_DIR}/${rel}"
  BASE="${rel%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"
  TMPDIR=$(mktemp -d)

  echo "-> ${IN} -> ${OUT}"

  SW=$(ffprobe -v quiet -select_streams v -show_entries stream=width -of csv=p=0 "$IN")
  SH=$(ffprobe -v quiet -select_streams v -show_entries stream=height -of csv=p=0 "$IN")
  TW=$(python3 -c "print(int(int(\"$SW\") * 0.50 / 2) * 2)")
  TH=$(python3 -c "print(int(int(\"$SH\") * 0.50 / 2) * 2)")

  # Downscale to 50% with Lanczos, output 10-bit YUV420
  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$IN" \
    -vf "scale=${TW}:${TH}:flags=lanczos" \
    -pix_fmt yuv420p10le -f rawvideo "pipe:1" | \
  LD_LIBRARY_PATH="${SVT_LIB}:${LD_LIBRARY_PATH:-}" "$SVT_BIN" \
    -i stdin \
    -w "$TW" -h "$TH" \
    --fps 20 \
    --preset 1 \
    --crf 34 \
    --keyint -1 \
    --irefresh-type 2 \
    --input-depth 10 \
    --film-grain 30 \
    --film-grain-denoise 0 \
    --lp 4 \
    -b "${TMPDIR}/out.ivf"

  # Mux IVF into MKV
  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -i "${TMPDIR}/out.ivf" \
    -c:v copy -r 20 "$OUT"

  rm -rf "$TMPDIR"
' _ {}

# zip archive
cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip"
