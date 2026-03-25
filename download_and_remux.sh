#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP_URL="https://huggingface.co/datasets/commaai/comma2k19/resolve/main/compression_challenge/test_videos.zip"
ZIP_PATH="${HERE}/test_videos.zip"
EXTRACT_DIR="${HERE}/test_videos"
OUT_DIR="${HERE}/videos"
SEGMENTS_FILE="${HERE}/public_test_segments.txt"
NAMES_FILE="${HERE}/public_test_video_names.txt"
FPS=20

SIZE_BYTES=$(curl -sIL "$ZIP_URL" | grep -i content-length | tail -1 | awk '{print $2}' | tr -d '\r')
SIZE_GB=$(awk "BEGIN {printf \"%.2f\", ${SIZE_BYTES} / 1024 / 1024 / 1024}")
echo "==> Downloading test_videos.zip (${SIZE_GB} GB) ..."
curl -L --progress-bar -o "$ZIP_PATH" "$ZIP_URL"
echo "    Downloaded: $(du -h "$ZIP_PATH" | cut -f1)"
echo ""
echo "==> Extracting to ${EXTRACT_DIR} ..."
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
unzip -q "$ZIP_PATH" -d "$EXTRACT_DIR"

ALL_HEVC=()
while IFS= read -r -d '' f; do
  ALL_HEVC+=("$f")
done < <(find "$EXTRACT_DIR" -name "*.hevc" -print0 | sort -z)

N_FILES=${#ALL_HEVC[@]}
echo "    Found ${N_FILES} .hevc files in archive."
echo "    First 10:"
for i in "${!ALL_HEVC[@]}"; do
  [[ $i -ge 10 ]] && { echo "    ... (${N_FILES} total)"; break; }
  echo "      [$i] ${ALL_HEVC[$i]#"$EXTRACT_DIR/"}"
done
[[ $N_FILES -le 10 ]] && echo "    ... (${N_FILES} total)"
echo ""
echo "==> Remuxing videos listed in $(basename "$SEGMENTS_FILE") ..."
mkdir -p "$OUT_DIR"

idx=0
> "$NAMES_FILE"

while IFS= read -r line; do
  [[ -z "$line" ]] && continue

  SRC="${EXTRACT_DIR}/${line}"
  DST="${OUT_DIR}/${idx}.mkv"

  if [[ ! -f "$SRC" ]]; then
    echo "  [ERROR] Not found in archive: ${line}"
    exit 1
  fi

  echo "  [$idx] ${line}  →  videos/${idx}.mkv"
  ffmpeg -y -loglevel error -f hevc -framerate "$FPS" -r "$FPS" -i "$SRC" -c copy -metadata segment="$line" "$DST"
  echo "${idx}.mkv" >> "$NAMES_FILE"
  idx=$((idx + 1))
done < "$SEGMENTS_FILE"

echo "==> Saved $(wc -l < "$NAMES_FILE") file names to $(basename "$NAMES_FILE")"
echo ""
echo "==> Cleaning up ..."
rm -f "$ZIP_PATH"
echo "    Deleted: $(basename "$ZIP_PATH")"
rm -rf "$EXTRACT_DIR"
echo "    Deleted: $(basename "$EXTRACT_DIR")/"
echo ""
echo " Summary"
echo "  Source archive:   ${N_FILES} .hevc files"
echo "  Remuxed: ${idx} → ${OUT_DIR}/"
echo ""
echo "  Output files:"
for f in "${OUT_DIR}"/*.mkv; do
  [[ -f "$f" ]] || continue
  size=$(du -h "$f" | cut -f1)
  echo "    ${size}  $(basename "$f")"
done
