#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VIDEO_DIR="$ROOT/videos"
ARCHIVE_DIR="$HERE/archive"

rm -rf "$ARCHIVE_DIR" && mkdir -p "$ARCHIVE_DIR"

for vid in "$VIDEO_DIR"/*.mkv; do
  base=$(basename "$vid" .mkv)
  echo "Processing $base..."
  
  # 1. ROI Preprocess (Lossless intermediate)
  python3 "$HERE/roi_preprocess.py" --input "$vid" --output "$ARCHIVE_DIR/${base}_pre.mkv"
  
  # 2. Downscale and Encode with SVT-AV1
  # CRF 40 is a good starting point for balancing SegNet and Rate.
  ffmpeg -i "$ARCHIVE_DIR/${base}_pre.mkv" \
    -vf "scale=512:384" \
    -c:v libsvtav1 -preset 6 -crf 40 \
    -y "$ARCHIVE_DIR/${base}.mkv"
    
  rm "$ARCHIVE_DIR/${base}_pre.mkv"
done

# 3. Create Submission Archive
cd "$HERE"
zip -j archive.zip archive/*.mkv
rm -rf "$ARCHIVE_DIR"
echo "Done! archive.zip created."