#!/bin/bash
INPUT_VIDEO="../../videos/0.mkv"
ARCHIVE_DIR="archive"
OUTPUT_VIDEO="$ARCHIVE_DIR/compressed.mp4"

rm -rf $ARCHIVE_DIR archive.zip
mkdir -p $ARCHIVE_DIR

echo "Compressing video using AV1..."
ffmpeg -i $INPUT_VIDEO \
    -c:v libsvtav1 \
    -preset 3 \
    -crf 45 \
    -svtav1-params tune=2 \
    -y $OUTPUT_VIDEO

echo "Zipping archive (without nested folders)..."
# The -j flag strips the directory path when zipping
zip -j archive.zip $OUTPUT_VIDEO

echo "Compression complete. Payload size:"
ls -lh archive.zip
