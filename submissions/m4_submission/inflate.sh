#!/bin/bash
# Get the exact directory where this script lives
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create the inflated directory
mkdir -p "$DIR/inflated"

# Change pixel format from yuv420p to rgb24 so PyTorch calculates the correct frame count
ffmpeg -y -i "$DIR/archive/compressed.mp4" -f rawvideo -pix_fmt rgb24 "$DIR/inflated/0.raw"
