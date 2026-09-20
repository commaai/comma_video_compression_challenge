#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$#" -gt 0 ]]; then
  exec "${PYTHON:-python}" "$HERE/compress.py" "$@"
fi
# Download the attributed reference, then verify its fixed SHA-256 in Python.
REFERENCE_URL="https://github.com/codexblack/comma_video_compression_challenge/releases/download/semantic-pose-HPAC_CPR1_polished-f26/archive.zip"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
curl --fail --location --retry 3 "$REFERENCE_URL" -o "$WORK_DIR/reference.zip"
"${PYTHON:-python}" "$HERE/compress.py" --source "$WORK_DIR/reference.zip"
