#!/usr/bin/env bash
# tcm_ctxmix inflater.  Usage: bash inflate.sh <archive_dir> <output_dir> <video_names_file>
# Builds the TCM token decoder (plain C, integer-only coding path) from tcm/tcm.c, then runs inflate.py,
# which decodes the tokens and renders the frames with PR #153's renderer and carrier.  Writes <output_dir>/0.raw.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <archive_dir> <output_dir> <video_names_file>" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="$(cd "$1" && pwd)"
mkdir -p "$2"
OUTPUT_DIR="$(cd "$2" && pwd)"
NAMES="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

CC_REQUESTED="${CC:-cc}"
CCBIN="$(command -v "$CC_REQUESTED" || true)"
if [ -z "$CCBIN" ]; then
  echo "ERROR: a C compiler is required to build the TCM decoder ('$CC_REQUESTED' not found; set CC)" >&2
  exit 69
fi
# Portable flags on purpose (no -march=native): the coding path is integer-only, identical on x86_64 and arm64.
"$CCBIN" -O2 -std=c11 -ffp-contract=off -o "$BUILD_DIR/tcm" "$HERE/tcm/tcm.c" \
  || { echo "ERROR: TCM decoder build failed" >&2; exit 69; }

TCM_BIN="$BUILD_DIR/tcm" "${PYTHON:-python}" "$HERE/inflate.py" "$ARCHIVE_DIR" "$OUTPUT_DIR" "$NAMES"
