#!/usr/bin/env bash
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
PR141="$HERE/vendor/external/pr141"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

CCBIN="$(command -v "${CC:-cc}" || true)"
if [ -z "$CCBIN" ]; then
  echo "a C compiler is required (set CC)" >&2
  exit 69
fi

"$CCBIN" -O3 -std=c11 -shared -fPIC "$PR141/runtime/entropy/rc64_backend.c" -o "$BUILD_DIR/rc64_backend.so"
export CPR1_RC64_LIBRARY="$BUILD_DIR/rc64_backend.so"
"$CCBIN" -O3 -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \
  "$PR141/runtime/f26_corrector_native.c" -lm -o "$BUILD_DIR/f26_corrector_native.so"
export F26_CORRECTOR_NATIVE_LIBRARY="$BUILD_DIR/f26_corrector_native.so"

"${PYTHON:-python3}" "$HERE/dz_inflate.py" "$ARCHIVE_DIR" "$OUTPUT_DIR" "$NAMES"
