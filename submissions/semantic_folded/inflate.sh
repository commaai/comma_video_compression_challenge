#!/usr/bin/env bash
# Submission inflater.
#
# Usage: bash inflate.sh <archive_dir> <output_dir> <video_names_file>
#
# Builds the native RC64 backend and the F26 corrector from the pinned external/pr141 sources
# (same flags as that submission's inflate.sh: -ffp-contract=off is load-bearing), then runs
# inflate.py -> canonical/inflate_canonical.py (integrated decoder + HPAC rowpack and RID1/RID2 renderer shims).
# Produces <output_dir>/<video_stem>.raw: headerless uint8 RGB, shape (1200, 874, 1164, 3).
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <archive_dir> <output_dir> <video_names_file>" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="$(cd "$1" && pwd)"
mkdir -p "$2"
OUTPUT_DIR="$(cd "$2" && pwd)"
VIDEO_NAMES_FILE="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
PR141="$HERE/external/pr141"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

CC_REQUESTED="${CC:-cc}"
CCBIN="$(command -v "$CC_REQUESTED" || true)"
if [ -z "$CCBIN" ]; then
  echo "a C compiler is required ('$CC_REQUESTED' unavailable; set CC)" >&2
  exit 69
fi

if [ -n "${CPR1_RC64_LIBRARY:-}" ]; then
  [ -f "$CPR1_RC64_LIBRARY" ] || { echo "missing CPR1_RC64_LIBRARY" >&2; exit 69; }
else
  "$CCBIN" -O3 -std=c11 -shared -fPIC "$PR141/runtime/entropy/rc64_backend.c" -o "$BUILD_DIR/rc64_backend.so"
  export CPR1_RC64_LIBRARY="$BUILD_DIR/rc64_backend.so"
fi

# The corrector build may fail without costing correctness: the decoder then falls back to the (much slower) Python corrector.
if [ -n "${F26_CORRECTOR_NATIVE_LIBRARY:-}" ]; then
  [ -f "$F26_CORRECTOR_NATIVE_LIBRARY" ] || { echo "missing F26 corrector library" >&2; exit 69; }
elif "$CCBIN" -O3 -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \
       "$PR141/runtime/f26_corrector_native.c" -lm -o "$BUILD_DIR/f26_corrector_native.so" 2>/dev/null; then
  export F26_CORRECTOR_NATIVE_LIBRARY="$BUILD_DIR/f26_corrector_native.so"
else
  echo "f26 corrector native build unavailable; using the python corrector" >&2
fi

cd "$HERE"  # a few runtime modules resolve external/pr141 relative to the submission root
"${PYTHON:-python3}" "$HERE/inflate.py" "$ARCHIVE_DIR" "$OUTPUT_DIR" "$VIDEO_NAMES_FILE"
