#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 3 ]]; then
  echo 'usage: inflate.sh <archive-dir> <output-dir> <file-list>' >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT
"${CC:-cc}" -O3 -std=c11 -shared -fPIC \
  "$HERE/runtime/entropy/rc64_backend.c" -o "$BUILD_DIR/rc64_backend.so"
export CPR1_RC64_LIBRARY="$BUILD_DIR/rc64_backend.so"
"${PYTHON:-python}" "$HERE/inflate.py" "$@" --device "${HPAC_DEVICE:-cuda}"
