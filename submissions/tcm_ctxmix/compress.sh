#!/usr/bin/env bash
# usage (repo root): uv run --group cpu bash submissions/tcm_ctxmix/compress.sh [--reference153 <archive.zip>] [--output-dir <dir>] [--device cpu|cuda]
#   input:  PR #153's public archive.zip (downloaded from PR153_URL unless --reference153 is given; sha256 must be PR153_SHA256)
#   output: <output-dir>/archive.zip (default: this folder)
#   needs:  cc, curl (for the download), python with numpy, torch, brotli
set -euo pipefail

PR153_URL="https://github.com/hypery11/comma_video_compression_challenge/releases/download/hypery11-carrier-joint-20261104/archive.zip"
PR153_SHA256="52a98a5f7840648c5b08fe958ac42d14e31f8a1e5452c72995f212a538e144a4"
TOKEN_SHA256="d64c2a094dbdcdf6b381c94810f9d672ac33c3c960eb6ae35ba09df0af51c3a1"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$HERE/tools"
PR153_DIR="$TOOLS/hypery11_carrier_joint"
PY="${PYTHON:-python}"
REF153=""
OUT_DIR="$HERE"
DEVICE="cpu"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --reference153) REF153="$2"; shift 2 ;;
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "usage: $0 [--reference153 <archive.zip>] [--output-dir <dir>] [--device cpu|cuda]" >&2; exit 2 ;;
  esac
done
case "$DEVICE" in cpu|cuda) ;; *) echo "ERROR: --device must be cpu or cuda" >&2; exit 2 ;; esac
CC_REQUESTED="${CC:-cc}"
CCBIN="$(command -v "$CC_REQUESTED" || true)"
[ -n "$CCBIN" ] || { echo "ERROR: a C compiler is required ('$CC_REQUESTED' not found; set CC)" >&2; exit 69; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
sha256_of() { "$PY" -I -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"; }
step() { echo "[compress $(date +%H:%M:%S)] $*"; }

# 1. PR #153's archive
if [ -n "$REF153" ]; then
  cp "$REF153" "$WORK/pr153.zip"
else
  step "downloading $PR153_URL"
  curl -L --fail --silent --show-error -o "$WORK/pr153.zip" "$PR153_URL"
fi
GOT="$(sha256_of "$WORK/pr153.zip")"
if [ "$GOT" != "$PR153_SHA256" ]; then
  echo "ERROR: PR #153 archive sha256 is $GOT, expected $PR153_SHA256; refusing to continue" >&2
  exit 1
fi
step "PR #153 archive ok (sha256 $GOT)"

# 2. token maps, with #153's own decoder (native libraries built with the flags of #153's vendor/inflate.sh)
PR141="$PR153_DIR/vendor/external/pr141"
"$CCBIN" -O3 -std=c11 -shared -fPIC "$PR141/runtime/entropy/rc64_backend.c" -o "$WORK/rc64_backend.so"
"$CCBIN" -O3 -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \
  "$PR141/runtime/f26_corrector_native.c" -lm -o "$WORK/f26_corrector_native.so"
step "decoding #153's token maps on $DEVICE"
CPR1_RC64_LIBRARY="$WORK/rc64_backend.so" F26_CORRECTOR_NATIVE_LIBRARY="$WORK/f26_corrector_native.so" \
  "$PY" -I -B "$TOOLS/decode_tokens153.py" "$PR153_DIR" "$WORK/pr153.zip" "$WORK/tokens.u8" "$DEVICE"
[ "$(sha256_of "$WORK/tokens.u8")" = "$TOKEN_SHA256" ] || { echo "ERROR: token maps differ from $TOKEN_SHA256" >&2; exit 1; }

# 3. TCM stream, encoded by this folder's tcm/tcm.c (same build flags as inflate.sh)
"$CCBIN" -O2 -std=c11 -ffp-contract=off -o "$WORK/tcm" "$HERE/tcm/tcm.c"
if grep -qv '^[a-z0-9]*=[0-9]*$' "$HERE/tcm/encoder_params.txt"; then
  echo "ERROR: tcm/encoder_params.txt must hold key=value lines only" >&2; exit 2
fi
set -f
PARAMS=( $(cat "$HERE/tcm/encoder_params.txt") )
set +f
step "encoding the token maps with tcm/tcm.c (${#PARAMS[@]} parameters)"
"$WORK/tcm" e "$WORK/tokens.u8" "$WORK/tokens.tcm" "${PARAMS[@]}"

# 4. archive: #153's non-token sections (re-coded by tcx_format) + the TCM stream, with all checks
"$PY" -I -B "$TOOLS/ref153_parts.py" "$PR153_DIR" "$WORK/pr153.zip" "$WORK/ref"
mkdir -p "$OUT_DIR"
"$PY" -I -B "$TOOLS/build_archive.py" --tcm "$WORK/tokens.tcm" --pr153-archive "$WORK/pr153.zip" \
  --pr153-dir "$PR153_DIR" --ref-parts "$WORK/ref/ref153_parts.json" --tcm-bin "$WORK/tcm" --out "$OUT_DIR/archive.zip"
step "wrote $OUT_DIR/archive.zip ($(wc -c < "$OUT_DIR/archive.zip" | tr -d ' ') B, sha256 $(sha256_of "$OUT_DIR/archive.zip"))"
