#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: inflate.sh <archive-dir> <output-dir> <file-list>" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

CC_REQUESTED="${CC:-cc}"
CCBIN="$(command -v "$CC_REQUESTED" || true)"
if [[ -z "$CCBIN" ]]; then
  echo "semantic_joint_ctxmix requires a C compiler; '$CC_REQUESTED' is unavailable (set CC)" >&2
  exit 69
fi

if ! python - <<'PY' >/dev/null 2>&1
import importlib.metadata
import brotli
raise SystemExit(importlib.metadata.version("Brotli") != "1.2.0")
PY
then
  echo "semantic_joint_ctxmix requires Brotli==1.2.0; install it before inflation" >&2
  exit 69
fi

"$CCBIN" -O3 -std=c11 -shared -fPIC \
  "$HERE/runtime/entropy/rc64_backend.c" \
  -o "$BUILD_DIR/rc64_backend.so"
export CPR1_RC64_LIBRARY="$BUILD_DIR/rc64_backend.so"

# ddm_rr8 -- the float64 free corrector, lowered to C.  Both attempts sit inside the `if`
# CONDITION so `set -e` does NOT apply to them: a compiler that cannot build this file must
# cost the speedup, never the submission.  Failure leaves the variable unset and the decoder
# falls back to the proven Python corrector.
#
# -ffp-contract=off is LOAD-BEARING, not hygiene.  FMA contraction fuses a multiply and an
# add into a single rounding step, which would change the emitted probabilities and
# desynchronise the arithmetic decoder.
if [[ -n "${F26_CORRECTOR_NATIVE_LIBRARY:-}" ]]; then
  [[ -f "$F26_CORRECTOR_NATIVE_LIBRARY" ]] || { echo "missing F26 corrector library" >&2; exit 69; }
elif "$CCBIN" -O3 -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \
       "$HERE/runtime/f26_corrector_native.c" -lm \
       -o "$BUILD_DIR/f26_corrector_native.so" 2>/dev/null; then
  export F26_CORRECTOR_NATIVE_LIBRARY="$BUILD_DIR/f26_corrector_native.so"
else
  echo "f26 corrector native build unavailable; using the python corrector" >&2
fi
export F26_TOKEN_DECODER="${F26_TOKEN_DECODER:-python}"
if [[ "$F26_TOKEN_DECODER" == "native-hpac" ]]; then
  if [[ -n "${F26_HPAC_NATIVE_LIBRARY:-}" ]]; then
    [[ -f "$F26_HPAC_NATIVE_LIBRARY" ]] || { echo "missing F26 native library" >&2; exit 69; }
  else
    case "$(uname -s)" in
      Darwin)
        LIBOMP_PREFIX="$(brew --prefix libomp)"
        "$CCBIN" -O3 -mcpu=native -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \
          -Xpreprocessor -fopenmp -I"$LIBOMP_PREFIX/include" \
          "$HERE/runtime/f26_hpac_native.c" -L"$LIBOMP_PREFIX/lib" -lomp \
          -Wl,-rpath,"$LIBOMP_PREFIX/lib" -lm -o "$BUILD_DIR/f26_hpac_native.so" ;;
      *)
        "$CCBIN" -O3 -march=native -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \
          -fopenmp "$HERE/runtime/f26_hpac_native.c" -lm -o "$BUILD_DIR/f26_hpac_native.so" ;;
    esac
    export F26_HPAC_NATIVE_LIBRARY="$BUILD_DIR/f26_hpac_native.so"
  fi
fi

mkdir -p "$OUTPUT_DIR"
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  base="${line%.*}"
  if [[ "$base" != "0" ]]; then
    echo "unsupported public video: $line" >&2
    exit 2
  fi
  python "$HERE/inflate.py" "$DATA_DIR" "$base" "$OUTPUT_DIR/$base.raw"
done < "$FILE_LIST"
