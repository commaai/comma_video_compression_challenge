#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Rebuild archive.zip for qlp_exactgrid — self-contained and byte-exact.
#
# archive.zip = (a) base member: the frozen PR #101 HNeRV decoder + the
# exact-grid-polished per-pair latents, entropy-coded with PR #112's context
# range coder (bundled pack_base.py); (b) FS1B tail: the re-searched frame0
# pose selector (committed selection.json, appended by fs1b_compress.py).
#
# The two inputs — the frozen decoder and the polished latents — are hosted as
# release assets (the polish itself is a ~90 min non-deterministic MPS run,
# exact-grid boundary-loss latent polish; see README). Given those two tensors,
# this script rebuilds archive.zip DETERMINISTICALLY and asserts its SHA-256.
#
# Usage: compress.sh            # fetches decoder/latents from the release
#        compress.sh <decoder.pt> <latents.pt>   # use local tensors
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL="https://github.com/ryanli0070/comma_video_compression_challenge/releases/download/qlp-exactgrid-v1"
WANT_SHA="869e666b4dd65d8da59f39a375505b65af28a695d94c73675e63864d678b8d1b"
PY="${PACT_PYTHON_BIN:-$(command -v python || command -v python3)}"

DEC="${1:-$HERE/decoder.pt}"; LAT="${2:-$HERE/latents.pt}"
[ -f "$DEC" ] || { echo "fetching decoder.pt"; curl -fL -o "$DEC" "$REL/decoder.pt"; }
[ -f "$LAT" ] || { echo "fetching latents.pt"; curl -fL -o "$LAT" "$REL/latents.pt"; }

"$PY" "$HERE/pack_base.py" "$DEC" "$LAT" "$HERE/_base.zip"
"$PY" "$HERE/fs1b_compress.py" "$HERE/_base.zip" "$HERE/selection.json" "$HERE/archive.zip" --base-kind 0
rm -f "$HERE/_base.zip"

GOT=$(shasum -a 256 "$HERE/archive.zip" | cut -d' ' -f1)
echo "archive.zip SHA-256: $GOT"
[ "$GOT" = "$WANT_SHA" ] && echo "OK: byte-exact ($(wc -c <"$HERE/archive.zip") bytes)" \
  || { echo "MISMATCH (expected $WANT_SHA)"; exit 1; }
