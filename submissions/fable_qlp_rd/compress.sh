#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Rebuild archive.zip for fable_qlp_rd.
#
# The archive = (a) base member: the frozen #101 decoder + exact-grid-polished
# per-pair latents, entropy-coded with PR #112's context range coder; (b) FS1B
# tail: the re-searched frame0 pose selector.
#
# Producing the polished latents from scratch is the campaign pipeline
# (non-deterministic, ~90 min on an Apple M5 Pro): work/qlp/train_qlp3.py
# (exact-grid boundary-loss latent polish, warm-started from the rate-aware QAT
# init), then the selector search in work/qlp/driver15.sh.
#
# This script performs the DETERMINISTIC final composition from those artifacts
# (frozen decoder + polished latents + selector JSON), asserting a byte-exact
# round-trip:  compress.sh [<decoder.pt> <latents.pt> <selection.json> [out.zip]]
# With no args it uses the campaign work-tree paths.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

DEC="${1:-$ROOT/work/ckpt_pr101/decoder_fp32.pt}"
LAT="${2:-$ROOT/work/qlp/ckpt_lam0_v3/best_latents.pt}"
SEL="${3:-$ROOT/work/qlp/final_selection_qlp_v3.json}"
OUT="${4:-$HERE/archive.zip}"

if [ ! -f "$LAT" ] || [ ! -f "$SEL" ]; then
  echo "Missing inputs ($LAT / $SEL). Run the campaign pipeline first (see header)," >&2
  echo "or fetch the shipped archive from the release asset." >&2
  exit 1
fi

# 1) base archive from frozen decoder + polished latents (fable_ft packer)
python "$ROOT/submissions/fable_ft/compress.py" "$DEC" "$LAT" "$ROOT/work/qlp/_qlp_base.zip"
# 2) append the FS1B selector tail (fable_selector composer)
python "$HERE/fs1b_compress.py" "$ROOT/work/qlp/_qlp_base.zip" "$SEL" "$OUT" --base-kind 0
shasum -a 256 "$OUT"
echo "Rebuilt $OUT ($(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT") bytes)"
