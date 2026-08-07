#!/usr/bin/env bash
# Train the HNeRV decoder + latents from scratch, then build archive.zip.
#
# ~50 hours on a single GPU from random init.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

cd "$ROOT"
python -m submissions.hnerv_muon.src.train

# The training pipeline writes the final compressed file as
# <out_root>/submission_archive/0.bin. Find the most recent run and zip it.
LATEST_RUN=$(ls -dt "$HERE"/src/ckpts/run_*/ 2>/dev/null | head -n1)
if [ -z "$LATEST_RUN" ]; then
  echo "ERROR: no run dir found under $HERE/src/ckpts/" >&2
  exit 1
fi
LATEST_RUN="${LATEST_RUN%/}"
ARCHIVE_BIN="${LATEST_RUN}/submission_archive/0.bin"

if [ ! -f "$ARCHIVE_BIN" ]; then
  echo "ERROR: $ARCHIVE_BIN not found" >&2
  exit 1
fi

cd "$(dirname "$ARCHIVE_BIN")"
rm -f "$HERE/archive.zip"
zip -j "$HERE/archive.zip" "0.bin"
echo "Wrote $HERE/archive.zip ($(stat -c%s "$HERE/archive.zip" 2>/dev/null || stat -f%z "$HERE/archive.zip") bytes)"
