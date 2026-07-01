#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Rebuild archive.zip from the pinned PR #101 / PR #110 release archives.
#
# Usage: compress.sh [--pr101 PATH] [--pr110 PATH] [--workdir DIR] [--out PATH]
#
# If --pr101/--pr110 are not given, the archives are downloaded with curl into
# the workdir and SHA-256-verified there (compress.py re-verifies them again
# before use either way). Run inside the challenge environment, e.g.:
#   cd <challenge-repo> && uv run --no-sync --group cpu \
#     bash <this-dir>/compress.sh
# Encode-side deps: numpy, constriction, brotli (no torch needed).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PR101_URL="https://github.com/SajayR/comma_video_compression_challenge/releases/download/hnerv-ft-microcodec-v1/archive.zip"
PR110_URL="https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip"
PR101_SHA256="b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e"
PR110_SHA256="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"

PR101=""
PR110=""
WORKDIR="${HERE}/build"
OUT="${HERE}/archive.zip"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pr101)   PR101="$2"; shift 2 ;;
    --pr110)   PR110="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --out)     OUT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -n "${PACT_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="$PACT_PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "ERROR: neither python nor python3 is available" >&2
  exit 127
fi

fetch() { # fetch <url> <dest> <sha256>
  local url="$1" dest="$2" sha="$3"
  if [ ! -f "$dest" ]; then
    echo "downloading $url"
    curl -fsSL -o "$dest" "$url"
  fi
  echo "${sha}  ${dest}" | sha256sum -c -
}

if [ -z "$PR101" ]; then
  mkdir -p "$WORKDIR"
  PR101="${WORKDIR}/pr101_archive.zip"
  fetch "$PR101_URL" "$PR101" "$PR101_SHA256"
fi
if [ -z "$PR110" ]; then
  mkdir -p "$WORKDIR"
  PR110="${WORKDIR}/pr110_archive.zip"
  fetch "$PR110_URL" "$PR110" "$PR110_SHA256"
fi

"$PYTHON_BIN" "$HERE/compress.py" --pr101 "$PR101" --pr110 "$PR110" --out "$OUT"
