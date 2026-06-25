#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Reproduce archive.zip from the attributed open HNeRV weights (no training).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi
cd "$ROOT"
"$PY" "$HERE/compress.py"
echo "Wrote ${HERE}/archive.zip"
