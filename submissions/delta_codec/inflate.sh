#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
PYTHON_BIN="${ROOT}/.venv/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="python3"

cd "$ROOT"
"$PYTHON_BIN" -m submissions.delta_codec.inflate "$1" "$2" "$3"

