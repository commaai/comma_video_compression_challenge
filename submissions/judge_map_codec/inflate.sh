#!/usr/bin/env bash
# comma runs: inflate.sh <archive_dir> <output_dir> <video_names_file>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${PYTHON:-python3}" "$HERE/inflate.py" "$1" "$2" "$3"
