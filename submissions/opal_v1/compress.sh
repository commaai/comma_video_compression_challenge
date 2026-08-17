#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${HERE}/archive.zip"
OUT="${HERE}/archive.zip"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "usage: compress.sh [--source PROMOTED_ARCHIVE] [--out PATH]" >&2; exit 2 ;;
  esac
done

if [[ ! -f "$SOURCE" ]]; then
  echo "missing frozen OPAL archive: $SOURCE" >&2
  exit 1
fi

if [[ "$SOURCE" != "$OUT" ]]; then
  mkdir -p "$(dirname "$OUT")"
  cp -- "$SOURCE" "$OUT"
fi

python3 "$HERE/verify_submission.py" --archive "$OUT"
echo "verified frozen OPAL archive at $OUT"
