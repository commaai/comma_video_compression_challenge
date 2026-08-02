#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_URL="${ARCHIVE_URL:-https://github.com/codexblack/comma_video_compression_challenge/releases/download/semantic-pose-HPAC_CPR1_polished-f26/archive.zip}"
OUT="${HERE}/archive.zip"
TEMPORARY=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --archive-url) ARCHIVE_URL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "usage: compress.sh [--archive-url URL] [--out PATH]" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$OUT")"
TEMPORARY="$(mktemp "${OUT}.tmp.XXXXXX")"
trap 'rm -f -- "$TEMPORARY"' EXIT

if command -v curl >/dev/null 2>&1; then
  curl -fsSL --retry 3 --retry-all-errors "$ARCHIVE_URL" -o "$TEMPORARY"
else
  python3 - "$ARCHIVE_URL" "$TEMPORARY" <<'PY'
import sys
import urllib.request

urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
fi

python3 - "$TEMPORARY" <<'PY'
import hashlib
import sys
import zipfile
from pathlib import Path

expected_sha256 = "12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004"
expected_bytes = 186_724
path = Path(sys.argv[1])
if path.stat().st_size != expected_bytes:
    raise SystemExit("archive size does not match promoted F26")
with path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
if digest != expected_sha256:
    raise SystemExit("archive digest does not match promoted F26")
with zipfile.ZipFile(path) as archive:
    entries = archive.infolist()
    if len(entries) != 1 or entries[0].filename != "p":
        raise SystemExit("archive must contain exactly one payload named p")
print(f"verified {path.name} sha256={digest}")
PY

mv -f "$TEMPORARY" "$OUT"
TEMPORARY=""
echo "wrote $OUT"
