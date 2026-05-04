#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE="${HERE}/archive.zip"
TMP="${ARCHIVE}.tmp"

URL="https://github.com/BradyMeighan/comma_video_compression_challenge/releases/download/hnerv-lc-v2-archive/archive.zip"
EXPECTED_SHA256="afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641"

rm -f "${TMP}"

if command -v curl >/dev/null 2>&1; then
  curl -L --fail --silent --show-error "${URL}" -o "${TMP}"
elif command -v python3 >/dev/null 2>&1; then
  python3 - "${URL}" "${TMP}" <<'PY'
import sys
import urllib.request

url, out = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url) as r, open(out, "wb") as f:
    f.write(r.read())
PY
else
  echo "ERROR: need curl or python3 to fetch upstream archive" >&2
  exit 1
fi

ACTUAL_SHA256="$(python3 - "${TMP}" <<'PY'
import hashlib
import sys

h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print(h.hexdigest())
PY
)"

if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]; then
  echo "ERROR: archive SHA256 mismatch" >&2
  echo "expected: ${EXPECTED_SHA256}" >&2
  echo "actual:   ${ACTUAL_SHA256}" >&2
  rm -f "${TMP}"
  exit 1
fi

mv "${TMP}" "${ARCHIVE}"
echo "wrote ${ARCHIVE}"
echo "sha256 ${EXPECTED_SHA256}"
