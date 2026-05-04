#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${REPO_ROOT:-$(cd "${HERE}/../.." && pwd)}"

SUBMISSION_DIR="${1:-$HERE}"
VIDEO_NAMES_FILE="${2:-$ROOT/public_test_video_names.txt}"
DEVICE="${DEVICE:-cuda}"
RUN_OFFICIAL="${RUN_OFFICIAL:-1}"
KEEP_TMP="${KEEP_TMP:-0}"

EXPECTED_FRAMES="${EXPECTED_FRAMES:-1200}"
FRAME_W="${FRAME_W:-1164}"
FRAME_H="${FRAME_H:-874}"
FRAME_C="${FRAME_C:-3}"
EXPECTED_RAW_SIZE=$((EXPECTED_FRAMES * FRAME_W * FRAME_H * FRAME_C))

if [ ! -f "${SUBMISSION_DIR}/archive.zip" ]; then
  echo "ERROR: ${SUBMISSION_DIR}/archive.zip not found" >&2
  exit 1
fi

tmp="$(mktemp -d)"
cleanup() {
  if [ "$KEEP_TMP" != "1" ]; then
    rm -rf "$tmp"
  else
    echo "Keeping temp dir: $tmp"
  fi
}
trap cleanup EXIT

mkdir -p "$tmp/submission" "$tmp/empty_cwd"
cp "${SUBMISSION_DIR}/archive.zip" "$tmp/submission/archive.zip"
cp "${SUBMISSION_DIR}/inflate.sh" "$tmp/submission/inflate.sh"
cp "${SUBMISSION_DIR}"/*.py "$tmp/submission/"
cp "$VIDEO_NAMES_FILE" "$tmp/public_test_video_names.txt"
chmod +x "$tmp/submission/inflate.sh"

mkdir -p "$tmp/submission/archive" "$tmp/submission/inflated"
unzip -q "$tmp/submission/archive.zip" -d "$tmp/submission/archive"

(
  cd "$tmp/empty_cwd"
  bash "$tmp/submission/inflate.sh" \
    "$tmp/submission/archive" \
    "$tmp/submission/inflated" \
    "$tmp/public_test_video_names.txt"
)

missing=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  base="${line%.*}"
  raw="$tmp/submission/inflated/${base}.raw"
  if [ ! -f "$raw" ]; then
    echo "ERROR: missing inflated file: $raw" >&2
    missing=$((missing + 1))
    continue
  fi
  size="$(stat -c '%s' "$raw")"
  if [ "$size" -ne "$EXPECTED_RAW_SIZE" ]; then
    echo "ERROR: wrong raw size for $raw: got $size, expected $EXPECTED_RAW_SIZE" >&2
    exit 1
  fi
done < "$tmp/public_test_video_names.txt"

if [ "$missing" -gt 0 ]; then
  exit 1
fi

echo "No-source-video inflation: PASS"

if [ "$RUN_OFFICIAL" = "1" ]; then
  if [ -n "${PYTHON_BIN:-}" ]; then
    PATH="$(dirname "$PYTHON_BIN"):$PATH" bash "$ROOT/evaluate.sh" --submission-dir "$SUBMISSION_DIR" --device "$DEVICE"
  else
    bash "$ROOT/evaluate.sh" --submission-dir "$SUBMISSION_DIR" --device "$DEVICE"
  fi
fi
