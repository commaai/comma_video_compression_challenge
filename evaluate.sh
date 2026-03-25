#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SUBMISSION_DIR="${HERE}/submissions/baseline"
VIDEO_NAMES_FILE="${HERE}/public_test_video_names.txt"
DEVICE="cpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submission-dir|--submission_dir)
      SUBMISSION_DIR="${2%/}"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    --device)
      DEVICE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--submission-dir <dir>] [--video-names-file <file>] [--device <cpu|cuda|mps>]" >&2
      exit 2 ;;
  esac
done

ARCHIVE_ZIP="${SUBMISSION_DIR}/archive.zip"
ARCHIVE_DIR="${SUBMISSION_DIR}/archive"
INFLATED_DIR="${SUBMISSION_DIR}/inflated"

INFLATE_SH="${SUBMISSION_DIR}/inflate.sh"

if [ ! -f "$ARCHIVE_ZIP" ]; then
  echo "ERROR: ${ARCHIVE_ZIP} not found" >&2
  exit 1
fi

if [ ! -f "$INFLATE_SH" ]; then
  echo "ERROR: ${INFLATE_SH} not found" >&2
  exit 1
fi

# unzip
rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"
unzip -o "$ARCHIVE_ZIP" -d "$ARCHIVE_DIR"

# inflate
bash "${SUBMISSION_DIR}/inflate.sh" "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE"

# assert all videos have been inflated
MISSING=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  RAW_PATH="${INFLATED_DIR}/${BASE}.raw"
  if [ ! -f "$RAW_PATH" ]; then
    echo "ERROR: missing inflated file: ${RAW_PATH}" >&2
    MISSING=$((MISSING + 1))
  fi
done < "$VIDEO_NAMES_FILE"

if [ "$MISSING" -gt 0 ]; then
  echo "ERROR: ${MISSING} video(s) not inflated" >&2
  exit 1
fi

echo "All videos inflated to ${INFLATED_DIR}"

# evaluate
python "$HERE/evaluate.py" \
  --submission-dir "$SUBMISSION_DIR" \
  --uncompressed-dir "$HERE/videos" \
  --report "$SUBMISSION_DIR/report.txt" \
  --video-names-file "$VIDEO_NAMES_FILE" \
  --device "$DEVICE"

echo "Evaluation complete. Report saved to ${SUBMISSION_DIR}/report.txt"
