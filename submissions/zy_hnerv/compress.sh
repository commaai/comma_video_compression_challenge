#!/usr/bin/env bash
#
# Rebuild submissions/zy_hnerv/archive.zip from the HNeRV pipeline in ./hnerv.
#
#   --mode repack    (default) re-zip the existing runs/<run>/final_archive.bin
#   --mode finalize  re-run the quantisation sweep on runs/<run>/best.pt
#   --mode train     precompute + train + finalize into a new run
#
# repack and finalize reproduce the submitted 277,090-byte archive exactly.
# train does not: the loop runs against a wall-clock budget (--minutes), so the
# step count and therefore the result depend on the machine it runs on.
#
# Examples:
#   bash compress.sh
#   bash compress.sh --mode finalize
#   bash compress.sh --mode train --run run1 --resume D --minutes 240 --threads 16
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
PIPELINE="${HERE}/hnerv"

# hnerv/common.py defaults to treating its own parent as the challenge repo,
# which was true when the pipeline lived at <repo>/hnerv. It now sits a level
# deeper, so the root has to be passed in explicitly.
export COMMA_REPO="${REPO}"

MODE="repack"
RUN="D"
RESUME=""
NAME="zy_hnerv"
MINUTES=240
THREADS="$(nproc 2>/dev/null || echo 2)"

usage() {
  echo "Usage: $0 [--mode repack|finalize|train] [--run <name>] [--resume <name>]" >&2
  echo "          [--name <dir>] [--minutes <n>] [--threads <n>]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)    MODE="$2";    shift 2 ;;
    --run)     RUN="$2";     shift 2 ;;
    --resume)  RESUME="$2";  shift 2 ;;
    --name)    NAME="$2";    shift 2 ;;
    --minutes) MINUTES="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

case "$MODE" in
  repack|finalize|train) ;;
  *) echo "Unknown mode: ${MODE}" >&2; usage; exit 2 ;;
esac

python -c 'import brotli' 2>/dev/null || {
  echo "ERROR: brotli is missing. Run: uv pip install brotli" >&2
  exit 1
}

cd "$PIPELINE"

# precompute.py decodes 0.mkv once and caches SegNet/PoseNet targets plus the
# 384x512 ground truth. Both train.py and finalize.py read it; ~830MB, ~4 min.
ensure_cache() {
  local missing=0
  for f in seg_targets.npy gt384.npy pose_targets.npy; do
    [[ -f "cache/${f}" ]] || missing=1
  done
  if [[ "$missing" -eq 1 ]]; then
    echo "==> precompute (writes ~830MB to hnerv/cache/)"
    python precompute.py
  else
    echo "==> cache present, skipping precompute"
  fi
}

if [[ "$MODE" == "train" ]]; then
  # train.py writes best.pt into runs/<run>, so training into a run that already
  # holds a checkpoint would destroy it. Warm-starting is what --resume is for.
  if [[ -f "runs/${RUN}/best.pt" ]]; then
    echo "ERROR: runs/${RUN}/best.pt exists and would be overwritten." >&2
    echo "       Pick a fresh --run, and pass --resume ${RUN} to start from it." >&2
    exit 1
  fi

  TRAIN_ARGS=(
    --name "$RUN" --minutes "$MINUTES" --threads "$THREADS" --device auto
    --seg-frac 1.0 --seg-loss l7 --seg-w 50
    --pixel-w 10 --pixel-decay-min 99999
    --lr 4e-4 --latent-lr-mult 10 --ema 0.99 --batch 8
    --qat-after 0.85 --eval-every-min 15 --eval-pairs 48
  )
  if [[ -n "$RESUME" ]]; then
    [[ -f "runs/${RESUME}/best.pt" || -f "runs/${RESUME}/last.pt" ]] || {
      echo "ERROR: no checkpoint in runs/${RESUME}" >&2; exit 1; }
    TRAIN_ARGS+=(--resume "$RESUME")
  fi

  ensure_cache
  echo "==> train run=${RUN} minutes=${MINUTES} threads=${THREADS}"
  python train.py "${TRAIN_ARGS[@]}"
fi

if [[ "$MODE" == "train" || "$MODE" == "finalize" ]]; then
  [[ -f "runs/${RUN}/best.pt" ]] || {
    echo "ERROR: runs/${RUN}/best.pt not found" >&2; exit 1; }
  ensure_cache
  echo "==> finalize (rate/distortion sweep over weight and latent levels)"
  python finalize.py --run "$RUN"
fi

[[ -f "runs/${RUN}/final_archive.bin" ]] || {
  echo "ERROR: runs/${RUN}/final_archive.bin not found; try --mode finalize" >&2
  exit 1; }

echo "==> package into submissions/${NAME}"
python package.py --run "$RUN" --name "$NAME" --archive final_archive.bin
