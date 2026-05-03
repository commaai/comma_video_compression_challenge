#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

IN_DIR="${ROOT}/videos"
VIDEO_NAMES_FILE="${ROOT}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
JOBS="1"

SEGMENT_SECONDS="${CODEX_SEGMENT_SECONDS:-30}"
SEGMENT_CUTS="${CODEX_SEGMENT_CUTS:-0,24,30,36,42,48}"
DEFAULT_WIDTH="${CODEX_WIDTH:-522}"
DEFAULT_HEIGHT="${CODEX_HEIGHT:-392}"
DEFAULT_CRF="${CODEX_CRF:-35}"
DEFAULT_FILM_GRAIN="${CODEX_FILM_GRAIN:-22}"
DEFAULT_PRESET="${CODEX_PRESET:-0}"
DEFAULT_SCALE_FLAGS="${CODEX_SCALE_FLAGS:-lanczos}"
DEFAULT_GOP="${CODEX_GOP:-180}"
DEFAULT_SVT_EXTRA="${CODEX_SVT_EXTRA:-}"
DEFAULT_FILTER_EXTRA="${CODEX_FILTER_EXTRA:-}"
DEFAULT_SEGMENT1_CRF="${CODEX_SEGMENT1_CRF:-32}"
DEFAULT_SEGMENT1_FILM_GRAIN="${CODEX_SEGMENT1_FILM_GRAIN:-22}"
DEFAULT_SEGMENT1_FILTER="${CODEX_SEGMENT1_FILTER-eq=saturation=0.93}"
DEFAULT_SEGMENT2_WIDTH="${CODEX_SEGMENT2_WIDTH:-522}"
DEFAULT_SEGMENT2_HEIGHT="${CODEX_SEGMENT2_HEIGHT:-392}"
DEFAULT_SEGMENT2_CRF="${CODEX_SEGMENT2_CRF:-32}"
DEFAULT_SEGMENT2_FILM_GRAIN="${CODEX_SEGMENT2_FILM_GRAIN:-20}"
DEFAULT_SEGMENT2_SCALE_FLAGS="${CODEX_SEGMENT2_SCALE_FLAGS:-lanczos}"
DEFAULT_SEGMENT2_FILTER="${CODEX_SEGMENT2_FILTER-eq=saturation=0.91}"
DEFAULT_SEGMENT3_WIDTH="${CODEX_SEGMENT3_WIDTH:-528}"
DEFAULT_SEGMENT3_HEIGHT="${CODEX_SEGMENT3_HEIGHT:-396}"
DEFAULT_SEGMENT3_CRF="${CODEX_SEGMENT3_CRF:-33}"
DEFAULT_SEGMENT3_FILM_GRAIN="${CODEX_SEGMENT3_FILM_GRAIN:-20}"
DEFAULT_SEGMENT3_SCALE_FLAGS="${CODEX_SEGMENT3_SCALE_FLAGS:-bicubic}"
DEFAULT_SEGMENT3_FILTER="${CODEX_SEGMENT3_FILTER-eq=saturation=0.85}"
DEFAULT_SEGMENT4_WIDTH="${CODEX_SEGMENT4_WIDTH:-524}"
DEFAULT_SEGMENT4_HEIGHT="${CODEX_SEGMENT4_HEIGHT:-392}"
DEFAULT_SEGMENT4_CRF="${CODEX_SEGMENT4_CRF:-33}"
DEFAULT_SEGMENT4_FILM_GRAIN="${CODEX_SEGMENT4_FILM_GRAIN:-20}"
DEFAULT_SEGMENT4_SCALE_FLAGS="${CODEX_SEGMENT4_SCALE_FLAGS:-spline}"
DEFAULT_SEGMENT4_FILTER="${CODEX_SEGMENT4_FILTER-eq=saturation=0.85}"
DEFAULT_SEGMENT5_WIDTH="${CODEX_SEGMENT5_WIDTH:-528}"
DEFAULT_SEGMENT5_HEIGHT="${CODEX_SEGMENT5_HEIGHT:-396}"
DEFAULT_SEGMENT5_CRF="${CODEX_SEGMENT5_CRF:-33}"
DEFAULT_SEGMENT5_FILM_GRAIN="${CODEX_SEGMENT5_FILM_GRAIN:-22}"
DEFAULT_SEGMENT5_SCALE_FLAGS="${CODEX_SEGMENT5_SCALE_FLAGS:-spline}"
DEFAULT_SEGMENT5_FILTER="${CODEX_SEGMENT5_FILTER-eq=saturation=0.85}"
SEGMENT_CONFIG="${CODEX_SEGMENT_CONFIG:-}"
SIDECHANNEL_MODE="${CODEX_SIDECHANNEL_MODE:-metric-y-shift}"
SIDECHANNEL_GAIN="${CODEX_SIDECHANNEL_GAIN:-1.0}"
SIDECHANNEL_STEP="${CODEX_SIDECHANNEL_STEP:-0.5}"
SIDECHANNEL_CANDIDATES="${CODEX_SIDECHANNEL_CANDIDATES:--12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12}"
SIDECHANNEL_SHIFT_CANDIDATES="${CODEX_SIDECHANNEL_SHIFT_CANDIDATES:--5,-4,-3,-2,-1,0,1,2,3,4,5}"
SIDECHANNEL_SCORE_MODE="${CODEX_SIDECHANNEL_SCORE_MODE:-exact}"
SIDECHANNEL_DEVICE="${CODEX_SIDECHANNEL_DEVICE:-auto}"
SIDECHANNEL_PASSES="${CODEX_SIDECHANNEL_PASSES:-2}"
SIDECHANNEL_PROGRESS_INTERVAL="${CODEX_SIDECHANNEL_PROGRESS_INTERVAL:-100}"
POSTFILTER_MODE="${CODEX_POSTFILTER_MODE:-none}"
POSTFILTER_STEPS="${CODEX_POSTFILTER_STEPS:-300}"
POSTFILTER_BATCH_SIZE="${CODEX_POSTFILTER_BATCH_SIZE:-4}"
POSTFILTER_WIDTH="${CODEX_POSTFILTER_WIDTH:-8}"
POSTFILTER_SCALE="${CODEX_POSTFILTER_SCALE:-16.0}"
POSTFILTER_LR="${CODEX_POSTFILTER_LR:-0.002}"
POSTFILTER_RESIDUAL_WEIGHT="${CODEX_POSTFILTER_RESIDUAL_WEIGHT:-0.015}"
POSTFILTER_METRIC_WEIGHT="${CODEX_POSTFILTER_METRIC_WEIGHT:-0.0}"
POSTFILTER_METRIC_EVERY="${CODEX_POSTFILTER_METRIC_EVERY:-20}"
POSTFILTER_METRIC_BATCH_SIZE="${CODEX_POSTFILTER_METRIC_BATCH_SIZE:-1}"
POSTFILTER_POSE_WEIGHT="${CODEX_POSTFILTER_POSE_WEIGHT:-6.3}"
POSTFILTER_SEG_CE_WEIGHT="${CODEX_POSTFILTER_SEG_CE_WEIGHT:-0.02}"
POSTFILTER_STRIDE="${CODEX_POSTFILTER_STRIDE:-1}"
POSTFILTER_MAX_FRAMES="${CODEX_POSTFILTER_MAX_FRAMES:-0}"
POSTFILTER_SIZE="${CODEX_POSTFILTER_SIZE:-}"
POSTFILTER_DEVICE="${CODEX_POSTFILTER_DEVICE:-auto}"
POSTFILTER_SEED="${CODEX_POSTFILTER_SEED:-13}"
LATENT_CANDIDATES="${CODEX_LATENT_CANDIDATES:--8,-4,0,4,8}"
LATENT_COEFF_STEP="${CODEX_LATENT_COEFF_STEP:-0.5}"
LATENT_PASSES="${CODEX_LATENT_PASSES:-1}"
LATENT_PROGRESS_INTERVAL="${CODEX_LATENT_PROGRESS_INTERVAL:-10}"
PAIR_ASYM_MODE="${CODEX_PAIR_ASYM_MODE:-none}"
PAIR_EVEN_WIDTH="${CODEX_PAIR_EVEN_WIDTH:-384}"
PAIR_EVEN_HEIGHT="${CODEX_PAIR_EVEN_HEIGHT:-288}"
PAIR_EVEN_CRF="${CODEX_PAIR_EVEN_CRF:-45}"
PAIR_EVEN_FILM_GRAIN="${CODEX_PAIR_EVEN_FILM_GRAIN:-16}"
PAIR_EVEN_SCALE_FLAGS="${CODEX_PAIR_EVEN_SCALE_FLAGS:-bicubic}"
PAIR_EVEN_FILTER="${CODEX_PAIR_EVEN_FILTER-}"
SUBMISSION_HERE="$HERE"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --jobs)
      JOBS="$2"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--in-dir <dir>] [--jobs <n>] [--video-names-file <file>]" >&2
      exit 2 ;;
  esac
done

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

export ROOT IN_DIR ARCHIVE_DIR SEGMENT_SECONDS SEGMENT_CUTS DEFAULT_WIDTH DEFAULT_HEIGHT DEFAULT_CRF DEFAULT_FILM_GRAIN DEFAULT_PRESET DEFAULT_SCALE_FLAGS DEFAULT_GOP DEFAULT_SVT_EXTRA DEFAULT_FILTER_EXTRA DEFAULT_SEGMENT1_CRF DEFAULT_SEGMENT1_FILM_GRAIN DEFAULT_SEGMENT1_FILTER DEFAULT_SEGMENT2_WIDTH DEFAULT_SEGMENT2_HEIGHT DEFAULT_SEGMENT2_CRF DEFAULT_SEGMENT2_FILM_GRAIN DEFAULT_SEGMENT2_SCALE_FLAGS DEFAULT_SEGMENT2_FILTER DEFAULT_SEGMENT3_WIDTH DEFAULT_SEGMENT3_HEIGHT DEFAULT_SEGMENT3_CRF DEFAULT_SEGMENT3_FILM_GRAIN DEFAULT_SEGMENT3_SCALE_FLAGS DEFAULT_SEGMENT3_FILTER DEFAULT_SEGMENT4_WIDTH DEFAULT_SEGMENT4_HEIGHT DEFAULT_SEGMENT4_CRF DEFAULT_SEGMENT4_FILM_GRAIN DEFAULT_SEGMENT4_SCALE_FLAGS DEFAULT_SEGMENT4_FILTER DEFAULT_SEGMENT5_WIDTH DEFAULT_SEGMENT5_HEIGHT DEFAULT_SEGMENT5_CRF DEFAULT_SEGMENT5_FILM_GRAIN DEFAULT_SEGMENT5_SCALE_FLAGS DEFAULT_SEGMENT5_FILTER SEGMENT_CONFIG SIDECHANNEL_MODE SIDECHANNEL_GAIN SIDECHANNEL_STEP SIDECHANNEL_CANDIDATES SIDECHANNEL_SHIFT_CANDIDATES SIDECHANNEL_DEVICE SIDECHANNEL_PASSES SIDECHANNEL_PROGRESS_INTERVAL POSTFILTER_MODE POSTFILTER_STEPS POSTFILTER_BATCH_SIZE POSTFILTER_WIDTH POSTFILTER_SCALE POSTFILTER_LR POSTFILTER_RESIDUAL_WEIGHT POSTFILTER_METRIC_WEIGHT POSTFILTER_METRIC_EVERY POSTFILTER_METRIC_BATCH_SIZE POSTFILTER_POSE_WEIGHT POSTFILTER_SEG_CE_WEIGHT POSTFILTER_STRIDE POSTFILTER_MAX_FRAMES POSTFILTER_SIZE POSTFILTER_DEVICE POSTFILTER_SEED LATENT_CANDIDATES LATENT_COEFF_STEP LATENT_PASSES LATENT_PROGRESS_INTERVAL PAIR_ASYM_MODE PAIR_EVEN_WIDTH PAIR_EVEN_HEIGHT PAIR_EVEN_CRF PAIR_EVEN_FILM_GRAIN PAIR_EVEN_SCALE_FLAGS PAIR_EVEN_FILTER SUBMISSION_HERE
export CODEX_SIDECHANNEL_SHIFT_CANDIDATES="$SIDECHANNEL_SHIFT_CANDIDATES"
export CODEX_SIDECHANNEL_SCORE_MODE="$SIDECHANNEL_SCORE_MODE"

xargs -P"$JOBS" -I{} bash -lc '
  set -euo pipefail

  params_for_segment() {
    local idx="$1"
    local width="$DEFAULT_WIDTH"
    local height="$DEFAULT_HEIGHT"
    local crf="$DEFAULT_CRF"
    local grain="$DEFAULT_FILM_GRAIN"
    local scale_flags="$DEFAULT_SCALE_FLAGS"
    local gop="$DEFAULT_GOP"
    local svt_extra="$DEFAULT_SVT_EXTRA"
    local filter_extra="$DEFAULT_FILTER_EXTRA"
    if [[ "$idx" == "1" && -n "${DEFAULT_SEGMENT1_FILTER:-}" ]]; then
      crf="$DEFAULT_SEGMENT1_CRF"
      grain="$DEFAULT_SEGMENT1_FILM_GRAIN"
      filter_extra="$DEFAULT_SEGMENT1_FILTER"
    fi
    if [[ "$idx" == "2" ]]; then
      width="$DEFAULT_SEGMENT2_WIDTH"
      height="$DEFAULT_SEGMENT2_HEIGHT"
      crf="$DEFAULT_SEGMENT2_CRF"
      grain="$DEFAULT_SEGMENT2_FILM_GRAIN"
      scale_flags="$DEFAULT_SEGMENT2_SCALE_FLAGS"
      filter_extra="$DEFAULT_SEGMENT2_FILTER"
    fi
    if [[ "$idx" == "3" ]]; then
      width="$DEFAULT_SEGMENT3_WIDTH"
      height="$DEFAULT_SEGMENT3_HEIGHT"
      crf="$DEFAULT_SEGMENT3_CRF"
      grain="$DEFAULT_SEGMENT3_FILM_GRAIN"
      scale_flags="$DEFAULT_SEGMENT3_SCALE_FLAGS"
      filter_extra="$DEFAULT_SEGMENT3_FILTER"
    fi
    if [[ "$idx" == "4" ]]; then
      width="$DEFAULT_SEGMENT4_WIDTH"
      height="$DEFAULT_SEGMENT4_HEIGHT"
      crf="$DEFAULT_SEGMENT4_CRF"
      grain="$DEFAULT_SEGMENT4_FILM_GRAIN"
      scale_flags="$DEFAULT_SEGMENT4_SCALE_FLAGS"
      filter_extra="$DEFAULT_SEGMENT4_FILTER"
    fi
    if [[ "$idx" == "5" ]]; then
      width="$DEFAULT_SEGMENT5_WIDTH"
      height="$DEFAULT_SEGMENT5_HEIGHT"
      crf="$DEFAULT_SEGMENT5_CRF"
      grain="$DEFAULT_SEGMENT5_FILM_GRAIN"
      scale_flags="$DEFAULT_SEGMENT5_SCALE_FLAGS"
      filter_extra="$DEFAULT_SEGMENT5_FILTER"
    fi

    if [[ -n "${SEGMENT_CONFIG:-}" ]]; then
      local old_ifs="$IFS"
      IFS=";"
      for entry in $SEGMENT_CONFIG; do
        [[ -z "$entry" ]] && continue
        IFS=":" read -r entry_idx entry_width entry_height entry_crf entry_grain entry_flags entry_gop entry_svt_extra entry_filter_extra <<< "$entry"
        if [[ "$entry_idx" == "$idx" ]]; then
          width="${entry_width:-$width}"
          height="${entry_height:-$height}"
          crf="${entry_crf:-$crf}"
          grain="${entry_grain:-$grain}"
          scale_flags="${entry_flags:-$scale_flags}"
          gop="${entry_gop:-$gop}"
          svt_extra="${entry_svt_extra:-$svt_extra}"
          filter_extra="${entry_filter_extra:-$filter_extra}"
        fi
      done
      IFS="$old_ifs"
    fi

    printf "%s:%s:%s:%s:%s:%s:%s:%s" "$width" "$height" "$crf" "$grain" "$scale_flags" "$gop" "$svt_extra" "$filter_extra"
  }

  rel="$1"
  [[ -z "$rel" ]] && exit 0

  src="${IN_DIR}/${rel}"
  base="${rel%.*}"
  out_dir="${ARCHIVE_DIR}/${base}"
  mkdir -p "$out_dir"

  duration="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$src")"
  segment_plan="$(python3 - "$duration" "$SEGMENT_SECONDS" "${SEGMENT_CUTS:-}" <<'"'"'PY'"'"'
import math
import sys

duration = float(sys.argv[1])
segment_seconds = float(sys.argv[2])
cuts_text = sys.argv[3].replace(";", ",").strip()

if cuts_text:
  starts = []
  for item in cuts_text.split(","):
    item = item.strip()
    if not item:
      continue
    start = float(item)
    if start < 0:
      raise SystemExit("segment cuts must be non-negative")
    starts.append(start)
  starts = sorted(set(starts))
  if not starts or starts[0] != 0.0:
    starts.insert(0, 0.0)
  starts = [start for start in starts if start < duration]
else:
  starts = [idx * segment_seconds for idx in range(max(1, math.ceil(duration / segment_seconds)))]

rows = []
for idx, start in enumerate(starts):
  end = starts[idx + 1] if idx + 1 < len(starts) else duration
  remaining = max(0.0, min(end, duration) - start)
  if remaining > 0:
    rows.append((idx, start, remaining))

for idx, start, remaining in rows:
  print(f"{idx}\t{start}\t{remaining}")
PY
)"
  segment_count="$(printf "%s\n" "$segment_plan" | sed '/^$/d' | wc -l | tr -d " ")"
  {
    printf "version\t1\n"
    printf "source\t%s\n" "$rel"
    printf "segment_seconds\t%s\n" "$SEGMENT_SECONDS"
    printf "pair_asym_mode\t%s\n" "$PAIR_ASYM_MODE"
    if [[ -n "${SEGMENT_CUTS:-}" ]]; then
      printf "segment_cuts\t%s\n" "$SEGMENT_CUTS"
    fi
    printf "segments\t%s\n" "$segment_count"
  } > "${out_dir}/manifest.tsv"

  while IFS=$'"'"'\t'"'"' read -r idx start remaining; do
    [[ -z "${idx:-}" ]] && continue
    IFS=":" read -r width height crf grain scale_flags gop svt_extra filter_extra <<< "$(params_for_segment "$idx")"
    dst="${out_dir}/$(printf "%03d" "$idx").ivf"
    echo "segment ${idx}/${segment_count}: ${src} @ ${start}s for ${remaining}s -> ${dst} (${width}x${height}, crf=${crf}, grain=${grain})"
    svt_params="film-grain=${grain}:keyint=${gop}:scd=0"
    if [[ -n "${svt_extra:-}" ]]; then
      svt_params="${svt_params}:${svt_extra}"
    fi
    if [[ "${PAIR_ASYM_MODE:-none}" == "split-even-lowq" ]]; then
      dst_even="${out_dir}/$(printf "%03d" "$idx")_even.ivf"
      dst_odd="${out_dir}/$(printf "%03d" "$idx")_odd.ivf"
      even_filter_extra="${PAIR_EVEN_FILTER:-$filter_extra}"
      vf_odd="select=eq(mod(n\\,2)\\,1),setpts=N/(10*TB),scale=${width}:${height}:flags=${scale_flags}"
      vf_even="select=eq(mod(n\\,2)\\,0),setpts=N/(10*TB),scale=${PAIR_EVEN_WIDTH}:${PAIR_EVEN_HEIGHT}:flags=${PAIR_EVEN_SCALE_FLAGS}"
      if [[ -n "${filter_extra:-}" ]]; then
        vf_odd="${vf_odd},${filter_extra}"
      fi
      if [[ -n "${even_filter_extra:-}" ]]; then
        vf_even="${vf_even},${even_filter_extra}"
      fi
      svt_params_even="film-grain=${PAIR_EVEN_FILM_GRAIN}:keyint=${gop}:scd=0"
      if [[ -n "${svt_extra:-}" ]]; then
        svt_params_even="${svt_params_even}:${svt_extra}"
      fi

      ffmpeg -nostdin -y -hide_banner -loglevel warning \
        -ss "$start" -i "$src" -t "$remaining" \
        -r 20 -fflags +genpts \
        -vf "$vf_even" \
        -pix_fmt yuv420p -an \
        -c:v libsvtav1 -preset "$DEFAULT_PRESET" -crf "$PAIR_EVEN_CRF" \
        -svtav1-params "$svt_params_even" \
        -r 10 -f ivf "$dst_even"

      ffmpeg -nostdin -y -hide_banner -loglevel warning \
        -ss "$start" -i "$src" -t "$remaining" \
        -r 20 -fflags +genpts \
        -vf "$vf_odd" \
        -pix_fmt yuv420p -an \
        -c:v libsvtav1 -preset "$DEFAULT_PRESET" -crf "$crf" \
        -svtav1-params "$svt_params" \
        -r 10 -f ivf "$dst_odd"

      printf "segment_split\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$idx" "$start" "$remaining" "$PAIR_EVEN_WIDTH" "$PAIR_EVEN_HEIGHT" "$PAIR_EVEN_CRF" "$PAIR_EVEN_FILM_GRAIN" "$width" "$height" "$crf" "$dst_even" "$dst_odd" >> "${out_dir}/manifest.tsv"
      continue
    fi
    vf="scale=${width}:${height}:flags=${scale_flags}"
    output_fps="20"
    case "${PAIR_ASYM_MODE:-none}" in
      none|"")
        ;;
      odd-duplicate|odd-prevblend|odd-motion)
        vf="select=eq(mod(n\\,2)\\,1),setpts=N/(10*TB),${vf}"
        output_fps="10"
        ;;
      *)
        echo "Unknown CODEX_PAIR_ASYM_MODE: ${PAIR_ASYM_MODE}" >&2
        exit 2
        ;;
    esac
    if [[ -n "${filter_extra:-}" ]]; then
      vf="${vf},${filter_extra}"
    fi

    ffmpeg -nostdin -y -hide_banner -loglevel warning \
      -ss "$start" -i "$src" -t "$remaining" \
      -r 20 -fflags +genpts \
      -vf "$vf" \
      -pix_fmt yuv420p -an \
      -c:v libsvtav1 -preset "$DEFAULT_PRESET" -crf "$crf" \
      -svtav1-params "$svt_params" \
      -r "$output_fps" -f ivf "$dst"

    printf "segment\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$idx" "$start" "$remaining" "$width" "$height" "$crf" "$grain" "${svt_extra:-}" "${filter_extra:-}" "$dst" >> "${out_dir}/manifest.tsv"
  done <<< "$segment_plan"

  if [[ -n "${SIDECHANNEL_MODE:-}" && "${SIDECHANNEL_MODE:-}" != "none" ]]; then
    "${ROOT}/.venv/bin/python" "${SUBMISSION_HERE}/generate_sidechannel.py" \
      --mode "$SIDECHANNEL_MODE" \
      --gain "$SIDECHANNEL_GAIN" \
      --step "$SIDECHANNEL_STEP" \
      --candidates="$SIDECHANNEL_CANDIDATES" \
      --metric-device "$SIDECHANNEL_DEVICE" \
      --metric-passes "$SIDECHANNEL_PASSES" \
      --progress-interval "$SIDECHANNEL_PROGRESS_INTERVAL" \
      "$src" "$out_dir"
  fi

  if [[ -n "${POSTFILTER_MODE:-}" && "${POSTFILTER_MODE:-}" != "none" ]]; then
    "${ROOT}/.venv/bin/python" "${SUBMISSION_HERE}/train_postfilter.py" \
      --mode "$POSTFILTER_MODE" \
      --steps "$POSTFILTER_STEPS" \
      --batch-size "$POSTFILTER_BATCH_SIZE" \
      --width "$POSTFILTER_WIDTH" \
      --scale "$POSTFILTER_SCALE" \
      --lr "$POSTFILTER_LR" \
      --residual-weight "$POSTFILTER_RESIDUAL_WEIGHT" \
      --metric-weight "$POSTFILTER_METRIC_WEIGHT" \
      --metric-every "$POSTFILTER_METRIC_EVERY" \
      --metric-batch-size "$POSTFILTER_METRIC_BATCH_SIZE" \
      --pose-weight "$POSTFILTER_POSE_WEIGHT" \
      --seg-ce-weight "$POSTFILTER_SEG_CE_WEIGHT" \
      --stride "$POSTFILTER_STRIDE" \
      --max-frames "$POSTFILTER_MAX_FRAMES" \
      --size "$POSTFILTER_SIZE" \
      --device "$POSTFILTER_DEVICE" \
      --seed "$POSTFILTER_SEED" \
      --latent-candidates="$LATENT_CANDIDATES" \
      --latent-coeff-step="$LATENT_COEFF_STEP" \
      --latent-passes="$LATENT_PASSES" \
      --latent-progress-interval="$LATENT_PROGRESS_INTERVAL" \
      "$src" "$out_dir"
  fi
' _ {} < "$VIDEO_NAMES_FILE"

cd "$ARCHIVE_DIR"
rm -f "${HERE}/archive.zip"
zip -q -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip"
