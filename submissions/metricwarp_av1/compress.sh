#!/usr/bin/env bash
# Reproduce archive.zip from videos/0.mkv. Run from anywhere; operates in the repo root.
# Requires: repo venv (uv sync --group cpu), git-lfs assets (video + models),
# and an ffmpeg with libsvtav1 (tools/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg,
# or set FFMPEG_BIN). The PoseNet-guided search takes ~3-4 h on 8 CPU cores.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
FFMPEG="${FFMPEG_BIN:-${ROOT}/tools/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg}"

# 0) bootstrap the search tooling into the repo-root work/ dir it expects
mkdir -p "${ROOT}/work"
cp "${HERE}"/tools/*.py "${ROOT}/work/"

# 1) decode ground truth exactly like the CPU harness (PyAV + BT.601)
[ -f work/gt.raw ] || "$PY" work/decode_gt.py

# 2) cache the metric nets' outputs on the ground truth (search targets)
[ -f work/gt_pose.npy ] || "$PY" work/precompute_gt.py

# 3) full-res SVT-AV1 encode + OBU strip
"$PY" - <<'EOF'
import sys; sys.path.insert(0, 'work')
from pathlib import Path
from sweep import encode
ivf = Path('work/finalA.ivf')
encode('svt', 56, 10, 'tune=2', 2, 1200, ivf, css='420', eh=874, ew=1164)
print(ivf.stat().st_size)
EOF
"$FFMPEG" -y -loglevel error -i work/finalA.ivf -c copy -f obu work/finalA.obu

# 4) metric-space cache of the decode + a 512-space uint8 render of it
"$PY" - <<'EOF'
import sys; sys.path.insert(0, 'work')
from correct_full import build_cache
import numpy as np
build_cache('work/finalA.obu', 'work/finalA_A.npy')
cache = np.load('work/finalA_A.npy', mmap_mode='r')
out = np.empty((1200,384,512,3), dtype=np.uint8)
for s in range(0,1200,100):
    out[s:s+100] = np.clip(np.round(np.asarray(cache[s:s+100], dtype=np.float32)), 0, 255).astype(np.uint8)
out.tofile('work/finalA_512.raw')
EOF

# 5) SegNet-guided greedy tile fixes on odd frames (integer, uint8-exact)
for r in "0 150" "150 300" "300 450" "450 600"; do
  set -- $r
  "$PY" work/segfix_full.py work/finalA_512.raw "work/segfix_$1_$2.npz" --start "$1" --end "$2"
done
"$PY" work/merge_segfix.py work/segfix_final.npz work/segfix_0_150.npz work/segfix_150_300.npz work/segfix_300_450.npz work/segfix_450_600.npz
"$PY" work/make_cache_v3.py work/finalA_A.npy work/segfix_final.npz work/finalA_A_v3.npy

# 6) PoseNet-guided per-pair warp search on the seg-fixed frames (rounding-aware)
for r in "0 150" "150 300" "300 450" "450 600"; do
  set -- $r
  "$PY" work/correct_full.py work/finalA.obu "work/corrC_$1_$2.npz" --cache work/finalA_A_v3.npy --start "$1" --end "$2"
done
"$PY" work/merge_corr.py work/corrC_final.npz work/corrC_0_150.npz work/corrC_150_300.npz work/corrC_300_450.npz work/corrC_450_600.npz

# 7) package
"$PY" work/package.py --name metricwarp_av1 --stream work/finalA.obu --corrections work/corrC_final.npz --segfix work/segfix_final.npz --chain hybrid2
echo "done: $(stat -c%s "${HERE}/archive.zip") bytes"
