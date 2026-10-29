#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-$(mktemp -d -t size-matters-XXXXXXXX)}"
python - "$HERE" "$RUN_DIR" <<'PY'
from pathlib import Path
import sys
source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
if destination == source or source in destination.parents:
    raise SystemExit('Choose a run directory outside the submission folder.')
PY
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
REPO="$RUN_DIR/challenge"

apt-get update -qq
apt-get install -y -qq git-lfs ffmpeg build-essential unzip
python -m pip install -q uv

GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout \
  https://github.com/commaai/comma_video_compression_challenge.git "$REPO"
git -C "$REPO" sparse-checkout init --no-cone
cat > "$REPO/.git/info/sparse-checkout" <<'PATTERNS'
/evaluate.py
/evaluate.sh
/frame_utils.py
/modules.py
/pyproject.toml
/uv.lock
/public_test_video_names.txt
/.gitattributes
/LICENSE
/models/
/videos/
PATTERNS
GIT_LFS_SKIP_SMUDGE=1 git -C "$REPO" checkout 0bbea33450719bc15cd799ece1fac4ddea1e0a14
git -C "$REPO" lfs install --local
git -C "$REPO" lfs pull --include='videos/0.mkv,models/segnet.safetensors,models/posenet.safetensors'
mkdir -p "$REPO/submissions/size_matters"
python - "$HERE" "$REPO/submissions/size_matters" <<'PY'
import shutil
import sys
shutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True,
    ignore=shutil.ignore_patterns('.venv', 'work', '__pycache__', '*.pyc', 'inflated', 'archive', '*.raw'))
PY
cd "$REPO"
uv sync --frozen --group cu128 --python 3.11
uv run --frozen --group cu128 python -c 'import torch; assert torch.cuda.is_available(), "Select a GPU runtime first"; print(torch.cuda.get_device_name())'
uv run --frozen --group cu128 python "$REPO/submissions/size_matters/verify.py"

# Show enough digits to compare against 0.125. The equation is unchanged.
uv run --frozen --group cu128 python - <<'PY'
from pathlib import Path
p = Path('evaluate.py')
s = p.read_text()
Path('evaluate_original.py').write_text(s)
for old, new in (
    ('{posenet_dist:.8f}', '{posenet_dist:.12g}'),
    ('{segnet_dist:.8f}', '{segnet_dist:.12g}'),
    ('{score:.2f}', '{score:.9f}'),
):
    if old not in s:
        raise RuntimeError(f'Evaluator format changed: {old}')
    s = s.replace(old, new)
p.write_text(s)
PY

STARTED=$(date +%s)
timeout --signal=TERM 1800 uv run --frozen --group cu128 bash evaluate.sh \
  --device cuda --submission-dir ./submissions/size_matters
ELAPSED=$(( $(date +%s) - STARTED ))
uv run --frozen --group cu128 python "$HERE/record_score.py" \
  "$REPO/submissions/size_matters" "$REPO/videos" "$ELAPSED"
cp "$REPO/submissions/size_matters/report.txt" "$RUN_DIR/report.txt"
cp "$REPO/submissions/size_matters/measured_score.json" "$RUN_DIR/measured_score.json"
echo "Results: $RUN_DIR"
