"""Run the 8-stage training curriculum from random init, then build the archive.

Usage:
  python -m submissions.hnerv_muon.src.train

~50 hours on a single GPU. No resume / no mid-pipeline shortcuts — this script
is the deterministic, from-scratch reproduction path.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import torch

from data import get_default_video_path
from stages.common import train_stage
from stages import (
    stage1_v328_ce,
    stage2_v331_softplus,
    stage3_v332_smooth,
    stage4_v332_qat,
    stage5_c1a_l7,
    stage6_lambda_sweep,
    stage7_sigma_sweep,
    stage8_muon_finetune,
    codec_stage,
)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_root = HERE.parent.parent / "ckpts" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {out_root}", flush=True)

    video_path = get_default_video_path()
    shared_state = {}
    t0 = time.time()

    prev = None
    builders = [
        stage1_v328_ce.make_config,
        stage2_v331_softplus.make_config,
        stage3_v332_smooth.make_config,
        stage4_v332_qat.make_config,
        stage5_c1a_l7.make_config,
        stage6_lambda_sweep.make_config,
        stage7_sigma_sweep.make_config,
        stage8_muon_finetune.make_config,
    ]
    for i, build in enumerate(builders, start=1):
        stage_out = out_root / f"stage{i}"
        cfg = build(stage_out) if i == 1 else build(prev, stage_out)
        result = train_stage(cfg, device, video_path=video_path,
                             shared_state=shared_state)
        print(f"[Stage {i}] best={result['best_score']:.4f} at ep{result['best_ep']} "
              f"(archive {result['archive_size']:,} bytes)", flush=True)
        prev = stage_out

    codec_out = out_root / "submission_archive"
    print(f"\n[codec] re-encoding from {prev}", flush=True)
    r = codec_stage.run_codec_stage(prev, codec_out, video_path)
    print(f"[codec] archive bytes: {r['final_archive_bytes']:,}", flush=True)
    print(f"\nTotal wallclock: {(time.time() - t0) / 3600:.1f} hr", flush=True)
    print(f"Final archive: {codec_out / '0.bin'}", flush=True)


if __name__ == "__main__":
    main()
