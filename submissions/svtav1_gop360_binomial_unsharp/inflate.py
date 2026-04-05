#!/usr/bin/env python
"""
inflate.py  —  Decode compressed MKV -> raw RGB frames (.raw)

Uses the exact binomial unsharp kernel from PR#24 (svtav1_cheetah, score 2.05).
Kernel: C(8,k) outer product / 65536, sigma_eff ~= 1.414, amount = 0.85

Reads optional sibling inflate_config.json for per-submission tuning:
  {
    "unsharp": true,
    "unsharp_amount": 0.85
  }

Usage:
  python inflate.py <src.mkv> <dst.raw>
"""

import json
import sys
from pathlib import Path

# ── Repo root discovery ───────────────────────────────────────────────────────
def _find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent,
                      here.parent.parent.parent]:
        if (candidate / "evaluate.py").exists():
            return candidate
    return here

_ROOT = _find_repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import av
import torch
import torch.nn.functional as F

from frame_utils import camera_size, yuv420_to_rgb

# ── Exact binomial kernel from PR#24 (svtav1_cheetah, score 2.05) ─────────────
# Row vector: Pascal's triangle row 8 = C(8,k) for k=0..8
# Sum = 256, outer product sums to 65536 = 2^16
# Effective Gaussian sigma ~= sqrt(8/4) = 1.414
BINOMIAL_KERNEL = torch.tensor([
    [1.,   8.,   28.,   56.,   70.,   56.,   28.,   8.,   1.],
    [8.,   64.,  224.,  448.,  560.,  448.,  224.,  64.,   8.],
    [28.,  224., 784.,  1568., 1960., 1568., 784.,  224.,  28.],
    [56.,  448., 1568., 3136., 3920., 3136., 1568., 448.,  56.],
    [70.,  560., 1960., 3920., 4900., 3920., 1960., 560.,  70.],
    [56.,  448., 1568., 3136., 3920., 3136., 1568., 448.,  56.],
    [28.,  224., 784.,  1568., 1960., 1568., 784.,  224.,  28.],
    [8.,   64.,  224.,  448.,  560.,  448.,  224.,  64.,   8.],
    [1.,   8.,   28.,   56.,   70.,   56.,   28.,   8.,   1.],
], dtype=torch.float32) / 65536.0


def _load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent / "inflate_config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def decode_and_write_raw(video_path: str, raw_path: str) -> int:
    cfg           = _load_config()
    do_unsharp    = bool(cfg.get("unsharp", False))
    unsharp_amount = float(cfg.get("unsharp_amount", 0.85))

    target_w, target_h = camera_size  # (1164, 874)

    fmt       = "hevc" if video_path.endswith(".hevc") else None
    container = av.open(video_path, format=fmt)
    stream    = container.streams.video[0]

    n_frames = 0
    with open(raw_path, "wb") as out_f:
        for frame in container.decode(stream):
            rgb = yuv420_to_rgb(frame)          # (H, W, 3) uint8
            H, W, _ = rgb.shape

            if H != target_h or W != target_w:
                x = rgb.permute(2, 0, 1).unsqueeze(0).float()   # (1,3,H,W)
                x = F.interpolate(x, size=(target_h, target_w),
                                  mode="bicubic", align_corners=False)

                if do_unsharp:
                    kernel = BINOMIAL_KERNEL.to(device=x.device).expand(3, 1, 9, 9)
                    blur   = F.conv2d(x, kernel, padding=4, groups=3)
                    x      = x + unsharp_amount * (x - blur)

                x   = x.clamp(0, 255)
                rgb = x.squeeze(0).permute(1, 2, 0).round().to(torch.uint8)

            out_f.write(rgb.contiguous().numpy().tobytes())
            n_frames += 1

    container.close()
    return n_frames


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <src.mkv> <dst.raw>", file=sys.stderr)
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    if not Path(src).exists():
        print(f"[ERROR] Source not found: {src}", file=sys.stderr)
        sys.exit(1)

    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    cfg = _load_config()
    if cfg.get("unsharp"):
        print(f"  unsharp: binomial 9x9  amount={cfg.get('unsharp_amount', 0.85)}")

    n = decode_and_write_raw(src, dst)
    print(f"saved {n} frames")


if __name__ == "__main__":
    main()
