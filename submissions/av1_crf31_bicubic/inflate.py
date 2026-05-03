#!/usr/bin/env python
"""Decode -> bicubic upscale -> binomial unsharp -> raw uint8 RGB.

Bicubic beats lanczos by ~0.0001 on PoseNet in my sweeps (within noise
but consistent), and the 9-tap binomial unsharp at 27% recovers the
high-frequency edges that segnet cares about without adding visible
ringing around lane markings.
"""
import os, sys
import av, torch, numpy as np
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from frame_utils import camera_size, yuv420_to_rgb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
TARGET_W, TARGET_H = camera_size

# 9-tap binomial kernel = outer product of Pascal row 8, normalized
_row = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL = (torch.outer(_row, _row) / (_row.sum() ** 2)).to(DEVICE).expand(3, 1, 9, 9)
UNSHARP = 0.27


def inflate_one(src_path: str, dst_path: str) -> int:
  fmt = 'hevc' if src_path.endswith('.hevc') else None
  container = av.open(src_path, format=fmt)
  stream = container.streams.video[0]
  count = 0
  with open(dst_path, 'wb') as fout:
    for frame in container.decode(stream):
      rgb = yuv420_to_rgb(frame)  # (H, W, 3) uint8
      h, w, _ = rgb.shape
      if (h, w) != (TARGET_H, TARGET_W):
        up = Image.fromarray(rgb.numpy()).resize((TARGET_W, TARGET_H), Image.BICUBIC)
        x = torch.from_numpy(np.array(up)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blurred = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'), KERNEL, padding=0, groups=3)
        x = x + UNSHARP * (x - blurred)
        rgb = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
      fout.write(rgb.contiguous().numpy().tobytes())
      count += 1
  container.close()
  return count


if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("usage: inflate.py <input.mkv> <output.raw>", file=sys.stderr)
    sys.exit(2)
  n = inflate_one(sys.argv[1], sys.argv[2])
  print(f"wrote {n} frames")
