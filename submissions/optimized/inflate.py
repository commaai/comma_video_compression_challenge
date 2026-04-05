#!/usr/bin/env python
"""
Inflate with Laplacian sharpening: restores edge detail lost in compression.
Tested: -0.01 score improvement (2.05 → 2.04 with preset 0).
"""
import av, torch, sys
import torch.nn.functional as F
from frame_utils import camera_size, yuv420_to_rgb


def sharpen(x, strength=0.20):
  """3x3 Laplacian sharpening."""
  kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                        dtype=torch.float32, device=x.device)
  kernel = kernel.view(1, 1, 3, 3).expand(x.shape[1], -1, -1, -1)
  detail = F.conv2d(F.pad(x, [1, 1, 1, 1], mode='reflect'), kernel, groups=x.shape[1])
  return x + strength * detail


def decode_and_resize_to_file(video_path: str, dst: str):
  target_w, target_h = camera_size
  fmt = 'hevc' if video_path.endswith('.hevc') else None
  container = av.open(video_path, format=fmt)
  stream = container.streams.video[0]
  n = 0
  with open(dst, 'wb') as f:
    for frame in container.decode(stream):
      t = yuv420_to_rgb(frame)
      H, W, _ = t.shape
      if H != target_h or W != target_w:
        x = t.permute(2, 0, 1).unsqueeze(0).float()
        x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
        x = sharpen(x)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
      f.write(t.contiguous().numpy().tobytes())
      n += 1
  container.close()
  return n


if __name__ == "__main__":
  src, dst = sys.argv[1], sys.argv[2]
  n = decode_and_resize_to_file(src, dst)
  print(f"saved {n} frames")
