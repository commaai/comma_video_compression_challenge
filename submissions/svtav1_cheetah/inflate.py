#!/usr/bin/env python
import av, torch
import torch.nn.functional as F
from frame_utils import camera_size, yuv420_to_rgb

UNSHARP_KERNEL = torch.tensor([
  [1., 8., 28., 56., 70., 56., 28., 8., 1.],
  [8., 64., 224., 448., 560., 448., 224., 64., 8.],
  [28., 224., 784., 1568., 1960., 1568., 784., 224., 28.],
  [56., 448., 1568., 3136., 3920., 3136., 1568., 448., 56.],
  [70., 560., 1960., 3920., 4900., 3920., 1960., 560., 70.],
  [56., 448., 1568., 3136., 3920., 3136., 1568., 448., 56.],
  [28., 224., 784., 1568., 1960., 1568., 784., 224., 28.],
  [8., 64., 224., 448., 560., 448., 224., 64., 8.],
  [1., 8., 28., 56., 70., 56., 28., 8., 1.],
], dtype=torch.float32) / 65536.0


def decode_and_resize_to_file(video_path: str, dst: str):
  target_w, target_h = camera_size
  fmt = 'hevc' if video_path.endswith('.hevc') else None
  container = av.open(video_path, format=fmt)
  stream = container.streams.video[0]
  n = 0
  with open(dst, 'wb') as f:
    for frame in container.decode(stream):
      t = yuv420_to_rgb(frame)  # (H, W, 3)
      H, W, _ = t.shape
      if H != target_h or W != target_w:
        x = t.permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
        x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
        kernel = UNSHARP_KERNEL.to(device=x.device).expand(3, 1, 9, 9)
        blur = F.conv2d(x, kernel, padding=4, groups=3)
        x = x + 0.85 * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
      f.write(t.contiguous().numpy().tobytes())
      n += 1
  container.close()
  return n


if __name__ == "__main__":
  import sys
  src, dst = sys.argv[1], sys.argv[2]
  n = decode_and_resize_to_file(src, dst)
  print(f"saved {n} frames")
