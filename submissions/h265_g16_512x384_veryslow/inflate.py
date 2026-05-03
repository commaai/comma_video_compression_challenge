#!/usr/bin/env python
import av, torch
import torch.nn.functional as F
from frame_utils import camera_size, yuv420_to_rgb


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
        # Slightly less aggressive sharpen — veryslow gives higher base quality
        blur = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), 5, stride=1)
        x = (x + 3.0 * (x - blur)).clamp(0, 255)
        t = x.squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
      f.write(t.contiguous().numpy().tobytes())
      n += 1
  container.close()
  return n


if __name__ == "__main__":
  import sys
  src, dst = sys.argv[1], sys.argv[2]
  n = decode_and_resize_to_file(src, dst)
  print(f"saved {n} frames")
