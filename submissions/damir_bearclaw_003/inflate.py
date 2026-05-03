#!/usr/bin/env python3
import sys

import av
import torch
import torch.nn.functional as F

from frame_utils import camera_size, yuv420_to_rgb


def decode_and_reconstruct_to_file(video_path: str, dst: str) -> int:
  target_w, target_h = camera_size
  top_h = target_h // 4
  bottom_h = target_h // 4
  mid_y0 = top_h
  mid_y1 = target_h - bottom_h
  mid_h = mid_y1 - mid_y0

  top_val = 24
  bottom_val = 32

  container = av.open(video_path)
  stream = container.streams.video[0]
  n = 0

  with open(dst, "wb") as f:
    for frame in container.decode(stream):
      t = yuv420_to_rgb(frame)
      if t.shape[0] != mid_h or t.shape[1] != target_w:
        x = t.permute(2, 0, 1).unsqueeze(0).float()
        x = F.interpolate(x, size=(mid_h, target_w), mode="bicubic", align_corners=False)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
      full = torch.empty((target_h, target_w, 3), dtype=torch.uint8)
      full[:top_h].fill_(top_val)
      full[mid_y0:mid_y1] = t
      full[mid_y1:].fill_(bottom_val)
      f.write(full.contiguous().numpy().tobytes())
      n += 1

  container.close()
  return n


if __name__ == "__main__":
  src, dst = sys.argv[1], sys.argv[2]
  n = decode_and_reconstruct_to_file(src, dst)
  print(f"saved {n} frames")
