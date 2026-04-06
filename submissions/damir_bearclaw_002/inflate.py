#!/usr/bin/env python
import os

import av
import torch
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


def build_alpha_map(luma: torch.Tensor, alpha_min: float, alpha_max: float, var_k: float) -> torch.Tensor:
  local_mean = F.avg_pool2d(F.pad(luma, (4, 4, 4, 4), mode="reflect"), 9, stride=1)
  local_sq_mean = F.avg_pool2d(F.pad(luma ** 2, (4, 4, 4, 4), mode="reflect"), 9, stride=1)
  local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)
  return alpha_min + (alpha_max - alpha_min) * (local_var / (local_var + var_k))


def apply_sharpen(x: torch.Tensor, sharpen_mode: str, amount: float, alpha_min: float, alpha_max: float, var_k: float) -> torch.Tensor:
  if sharpen_mode == "none":
    return x
  kernel = UNSHARP_KERNEL.to(device=x.device).expand(3, 1, 9, 9)
  blur = F.conv2d(x, kernel, padding=4, groups=3)
  detail = x - blur
  luma = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
  if sharpen_mode == "unsharp_rgb":
    return x + amount * detail
  if sharpen_mode == "adaptive_rgb":
    alpha_map = build_alpha_map(luma, alpha_min, alpha_max, var_k)
    return x + alpha_map * detail
  if sharpen_mode == "adaptive_luma":
    luma_blur = 0.299 * blur[:, 0:1] + 0.587 * blur[:, 1:2] + 0.114 * blur[:, 2:3]
    luma_detail = luma - luma_blur
    alpha_map = build_alpha_map(luma, alpha_min, alpha_max, var_k)
    return x + amount * alpha_map * luma_detail
  if sharpen_mode == "adaptive":
    alpha_map = build_alpha_map(luma, alpha_min, alpha_max, var_k)
    return x + alpha_map * detail
  if sharpen_mode == "unsharp":
    return x + amount * detail
  raise ValueError(f"unknown sharpen mode: {sharpen_mode}")


def decode_and_resize_to_file(video_path: str, dst: str):
  target_w, target_h = camera_size
  fmt = "hevc" if video_path.endswith(".hevc") else None
  sharpen_mode = os.getenv("MY_SUBMISSION_SHARPEN_MODE", os.getenv("MY_SUBMISSION_SHARPEN", "none"))
  sharpen_amount = float(os.getenv("MY_SUBMISSION_SHARPEN_AMOUNT", "0.85"))
  sharpen_min = float(os.getenv("MY_SUBMISSION_SHARPEN_MIN", "0.40"))
  sharpen_max = float(os.getenv("MY_SUBMISSION_SHARPEN_MAX", "1.20"))
  sharpen_var_k = float(os.getenv("MY_SUBMISSION_SHARPEN_VAR_K", "100.0"))
  container = av.open(video_path, format=fmt)
  stream = container.streams.video[0]
  n = 0
  with open(dst, "wb") as f:
    for frame in container.decode(stream):
      t = yuv420_to_rgb(frame)
      height, width, _ = t.shape
      if height != target_h or width != target_w:
        x = t.permute(2, 0, 1).unsqueeze(0).float()
        x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
        x = apply_sharpen(x, sharpen_mode, sharpen_amount, sharpen_min, sharpen_max, sharpen_var_k)
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

