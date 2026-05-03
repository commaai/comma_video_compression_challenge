#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import av
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from frame_utils import yuv420_to_rgb


def rgb_to_yuv(rgb: torch.Tensor) -> torch.Tensor:
  r = rgb[..., 0:1]
  g = rgb[..., 1:2]
  b = rgb[..., 2:3]
  y = 0.299 * r + 0.587 * g + 0.114 * b
  u = (b - y) / 1.772 + 128.0
  v = (r - y) / 1.402 + 128.0
  return torch.cat([y, u, v], dim=-1)


def blur_rgb(rgb: torch.Tensor, radius: int) -> torch.Tensor:
  x = rgb.permute(2, 0, 1).unsqueeze(0)
  y = F.avg_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
  return y.squeeze(0).permute(1, 2, 0)


def grad_mag(y: torch.Tensor) -> torch.Tensor:
  x = y.permute(2, 0, 1).unsqueeze(0)
  kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
  ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
  gx = F.conv2d(x, kx, padding=1)
  gy = F.conv2d(x, ky, padding=1)
  mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
  return mag.squeeze(0).permute(1, 2, 0)


def apply_middle_bps_medium(mid: torch.Tensor) -> torch.Tensor:
  y = rgb_to_yuv(mid)[..., 0:1]
  edge = torch.clamp(grad_mag(y) / 40.0, 0.0, 1.0)
  blurred = blur_rgb(mid, 5)
  keep = 1.00 * edge
  return blurred * (1.0 - keep) + mid * keep


def main() -> None:
  parser = argparse.ArgumentParser(description="Create cropped middle-band BPS-medium carrier.")
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  args.output.parent.mkdir(parents=True, exist_ok=True)

  in_container = av.open(str(args.input))
  in_stream = in_container.streams.video[0]
  width = in_stream.width
  height = in_stream.height

  top_h = height // 4
  bottom_h = height // 4
  mid_y0 = top_h
  mid_y1 = height - bottom_h
  mid_h = mid_y1 - mid_y0

  out_container = av.open(str(args.output), mode="w")
  out_stream = out_container.add_stream("ffv1", rate=20)
  out_stream.width = width
  out_stream.height = mid_h
  out_stream.pix_fmt = "yuv420p"

  for frame in in_container.decode(in_stream):
    rgb = yuv420_to_rgb(frame).float()
    mid = apply_middle_bps_medium(rgb[mid_y0:mid_y1]).clamp(0, 255).round().to(torch.uint8).numpy()
    video_frame = av.VideoFrame.from_ndarray(mid, format="rgb24")
    for packet in out_stream.encode(video_frame):
      out_container.mux(packet)

  for packet in out_stream.encode():
    out_container.mux(packet)

  out_container.close()
  in_container.close()


if __name__ == "__main__":
  main()

