#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import av
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from frame_utils import yuv420_to_rgb


def collapse_chroma(x: torch.Tensor, mode: str) -> torch.Tensor:
  if mode == "normal":
    return x
  if mode == "soft":
    k = 1
  elif mode == "medium":
    k = 2
  elif mode == "strong":
    k = 4
  else:
    raise ValueError(f"unknown chroma mode: {mode}")
  uv = x[:, 1:3]
  uv = F.avg_pool2d(uv, kernel_size=k * 2 + 1, stride=1, padding=k)
  x[:, 1:3] = uv
  return x


def apply_luma_denoise(x: torch.Tensor, strength: float) -> torch.Tensor:
  if strength <= 0:
    return x
  kernel_size = 3 if strength <= 2.0 else 5
  sigma = max(0.1, strength * 0.35)
  coords = torch.arange(kernel_size, device=x.device) - kernel_size // 2
  g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
  kernel_1d = (g / g.sum()).float()
  kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, kernel_size, kernel_size)
  y = x[:, 0:1]
  y_blur = F.conv2d(y, kernel_2d, padding=kernel_size // 2)
  blend = min(0.9, strength / 3.0)
  x[:, 0:1] = (1 - blend) * y + blend * y_blur
  return x


def rgb_to_yuv(rgb: torch.Tensor) -> torch.Tensor:
  r = rgb[:, 0:1]
  g = rgb[:, 1:2]
  b = rgb[:, 2:3]
  y = 0.299 * r + 0.587 * g + 0.114 * b
  u = (b - y) / 1.772 + 128.0
  v = (r - y) / 1.402 + 128.0
  return torch.cat([y, u, v], dim=1)


def yuv_to_rgb(yuv: torch.Tensor) -> torch.Tensor:
  y = yuv[:, 0:1]
  u = yuv[:, 1:2] - 128.0
  v = yuv[:, 2:3] - 128.0
  r = y + 1.402 * v
  g = y - 0.344136 * u - 0.714136 * v
  b = y + 1.772 * u
  return torch.cat([r, g, b], dim=1)


def segment_polygon(frame_idx: int, width: int, height: int) -> list[tuple[float, float]]:
  segments = [
    (0, 299, [(0.14, 0.52), (0.82, 0.48), (0.98, 1.00), (0.05, 1.00)]),
    (300, 599, [(0.10, 0.50), (0.76, 0.47), (0.92, 1.00), (0.00, 1.00)]),
    (600, 899, [(0.18, 0.50), (0.84, 0.47), (0.98, 1.00), (0.06, 1.00)]),
    (900, 1199, [(0.22, 0.52), (0.90, 0.49), (1.00, 1.00), (0.10, 1.00)]),
  ]
  for start, end, poly in segments:
    if start <= frame_idx <= end:
      return [(x * width, y * height) for x, y in poly]
  return [(0.15 * width, 0.52 * height), (0.85 * width, 0.48 * height), (width, height), (0, height)]


def build_mask(frame_idx: int, width: int, height: int, feather_radius: int) -> torch.Tensor:
  img = Image.new("L", (width, height), 0)
  draw = ImageDraw.Draw(img)
  draw.polygon(segment_polygon(frame_idx, width, height), fill=255)
  if feather_radius > 0:
    img = img.filter(ImageFilter.GaussianBlur(radius=feather_radius))
  mask = torch.frombuffer(memoryview(img.tobytes()), dtype=torch.uint8).clone().view(height, width).float() / 255.0
  return mask.unsqueeze(0).unsqueeze(0)


def process_frame(
  frame_rgb: torch.Tensor,
  frame_idx: int,
  outside_luma_denoise: float,
  outside_chroma_mode: str,
  feather_radius: int,
  outside_blend: float,
) -> torch.Tensor:
  chw = frame_rgb.permute(2, 0, 1).float().unsqueeze(0)
  mask = build_mask(frame_idx, chw.shape[-1], chw.shape[-2], feather_radius).to(chw.device)
  yuv = rgb_to_yuv(chw)
  processed = yuv.clone()
  processed = apply_luma_denoise(processed, outside_luma_denoise)
  processed = collapse_chroma(processed, outside_chroma_mode)
  processed_rgb = yuv_to_rgb(processed)
  outside_alpha = (1.0 - mask) * outside_blend
  mixed = chw * (1.0 - outside_alpha) + processed_rgb * outside_alpha
  return mixed.clamp(0, 255).round().to(torch.uint8).squeeze(0).permute(1, 2, 0)


def main() -> None:
  parser = argparse.ArgumentParser(description="Hand-authored ROI preprocessor for AV1 encode.")
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--outside-luma-denoise", type=float, default=0.0)
  parser.add_argument("--outside-chroma-mode", type=str, default="normal")
  parser.add_argument("--feather-radius", type=int, default=32)
  parser.add_argument("--outside-blend", type=float, default=1.0)
  args = parser.parse_args()

  in_container = av.open(str(args.input))
  in_stream = in_container.streams.video[0]
  width = in_stream.width
  height = in_stream.height

  out_container = av.open(str(args.output), mode="w")
  out_stream = out_container.add_stream("ffv1", rate=20)
  out_stream.width = width
  out_stream.height = height
  out_stream.pix_fmt = "yuv420p"

  for frame_idx, frame in enumerate(in_container.decode(in_stream)):
    rgb = yuv420_to_rgb(frame)
    out_rgb = process_frame(
      rgb,
      frame_idx=frame_idx,
      outside_luma_denoise=args.outside_luma_denoise,
      outside_chroma_mode=args.outside_chroma_mode,
      feather_radius=args.feather_radius,
      outside_blend=args.outside_blend,
    )
    video_frame = av.VideoFrame.from_ndarray(out_rgb.cpu().numpy(), format="rgb24")
    for packet in out_stream.encode(video_frame):
      out_container.mux(packet)

  for packet in out_stream.encode():
    out_container.mux(packet)

  out_container.close()
  in_container.close()


if __name__ == "__main__":
  main()
