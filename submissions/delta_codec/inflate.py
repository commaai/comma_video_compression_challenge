#!/usr/bin/env python
import json
import sys
from pathlib import Path

import av
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from frame_utils import camera_size, yuv420_to_rgb


def segment_polygon(frame_idx: int, width: int, height: int) -> list[tuple[float, float]]:
  segments = [
    (0,   299, [(0.14, 0.52), (0.82, 0.48), (0.98, 1.00), (0.05, 1.00)]),
    (300, 599, [(0.10, 0.50), (0.76, 0.47), (0.92, 1.00), (0.00, 1.00)]),
    (600, 899, [(0.18, 0.50), (0.84, 0.47), (0.98, 1.00), (0.06, 1.00)]),
    (900, 1199, [(0.22, 0.52), (0.90, 0.49), (1.00, 1.00), (0.10, 1.00)]),
  ]
  for start, end, poly in segments:
    if start <= frame_idx <= end:
      return [(x * width, y * height) for x, y in poly]
  return [(0.15 * width, 0.52 * height), (0.85 * width, 0.48 * height), (width, height), (0, height)]


def build_roi_mask(frame_idx: int, width: int, height: int, feather_radius: int) -> torch.Tensor:
  img = Image.new("L", (width, height), 0)
  draw = ImageDraw.Draw(img)
  draw.polygon(segment_polygon(frame_idx, width, height), fill=255)
  if feather_radius > 0:
    img = img.filter(ImageFilter.GaussianBlur(radius=feather_radius))
  mask = torch.frombuffer(memoryview(img.tobytes()), dtype=torch.uint8).clone().view(height, width).float() / 255.0
  return mask.unsqueeze(0)


def decode_iter(path: Path):
  container = av.open(str(path))
  stream = container.streams.video[0]
  try:
    for frame in container.decode(stream):
      yield yuv420_to_rgb(frame).permute(2, 0, 1).float()
  finally:
    container.close()


@torch.inference_mode()
def main():
  if len(sys.argv) != 4:
    print("Usage: python inflate.py <data_dir> <output_dir> <file_list_txt>")
    sys.exit(2)

  data_dir = Path(sys.argv[1])
  out_dir = Path(sys.argv[2])
  file_list = Path(sys.argv[3])
  out_dir.mkdir(parents=True, exist_ok=True)
  meta = json.loads((data_dir / "meta.json").read_text())
  target_w, target_h = camera_size

  names = [line.strip() for line in file_list.read_text().splitlines() if line.strip()]
  for name in names:
    info = meta["files"][name]
    dst = out_dir / f"{Path(name).stem}.raw"
    base_frames = decode_iter(data_dir / info["base"])
    delta_frames = decode_iter(data_dir / info["delta"])
    written = 0
    with open(dst, "wb") as f:
      pbar = tqdm(total=info["pair_count"], desc=f"inflating {name}")
      for pair_idx, (base, delta_img) in enumerate(zip(base_frames, delta_frames)):
        q = delta_img - 128.0
        inside_step = float(meta["delta_step"])
        outside_step = float(meta.get("outside_delta_step", inside_step))
        if outside_step != inside_step:
          mask = build_roi_mask(pair_idx * 2, base.shape[-1], base.shape[-2], int(meta.get("roi_feather", 6)))
          step_map = mask * inside_step + (1.0 - mask) * outside_step
          residual = q * step_map
        else:
          residual = q * inside_step
        frame0 = base
        frame1 = (base + residual).clamp(0, 255)
        frames = torch.stack([frame0, frame1])
        frames = F.interpolate(frames, size=(target_h, target_w), mode="bicubic", align_corners=False)
        frames = frames.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for frame in frames:
          if written >= info["frame_count"]:
            break
          f.write(frame.tobytes())
          written += 1
        pbar.update(1)
      pbar.close()
    if written != info["frame_count"]:
      raise RuntimeError(f"{name}: wrote {written} frames, expected {info['frame_count']}")
    print(f"saved {dst}")


if __name__ == "__main__":
  main()
