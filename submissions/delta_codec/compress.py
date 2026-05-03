#!/usr/bin/env python
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import av
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from frame_utils import yuv420_to_rgb


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


def ffmpeg_bin() -> str:
  local = ROOT / "ffmpeg-new"
  if local.is_file() and local.stat().st_mode & 0o111:
    try:
      subprocess.run([str(local), "-hide_banner", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
      return str(local)
    except (OSError, subprocess.SubprocessError):
      pass
  found = shutil.which("ffmpeg")
  if found:
    return found
  raise FileNotFoundError("ffmpeg not found")


def encode_raw_rgb(raw_path: Path, out_path: Path, width: int, height: int, fps: int, encoder: str, crf: int, preset: str):
  cmd = [
    ffmpeg_bin(),
    "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
    "-f", "rawvideo",
    "-pix_fmt", "rgb24",
    "-s", f"{width}x{height}",
    "-r", str(fps),
    "-i", str(raw_path),
    "-pix_fmt", "yuv420p",
    "-c:v", encoder,
  ]
  if encoder == "libsvtav1":
    cmd += ["-preset", preset, "-crf", str(crf), "-svtav1-params", "keyint=180:scd=0"]
  elif encoder == "libx265":
    cmd += ["-preset", preset, "-crf", str(crf), "-x265-params", "log-level=warning:keyint=180:min-keyint=1:scenecut=0"]
  else:
    cmd += ["-crf", str(crf)]
  cmd += [str(out_path)]
  subprocess.run(cmd, check=True)


def downsample_pair(frame0: torch.Tensor, frame1: torch.Tensor, width: int, height: int) -> torch.Tensor:
  pair = torch.stack([frame0, frame1]).float()
  return F.interpolate(pair, size=(height, width), mode="bicubic", align_corners=False).clamp(0, 255)


def write_pair_streams(video_path: Path, base_raw: Path, delta_raw: Path, args) -> tuple[int, int, int]:
  fmt = "hevc" if video_path.suffix == ".hevc" else None
  container = av.open(str(video_path), format=fmt)
  stream = container.streams.video[0]
  pending = None
  frame_count = 0
  pair_count = 0
  skipped = 0

  with open(base_raw, "wb") as base_f, open(delta_raw, "wb") as delta_f:
    for frame in tqdm(container.decode(stream), desc=f"delta-prep {video_path.name}"):
      rgb = yuv420_to_rgb(frame).permute(2, 0, 1)
      frame_count += 1
      if pending is None:
        pending = rgb
        continue

      small = downsample_pair(pending, rgb, args.width, args.height)
      base = small[0].round().to(torch.uint8)
      residual = small[1] - small[0]
      inside_step = args.delta_step
      outside_step = args.outside_delta_step if args.outside_delta_step is not None else inside_step
      inside_deadzone = args.deadzone
      outside_deadzone = args.outside_deadzone if args.outside_deadzone is not None else inside_deadzone
      if outside_step != inside_step or outside_deadzone != inside_deadzone:
        mask = build_roi_mask(frame_count - 2, args.width, args.height, args.roi_feather)
        step_map = mask * inside_step + (1.0 - mask) * outside_step
        deadzone_map = mask * inside_deadzone + (1.0 - mask) * outside_deadzone
      else:
        step_map = inside_step
        deadzone_map = inside_deadzone
      if args.skip_threshold > 0 and residual.abs().mean().item() < args.skip_threshold:
        q = torch.zeros_like(residual, dtype=torch.int16)
        skipped += 1
      else:
        if isinstance(deadzone_map, torch.Tensor) or deadzone_map > 0:
          residual = torch.where(residual.abs() < deadzone_map, torch.zeros_like(residual), residual)
        q = torch.round(residual / step_map).clamp(-127, 127).to(torch.int16)
      delta = (q + 128).clamp(0, 255).to(torch.uint8)

      base_f.write(base.permute(1, 2, 0).contiguous().numpy().tobytes())
      delta_f.write(delta.permute(1, 2, 0).contiguous().numpy().tobytes())
      pair_count += 1
      pending = None

    if pending is not None:
      small = F.interpolate(pending.unsqueeze(0).float(), size=(args.height, args.width), mode="bicubic", align_corners=False).clamp(0, 255)
      base = small[0].round().to(torch.uint8)
      delta = torch.full_like(base, 128, dtype=torch.uint8)
      base_f.write(base.permute(1, 2, 0).contiguous().numpy().tobytes())
      delta_f.write(delta.permute(1, 2, 0).contiguous().numpy().tobytes())
      pair_count += 1
      skipped += 1

  container.close()
  return frame_count, pair_count, skipped


def zip_archive(archive_dir: Path, zip_path: Path):
  if zip_path.exists():
    zip_path.unlink()
  with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
    for path in sorted(archive_dir.iterdir()):
      zf.write(path, arcname=path.name)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--in-dir", type=Path, default=ROOT / "videos")
  p.add_argument("--video-names-file", type=Path, default=ROOT / "public_test_video_names.txt")
  p.add_argument("--width", type=int, default=256)
  p.add_argument("--height", type=int, default=192)
  p.add_argument("--fps", type=int, default=10, help="Pair stream FPS. Timing is not used by evaluator, but helps containers.")
  p.add_argument("--encoder", type=str, default="libsvtav1", choices=["libsvtav1", "libx265"])
  p.add_argument("--base-crf", type=int, default=34)
  p.add_argument("--delta-crf", type=int, default=24)
  p.add_argument("--preset", type=str, default="0")
  p.add_argument("--delta-step", type=float, default=2.0)
  p.add_argument("--deadzone", type=float, default=0.0)
  p.add_argument("--outside-delta-step", type=float, default=None, help="Use coarser residual quantization outside the driving corridor.")
  p.add_argument("--outside-deadzone", type=float, default=None, help="Zero more tiny residuals outside the driving corridor.")
  p.add_argument("--roi-feather", type=int, default=6)
  p.add_argument("--skip-threshold", type=float, default=0.0)
  args = p.parse_args()

  here = Path(__file__).resolve().parent
  archive_dir = here / "archive"
  archive_dir.mkdir(parents=True, exist_ok=True)
  for child in archive_dir.iterdir():
    child.unlink()

  names = [line.strip() for line in args.video_names_file.read_text().splitlines() if line.strip()]
  meta = {
    "width": args.width,
    "height": args.height,
    "fps": args.fps,
    "encoder": args.encoder,
    "preset": args.preset,
    "base_crf": args.base_crf,
    "delta_crf": args.delta_crf,
    "delta_step": args.delta_step,
    "deadzone": args.deadzone,
    "outside_delta_step": args.outside_delta_step if args.outside_delta_step is not None else args.delta_step,
    "outside_deadzone": args.outside_deadzone if args.outside_deadzone is not None else args.deadzone,
    "roi_feather": args.roi_feather,
    "skip_threshold": args.skip_threshold,
    "files": {},
  }

  with tempfile.TemporaryDirectory(prefix="delta_codec_") as tmp:
    tmp_dir = Path(tmp)
    for name in names:
      stem = Path(name).stem
      base_raw = tmp_dir / f"{stem}.base.rgb"
      delta_raw = tmp_dir / f"{stem}.delta.rgb"
      base_mkv = archive_dir / f"{stem}.base.mkv"
      delta_mkv = archive_dir / f"{stem}.delta.mkv"

      frame_count, pair_count, skipped = write_pair_streams(args.in_dir / name, base_raw, delta_raw, args)
      print(f"{name}: {frame_count} frames -> {pair_count} base/delta pairs, skipped={skipped}")
      encode_raw_rgb(base_raw, base_mkv, args.width, args.height, args.fps, args.encoder, args.base_crf, args.preset)
      encode_raw_rgb(delta_raw, delta_mkv, args.width, args.height, args.fps, args.encoder, args.delta_crf, args.preset)

      meta["files"][name] = {
        "frame_count": frame_count,
        "pair_count": pair_count,
        "base": base_mkv.name,
        "delta": delta_mkv.name,
        "skipped": skipped,
      }

  (archive_dir / "meta.json").write_text(json.dumps(meta, indent=2))
  zip_archive(archive_dir, here / "archive.zip")
  print(f"Compressed to {here / 'archive.zip'}")


if __name__ == "__main__":
  main()
