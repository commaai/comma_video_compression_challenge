#!/usr/bin/env python
import os
import struct
import sys
from itertools import zip_longest
from pathlib import Path

import av
import torch
import torch.nn.functional as F

from frame_utils import camera_size, segnet_model_input_size, yuv420_to_rgb


UPSCALE_MODE = os.environ.get("CODEX_UPSCALE_MODE", "bicubic")
CAMERA_UPSCALE_MODE = os.environ.get("CODEX_CAMERA_UPSCALE_MODE", UPSCALE_MODE)
INVERSE_ITERS = int(os.environ.get("CODEX_INVERSE_ITERS", "2"))
INVERSE_GAIN = float(os.environ.get("CODEX_INVERSE_GAIN", "1.0"))
INVERSE_BASE_MODE = os.environ.get("CODEX_INVERSE_BASE_MODE", "bicubic")
EDGE_BASE_MODE = os.environ.get("CODEX_EDGE_BASE_MODE", "bicubic")
EDGE_AMOUNT = float(os.environ.get("CODEX_EDGE_AMOUNT", "0.12"))
EDGE_LOW = float(os.environ.get("CODEX_EDGE_LOW", "2.0"))
EDGE_HIGH = float(os.environ.get("CODEX_EDGE_HIGH", "16.0"))
EDGE_KERNEL = int(os.environ.get("CODEX_EDGE_KERNEL", "3"))
EDGE_FLAT_BLUR = float(os.environ.get("CODEX_EDGE_FLAT_BLUR", "0.0"))
POST_UNSHARP_AMOUNT = float(os.environ.get("CODEX_POST_UNSHARP_AMOUNT", "0.0"))
POST_UNSHARP_KERNEL = int(os.environ.get("CODEX_POST_UNSHARP_KERNEL", "3"))
POST_UNSHARP_THRESHOLD = float(os.environ.get("CODEX_POST_UNSHARP_THRESHOLD", "0.0"))
POST_BLUR_BLEND = float(os.environ.get("CODEX_POST_BLUR_BLEND", "0.0"))
POST_SATURATION = os.environ.get("CODEX_POST_SATURATION", "1.0")
POST_LUMA_GAIN = os.environ.get("CODEX_POST_LUMA_GAIN", "1.0")
POST_LUMA_BIAS = os.environ.get("CODEX_POST_LUMA_BIAS", "0.0")
POST_RGB_GAIN = os.environ.get("CODEX_POST_RGB_GAIN", "1.0,1.0,1.0")
POST_RGB_BIAS = os.environ.get("CODEX_POST_RGB_BIAS", "0.0,0.0,0.0")
POST_SCHEDULE_FRAMES = int(os.environ.get("CODEX_POST_SCHEDULE_FRAMES", "10"))
POSTPROCESS_SEGMENTS = {
  int(item)
  for item in os.environ.get("CODEX_POSTPROCESS_SEGMENTS", "").replace(",", " ").split()
  if item.strip()
}
TEMPORAL_BLEND = float(os.environ.get("CODEX_TEMPORAL_BLEND", "0.025"))
TEMPORAL_MODE = os.environ.get("CODEX_TEMPORAL_MODE", "fixed")
TEMPORAL_DIFF_LOW = float(os.environ.get("CODEX_TEMPORAL_DIFF_LOW", "4.0"))
TEMPORAL_DIFF_HIGH = float(os.environ.get("CODEX_TEMPORAL_DIFF_HIGH", "18.0"))
TEMPORAL_MIN_FACTOR = float(os.environ.get("CODEX_TEMPORAL_MIN_FACTOR", "0.0"))
TEMPORAL_MAX_FACTOR = float(os.environ.get("CODEX_TEMPORAL_MAX_FACTOR", "1.0"))
TEMPORAL_KERNEL = int(os.environ.get("CODEX_TEMPORAL_KERNEL", "5"))
TEMPORAL_MOTION_RADIUS = int(os.environ.get("CODEX_TEMPORAL_MOTION_RADIUS", "2"))
TEMPORAL_MOTION_DOWNSAMPLE = int(os.environ.get("CODEX_TEMPORAL_MOTION_DOWNSAMPLE", "2"))
TEMPORAL_SCHEDULE_FRAMES = int(os.environ.get("CODEX_TEMPORAL_SCHEDULE_FRAMES", "10"))
SIDECHANNEL_FEEDBACK = os.environ.get("CODEX_SIDECHANNEL_FEEDBACK", "0") in {"1", "true", "yes", "on"}
POSTFILTER_DEVICE = os.environ.get("CODEX_POSTFILTER_DEVICE", "auto")
POSTFILTER_SIZE = os.environ.get("CODEX_POSTFILTER_SIZE", "")
POSTFILTER_RESIZE_MODE = os.environ.get("CODEX_POSTFILTER_RESIZE_MODE", "bilinear")
POSTFILTER_PATH = os.environ.get("CODEX_POSTFILTER_PATH", "")
PAIR_ASYM_MODE_ENV = os.environ.get("CODEX_PAIR_ASYM_MODE")
PAIR_ASYM_BLEND = float(os.environ.get("CODEX_PAIR_ASYM_BLEND", "0.5"))
SIDECHANNEL_MAGIC = b"SC01"
SIDECHANNEL_HEADER = struct.Struct("<4sBBIf")
SIDECHANNEL_MODE_Y_SAT = 6
SIDECHANNEL_MODE_Y_SHIFT = 7
SIDECHANNEL_SAT_RAW_STEP = 0.01
LATENT_LUMA_MAGIC = b"LRL1"
LATENT_LUMA_HEADER = struct.Struct("<4sBBHHHff")


def interpolate_image(x: torch.Tensor, height: int, width: int, mode: str) -> torch.Tensor:
  if mode == "nearest":
    return F.interpolate(x, size=(height, width), mode=mode)
  return F.interpolate(x, size=(height, width), mode=mode, align_corners=False)


def choose_device(name: str) -> torch.device:
  if name != "auto":
    return torch.device(name)
  if torch.cuda.is_available():
    return torch.device("cuda", 0)
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def load_manifest_settings(segment_dir: Path) -> dict[str, str]:
  path = segment_dir / "manifest.tsv"
  if not path.exists():
    return {}
  settings = {}
  for line in path.read_text().splitlines():
    parts = line.split("\t")
    if len(parts) >= 2 and parts[0] != "segment":
      settings[parts[0]] = parts[1]
  return settings


def pair_asym_mode(segment_dir: Path) -> str:
  if PAIR_ASYM_MODE_ENV:
    return PAIR_ASYM_MODE_ENV
  return load_manifest_settings(segment_dir).get("pair_asym_mode", "none")


def parse_size(text: str, default_width: int, default_height: int) -> tuple[int, int]:
  if not text:
    return default_width, default_height
  normalized = text.lower().replace("x", ",")
  parts = [part.strip() for part in normalized.split(",") if part.strip()]
  if len(parts) != 2:
    raise SystemExit(f"expected WIDTHxHEIGHT size, got: {text}")
  return max(1, int(parts[0])), max(1, int(parts[1]))


def evaluator_inverse_resize(frame: torch.Tensor, width: int, height: int) -> torch.Tensor:
  eval_w, eval_h = segnet_model_input_size
  x = frame.permute(2, 0, 1).unsqueeze(0).float()
  low = interpolate_image(x, eval_h, eval_w, UPSCALE_MODE)
  full = interpolate_image(low, height, width, INVERSE_BASE_MODE)
  for _ in range(max(0, INVERSE_ITERS)):
    observed = interpolate_image(full, eval_h, eval_w, "bilinear")
    correction = interpolate_image(observed - low, height, width, INVERSE_BASE_MODE)
    full = full - INVERSE_GAIN * correction
  return full.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)


def edge_mask_nchw(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
  coeffs = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
  y = (x * coeffs).sum(dim=1, keepdim=True)
  padded_x = F.pad(y, (1, 1, 0, 0), mode="reflect")
  padded_y = F.pad(y, (0, 0, 1, 1), mode="reflect")
  gx = (padded_x[:, :, :, 2:] - padded_x[:, :, :, :-2]).abs() * 0.5
  gy = (padded_y[:, :, 2:, :] - padded_y[:, :, :-2, :]).abs() * 0.5
  edge = gx + gy
  span = max(1e-6, high - low)
  return ((edge - low) / span).clamp(0.0, 1.0)


def edge_aware_resize(frame: torch.Tensor, width: int, height: int) -> torch.Tensor:
  x = frame.permute(2, 0, 1).unsqueeze(0).float()
  base = interpolate_image(x, height, width, EDGE_BASE_MODE)
  blurred = blur_nchw(base, EDGE_KERNEL)
  detail = base - blurred
  mask = edge_mask_nchw(base, EDGE_LOW, EDGE_HIGH)
  out = base + EDGE_AMOUNT * detail * mask
  if EDGE_FLAT_BLUR != 0.0:
    flat_weight = EDGE_FLAT_BLUR * (1.0 - mask)
    out = out * (1.0 - flat_weight) + blurred * flat_weight
  return out.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)


def camera_upscale_mode(segment_idx: int) -> str:
  segment_key = f"CODEX_CAMERA_UPSCALE_SEGMENT{segment_idx}_MODE"
  if segment_key in os.environ:
    return os.environ[segment_key]
  if segment_idx >= 1 and "CODEX_CAMERA_UPSCALE_MODE" not in os.environ:
    return "eval-inverse"
  return os.environ.get(f"CODEX_CAMERA_UPSCALE_SEGMENT{segment_idx}_MODE", CAMERA_UPSCALE_MODE)


def resize_rgb(frame: torch.Tensor, width: int, height: int, segment_idx: int) -> torch.Tensor:
  mode = camera_upscale_mode(segment_idx)
  if mode in {"eval-inverse", "eval_inverse", "evaluator-inverse", "evaluator_inverse"}:
    return evaluator_inverse_resize(frame, width, height)
  if mode in {"edge", "edge-aware", "edge_aware", "edge-sharp", "edge_sharp"}:
    return edge_aware_resize(frame, width, height)
  x = frame.permute(2, 0, 1).unsqueeze(0).float()
  x = interpolate_image(x, height, width, mode)
  return x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)


def blur_nchw(x: torch.Tensor, kernel: int) -> torch.Tensor:
  kernel = max(1, kernel | 1)
  if kernel == 1:
    return x
  pad = kernel // 2
  padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
  return F.avg_pool2d(padded, kernel_size=kernel, stride=1)


def luma_plane(rgb: torch.Tensor) -> torch.Tensor:
  coeffs = torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype, device=rgb.device)
  return (rgb * coeffs).sum(dim=2)


def luma_plane_correction(height: int, width: int, values: torch.Tensor) -> torch.Tensor:
  y = torch.linspace(-1.0, 1.0, height, dtype=values.dtype, device=values.device).view(height, 1)
  x = torch.linspace(-1.0, 1.0, width, dtype=values.dtype, device=values.device).view(1, width)
  if values.numel() == 3:
    return values[0] + values[1] * x + values[2] * y
  if values.numel() == 6:
    x2 = x * x - (1.0 / 3.0)
    y2 = y * y - (1.0 / 3.0)
    return values[0] + values[1] * x + values[2] * y + values[3] * (x * y) + values[4] * x2 + values[5] * y2
  raise SystemExit("luma plane sidechannel requires 3 or 6 channels")


def grid_shape_for_channels(channels: int) -> tuple[int, int]:
  known = {
    6: (2, 3),
    9: (3, 3),
    12: (3, 4),
    16: (4, 4),
    24: (4, 6),
    48: (6, 8),
    96: (8, 12),
    192: (12, 16),
  }
  if channels in known:
    return known[channels]
  side = int(channels ** 0.5)
  if side * side == channels:
    return side, side
  raise SystemExit(f"unsupported luma grid channel count: {channels}")


def downsample_plane(plane: torch.Tensor, factor: int) -> torch.Tensor:
  factor = max(1, factor)
  if factor == 1:
    return plane
  h, w = plane.shape
  h = h - (h % factor)
  w = w - (w % factor)
  if h <= 0 or w <= 0:
    return plane
  x = plane[:h, :w].unsqueeze(0).unsqueeze(0)
  return F.avg_pool2d(x, kernel_size=factor, stride=factor).squeeze(0).squeeze(0)


def shifted_error(current: torch.Tensor, previous: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
  h, w = current.shape
  src_y0 = max(0, -dy)
  src_y1 = min(h, h - dy)
  src_x0 = max(0, -dx)
  src_x1 = min(w, w - dx)
  dst_y0 = max(0, dy)
  dst_y1 = min(h, h + dy)
  dst_x0 = max(0, dx)
  dst_x1 = min(w, w + dx)
  if src_y1 <= src_y0 or src_x1 <= src_x0:
    return torch.tensor(float("inf"), dtype=current.dtype, device=current.device)
  return (current[dst_y0:dst_y1, dst_x0:dst_x1] - previous[src_y0:src_y1, src_x0:src_x1]).abs().mean()


def estimate_global_shift(current: torch.Tensor, previous: torch.Tensor, radius: int, downsample: int) -> tuple[int, int]:
  radius = max(0, radius)
  if radius == 0:
    return 0, 0

  factor = max(1, downsample)
  cur_luma = downsample_plane(luma_plane(current), factor)
  prev_luma = downsample_plane(luma_plane(previous), factor)
  low_radius = max(1, (radius + factor - 1) // factor)
  best_error = None
  best_shift = (0, 0)
  for low_dy in range(-low_radius, low_radius + 1):
    for low_dx in range(-low_radius, low_radius + 1):
      dy = low_dy * factor
      dx = low_dx * factor
      if abs(dy) > radius or abs(dx) > radius:
        continue
      err = shifted_error(cur_luma, prev_luma, low_dy, low_dx)
      if best_error is None or err.item() < best_error:
        best_error = err.item()
        best_shift = (dy, dx)
  return best_shift


def shift_rgb(previous: torch.Tensor, fallback: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
  if dy == 0 and dx == 0:
    return previous
  h, w, _ = previous.shape
  src_y0 = max(0, -dy)
  src_y1 = min(h, h - dy)
  src_x0 = max(0, -dx)
  src_x1 = min(w, w - dx)
  dst_y0 = max(0, dy)
  dst_y1 = min(h, h + dy)
  dst_x0 = max(0, dx)
  dst_x1 = min(w, w + dx)
  out = fallback.clone()
  if src_y1 > src_y0 and src_x1 > src_x0:
    out[dst_y0:dst_y1, dst_x0:dst_x1] = previous[src_y0:src_y1, src_x0:src_x1]
  return out


def reconstruct_pair_even_frame(current_odd: torch.Tensor, previous_odd: torch.Tensor | None, mode: str) -> torch.Tensor:
  normalized = mode.replace("_", "-")
  if normalized == "odd-duplicate" or previous_odd is None:
    return current_odd
  if normalized == "odd-prevblend":
    blend = max(0.0, min(1.0, PAIR_ASYM_BLEND))
    out = current_odd.float() * (1.0 - blend) + previous_odd.float() * blend
    return out.clamp(0, 255).round().to(torch.uint8)
  if normalized == "odd-motion":
    dy, dx = estimate_global_shift(
      current_odd.float(),
      previous_odd.float(),
      TEMPORAL_MOTION_RADIUS,
      TEMPORAL_MOTION_DOWNSAMPLE,
    )
    shifted_prev = shift_rgb(previous_odd, current_odd, int(round(dy * PAIR_ASYM_BLEND)), int(round(dx * PAIR_ASYM_BLEND)))
    out = current_odd.float() * (1.0 - PAIR_ASYM_BLEND) + shifted_prev.float() * PAIR_ASYM_BLEND
    return out.clamp(0, 255).round().to(torch.uint8)
  raise SystemExit(f"unknown pair asymmetry mode: {mode}")


def postprocess_rgb(frame: torch.Tensor, segment_idx: int) -> torch.Tensor:
  if POSTPROCESS_SEGMENTS and segment_idx not in POSTPROCESS_SEGMENTS:
    return frame
  if POST_UNSHARP_AMOUNT == 0.0 and POST_BLUR_BLEND == 0.0:
    return frame

  x = frame.permute(2, 0, 1).unsqueeze(0).float()
  kernel = POST_UNSHARP_KERNEL
  blurred = blur_nchw(x, kernel)

  if POST_BLUR_BLEND != 0.0:
    x = x * (1.0 - POST_BLUR_BLEND) + blurred * POST_BLUR_BLEND

  if POST_UNSHARP_AMOUNT != 0.0:
    detail = x - blurred
    if POST_UNSHARP_THRESHOLD > 0.0:
      detail = torch.where(detail.abs() >= POST_UNSHARP_THRESHOLD, detail, torch.zeros_like(detail))
    x = x + POST_UNSHARP_AMOUNT * detail

  return x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)


def segment_env(segment_idx: int, suffix: str, default: str, final_segment_idx: int | None = None) -> str:
  if suffix == "BLEND" and segment_idx >= 1 and "CODEX_TEMPORAL_BLEND" not in os.environ:
    is_final_late_segment = final_segment_idx is not None and segment_idx == final_segment_idx and segment_idx >= 2
    if segment_idx == 1:
      default = "0.025"
    elif segment_idx in {2, 3} or is_final_late_segment:
      default = "0.020"
    else:
      default = "0.015"
  return os.environ.get(f"CODEX_TEMPORAL_SEGMENT{segment_idx}_{suffix}", default)


def scheduled_segment_env(
  segment_idx: int,
  frame_in_segment: int,
  suffix: str,
  default: str,
  final_segment_idx: int | None = None,
) -> str:
  schedule_key = f"CODEX_TEMPORAL_SEGMENT{segment_idx}_{suffix}_SCHEDULE"
  values = None
  if schedule_key in os.environ:
    values = [item.strip() for item in os.environ[schedule_key].replace(";", ",").split(",") if item.strip()]
  else:
    global_schedule_key = f"CODEX_TEMPORAL_{suffix}_SCHEDULE"
    if global_schedule_key in os.environ:
      values = [item.strip() for item in os.environ[global_schedule_key].replace(";", ",").split(",") if item.strip()]
    elif suffix == "BLEND":
      if segment_idx == 1:
        values = [
          "0.020",
          "0.020",
          "0.020",
          "0.020",
          "0.025",
          "0.025",
          "0.025",
          "0.025",
          "0.030",
          "0.030",
          "0.030",
          "0.030",
        ]
      elif segment_idx == 2:
        values = [
          "0.020",
          "0.020",
          "0.020",
          "0.020",
          "0.020",
          "0.020",
          "0.025",
          "0.025",
          "0.025",
          "0.020",
          "0.015",
          "0.010",
        ]
      elif segment_idx == 3:
        values = [
          "0.012",
          "0.015",
          "0.015",
          "0.017",
          "0.020",
          "0.020",
          "0.022",
          "0.022",
          "0.025",
          "0.025",
          "0.027",
          "0.027",
        ]
      elif segment_idx == 4:
        values = [
          "0.000",
          "0.000",
          "0.005",
          "0.010",
          "0.015",
          "0.015",
          "0.015",
          "0.020",
          "0.020",
          "0.020",
          "0.020",
          "0.018",
        ]
      elif segment_idx == 5:
        values = [
          "0.025",
          "0.025",
          "0.025",
          "0.025",
          "0.025",
          "0.025",
          "0.020",
          "0.020",
          "0.020",
          "0.020",
          "0.020",
          "0.020",
          "0.018",
          "0.018",
          "0.018",
          "0.018",
          "0.018",
          "0.018",
          "0.015",
          "0.015",
          "0.015",
          "0.015",
          "0.015",
          "0.015",
        ]
  if values is not None:
    if not values:
      return default
    schedule_frames = max(1, TEMPORAL_SCHEDULE_FRAMES)
    value_idx = min(len(values) - 1, frame_in_segment // schedule_frames)
    return values[value_idx]
  return segment_env(segment_idx, suffix, default, final_segment_idx)


def post_env(segment_idx: int, suffix: str, default: str) -> str:
  key = f"CODEX_POST_SEGMENT{segment_idx}_{suffix}"
  if key in os.environ:
    return os.environ[key]
  if suffix == "LUMA_BIAS":
    if segment_idx == 1:
      return "5.0"
    if segment_idx == 2:
      return "-1.0"
    if segment_idx == 4:
      return "11.0"
  if suffix == "SATURATION" and segment_idx == 5:
    return "1.06"
  return default


def scheduled_post_env(segment_idx: int, frame_in_segment: int, suffix: str, default: str) -> str:
  schedule_key = f"CODEX_POST_SEGMENT{segment_idx}_{suffix}_SCHEDULE"
  values = None
  if schedule_key in os.environ:
    values = [item.strip() for item in os.environ[schedule_key].replace(";", ",").split(",") if item.strip()]
  else:
    global_schedule_key = f"CODEX_POST_{suffix}_SCHEDULE"
    if global_schedule_key in os.environ:
      values = [item.strip() for item in os.environ[global_schedule_key].replace(";", ",").split(",") if item.strip()]
    elif suffix == "LUMA_BIAS":
      if segment_idx == 2:
        values = ["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-0.5", "-0.5", "-1.5", "-2.5", "-0.5", "-0.5"]
      elif segment_idx == 4:
        values = ["15.0", "15.0", "15.0", "15.0", "15.0", "15.0", "1.0", "-1.0", "-2.0", "-2.0", "-3.0", "-3.0"]
  if values is not None:
    if not values:
      return default
    schedule_frames = max(1, POST_SCHEDULE_FRAMES)
    value_idx = min(len(values) - 1, frame_in_segment // schedule_frames)
    return values[value_idx]
  return post_env(segment_idx, suffix, default)


def parse_rgb_triplet(text: str) -> tuple[float, float, float]:
  parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
  if len(parts) == 1:
    value = float(parts[0])
    return value, value, value
  if len(parts) != 3:
    raise SystemExit(f"expected scalar or r,g,b triplet, got: {text}")
  return float(parts[0]), float(parts[1]), float(parts[2])


def color_correct_rgb(frame: torch.Tensor, segment_idx: int, frame_in_segment: int) -> torch.Tensor:
  saturation = float(scheduled_post_env(segment_idx, frame_in_segment, "SATURATION", POST_SATURATION))
  luma_gain = float(scheduled_post_env(segment_idx, frame_in_segment, "LUMA_GAIN", POST_LUMA_GAIN))
  luma_bias = float(scheduled_post_env(segment_idx, frame_in_segment, "LUMA_BIAS", POST_LUMA_BIAS))
  rgb_gain = parse_rgb_triplet(scheduled_post_env(segment_idx, frame_in_segment, "RGB_GAIN", POST_RGB_GAIN))
  rgb_bias = parse_rgb_triplet(scheduled_post_env(segment_idx, frame_in_segment, "RGB_BIAS", POST_RGB_BIAS))
  if (
    saturation == 1.0
    and luma_gain == 1.0
    and luma_bias == 0.0
    and rgb_gain == (1.0, 1.0, 1.0)
    and rgb_bias == (0.0, 0.0, 0.0)
  ):
    return frame

  x = frame.float()
  y = luma_plane(x).unsqueeze(2)
  if saturation != 1.0:
    x = y + saturation * (x - y)
  if luma_gain != 1.0 or luma_bias != 0.0:
    adjusted_y = y * luma_gain + luma_bias
    x = x + (adjusted_y - y)
    y = adjusted_y
  gain = torch.tensor(rgb_gain, dtype=x.dtype, device=x.device)
  bias = torch.tensor(rgb_bias, dtype=x.dtype, device=x.device)
  x = x * gain + bias
  return x.clamp(0, 255).round().to(torch.uint8)


def load_sidechannel(segment_dir: Path):
  path = segment_dir / "sidechannel.bin"
  if not path.exists():
    return None
  data = path.read_bytes()
  if len(data) < SIDECHANNEL_HEADER.size:
    raise SystemExit(f"sidechannel too short: {path}")
  magic, mode_id, channels, frame_count, step = SIDECHANNEL_HEADER.unpack_from(data)
  if magic != SIDECHANNEL_MAGIC:
    raise SystemExit(f"bad sidechannel magic: {path}")
  expected = SIDECHANNEL_HEADER.size + frame_count * channels
  if len(data) != expected:
    raise SystemExit(f"bad sidechannel length: {path}: expected {expected}, got {len(data)}")
  raw = torch.tensor(list(data[SIDECHANNEL_HEADER.size:]), dtype=torch.int16)
  raw = torch.where(raw >= 128, raw - 256, raw).float()
  values = raw.reshape(frame_count, channels) * float(step)
  return {"mode_id": mode_id, "channels": channels, "step": float(step), "raw": raw.reshape(frame_count, channels), "values": values}


def apply_sidechannel_rgb(frame: torch.Tensor, sidechannel, frame_idx: int) -> torch.Tensor:
  if sidechannel is None or frame_idx >= sidechannel["values"].shape[0]:
    return frame

  mode_id = sidechannel["mode_id"]
  values = sidechannel["values"][frame_idx].to(device=frame.device)
  x = frame.float()

  if mode_id == 1:
    x = x + values[0]
  elif mode_id == 2:
    if values.numel() != 3:
      raise SystemExit("rgb sidechannel requires 3 channels")
    x = x + values.view(1, 1, 3)
  elif mode_id == 3:
    h = x.shape[0]
    bands = values.numel()
    correction = torch.empty((h,), dtype=x.dtype, device=x.device)
    for band in range(bands):
      y0 = (h * band) // bands
      y1 = (h * (band + 1)) // bands
      correction[y0:y1] = values[band]
    x = x + correction.view(h, 1, 1)
  elif mode_id == 4:
    h, w, _ = x.shape
    correction = luma_plane_correction(h, w, values)
    x = x + correction.view(h, w, 1)
  elif mode_id == 5:
    h, w, _ = x.shape
    rows, cols = grid_shape_for_channels(values.numel())
    grid = values.view(1, 1, rows, cols)
    correction = F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False).view(h, w)
    x = x + correction.view(h, w, 1)
  elif mode_id == SIDECHANNEL_MODE_Y_SAT:
    raw = sidechannel["raw"][frame_idx].to(device=frame.device)
    if raw.numel() != 2:
      raise SystemExit("y+sat sidechannel requires 2 channels")
    x = x + raw[0] * float(sidechannel["step"])
    sat = 1.0 + raw[1] * SIDECHANNEL_SAT_RAW_STEP
    y = luma_plane(x).unsqueeze(2)
    x = y + sat * (x - y)
  elif mode_id == SIDECHANNEL_MODE_Y_SHIFT:
    raw = sidechannel["raw"][frame_idx].to(device=frame.device)
    if raw.numel() != 3:
      raise SystemExit("y+shift sidechannel requires 3 channels")
    x = x + raw[0] * float(sidechannel["step"])
    dy = int(round(float(raw[1].item())))
    dx = int(round(float(raw[2].item())))
    x = shift_rgb(x, x, dy, dx).float()
  else:
    raise SystemExit(f"unsupported sidechannel mode id: {mode_id}")

  return x.clamp(0, 255).round().to(torch.uint8)


def load_postfilter_bundle(segment_dir: Path):
  path = Path(POSTFILTER_PATH) if POSTFILTER_PATH else segment_dir / "postfilter.pt"
  if not path.exists():
    return None

  try:
    from .tiny_postfilter import load_postfilter
  except ImportError:
    from tiny_postfilter import load_postfilter

  device = choose_device(POSTFILTER_DEVICE)
  model, meta = load_postfilter(path, device)
  low_width = int(meta.get("low_width", 512))
  low_height = int(meta.get("low_height", 384))
  low_width, low_height = parse_size(POSTFILTER_SIZE, low_width, low_height)
  return {
    "model": model,
    "device": device,
    "low_width": low_width,
    "low_height": low_height,
  }


def load_latent_luma_bundle(segment_dir: Path):
  path = segment_dir / "latent_luma.bin"
  if not path.exists():
    return None
  data = path.read_bytes()
  if len(data) < LATENT_LUMA_HEADER.size:
    raise SystemExit(f"latent luma file too short: {path}")
  magic, components, mode_id, width, height, frame_count, coeff_step, basis_step = LATENT_LUMA_HEADER.unpack_from(data)
  if magic != LATENT_LUMA_MAGIC:
    raise SystemExit(f"bad latent luma magic: {path}")
  if mode_id != 1:
    raise SystemExit(f"unsupported latent luma mode id: {mode_id}")
  basis_count = int(components) * int(height) * int(width)
  coeff_count = int(frame_count) * int(components)
  expected = LATENT_LUMA_HEADER.size + basis_count + coeff_count
  if len(data) != expected:
    raise SystemExit(f"bad latent luma length: {path}: expected {expected}, got {len(data)}")
  basis_start = LATENT_LUMA_HEADER.size
  coeff_start = basis_start + basis_count
  basis_raw = torch.frombuffer(
    bytearray(data[basis_start:coeff_start]),
    dtype=torch.int8,
  ).float()
  coeff_raw = torch.frombuffer(
    bytearray(data[coeff_start:]),
    dtype=torch.int8,
  ).float()
  return {
    "basis": (basis_raw * float(basis_step)).reshape(int(components), int(height), int(width)),
    "coeffs": (coeff_raw * float(coeff_step)).reshape(int(frame_count), int(components)),
    "cache": {},
  }


def apply_latent_luma_rgb(frame: torch.Tensor, latent_luma, frame_idx: int) -> torch.Tensor:
  if latent_luma is None or frame_idx >= latent_luma["coeffs"].shape[0]:
    return frame
  h, w, _ = frame.shape
  cache_key = (h, w)
  basis = latent_luma["cache"].get(cache_key)
  if basis is None:
    basis = F.interpolate(
      latent_luma["basis"].unsqueeze(0),
      size=(h, w),
      mode="bilinear",
      align_corners=False,
    ).squeeze(0)
    latent_luma["cache"][cache_key] = basis
  coeffs = latent_luma["coeffs"][frame_idx]
  correction = torch.einsum("k,khw->hw", coeffs, basis)
  x = frame.float() + correction.view(h, w, 1)
  return x.clamp(0, 255).round().to(torch.uint8)


def apply_postfilter_rgb(frame: torch.Tensor, postfilter) -> torch.Tensor:
  if postfilter is None:
    return frame

  model = postfilter["model"]
  device = postfilter["device"]
  low_width = postfilter["low_width"]
  low_height = postfilter["low_height"]

  with torch.inference_mode():
    x = frame.permute(2, 0, 1).unsqueeze(0).float().to(device=device)
    low = interpolate_image(x, low_height, low_width, POSTFILTER_RESIZE_MODE)
    corrected_low = model(low)
    residual_low = corrected_low - low
    residual = interpolate_image(residual_low, frame.shape[0], frame.shape[1], POSTFILTER_RESIZE_MODE)
    out = (x + residual).clamp(0, 255).round().to(torch.uint8)
  return out.squeeze(0).permute(1, 2, 0).cpu()


def temporal_factor_from_diff(
  diff: torch.Tensor,
  low: float,
  high: float,
  min_factor: float,
  max_factor: float,
) -> torch.Tensor:
  span = max(1e-6, high - low)
  factor = 1.0 - ((diff - low) / span)
  factor = factor.clamp(0.0, 1.0)
  if min_factor != 0.0 or max_factor != 1.0:
    factor = min_factor + factor * (max_factor - min_factor)
  return factor


def temporal_blend_rgb(
  frame: torch.Tensor,
  prev: torch.Tensor,
  segment_idx: int,
  frame_in_segment: int,
  final_segment_idx: int | None,
) -> torch.Tensor:
  blend = float(scheduled_segment_env(segment_idx, frame_in_segment, "BLEND", str(TEMPORAL_BLEND), final_segment_idx))
  if blend == 0.0 or prev is None:
    return frame

  mode = scheduled_segment_env(segment_idx, frame_in_segment, "MODE", TEMPORAL_MODE, final_segment_idx).replace("_", "-")
  low = float(scheduled_segment_env(segment_idx, frame_in_segment, "DIFF_LOW", str(TEMPORAL_DIFF_LOW), final_segment_idx))
  high = float(scheduled_segment_env(segment_idx, frame_in_segment, "DIFF_HIGH", str(TEMPORAL_DIFF_HIGH), final_segment_idx))
  min_factor = float(scheduled_segment_env(segment_idx, frame_in_segment, "MIN_FACTOR", str(TEMPORAL_MIN_FACTOR), final_segment_idx))
  max_factor = float(scheduled_segment_env(segment_idx, frame_in_segment, "MAX_FACTOR", str(TEMPORAL_MAX_FACTOR), final_segment_idx))
  motion_radius = int(scheduled_segment_env(segment_idx, frame_in_segment, "MOTION_RADIUS", str(TEMPORAL_MOTION_RADIUS), final_segment_idx))
  motion_downsample = int(scheduled_segment_env(segment_idx, frame_in_segment, "MOTION_DOWNSAMPLE", str(TEMPORAL_MOTION_DOWNSAMPLE), final_segment_idx))
  current = frame.float()
  previous = prev.float()

  if mode == "fixed":
    weight = blend
    out = current * (1.0 - weight) + previous * weight
  elif mode in {"pixel", "pixel-gated", "local"}:
    diff = (current - previous).abs().mean(dim=2, keepdim=True)
    weight = blend * temporal_factor_from_diff(diff, low, high, min_factor, max_factor)
    out = current * (1.0 - weight) + previous * weight
  elif mode in {"frame", "frame-gated", "global"}:
    diff = (current - previous).abs().mean()
    weight = float(blend * temporal_factor_from_diff(diff, low, high, min_factor, max_factor).item())
    out = current * (1.0 - weight) + previous * weight
  elif mode in {"lowpass", "low-pass", "lp"}:
    cur_low = blur_nchw(current.permute(2, 0, 1).unsqueeze(0), TEMPORAL_KERNEL).squeeze(0).permute(1, 2, 0)
    prev_low = blur_nchw(previous.permute(2, 0, 1).unsqueeze(0), TEMPORAL_KERNEL).squeeze(0).permute(1, 2, 0)
    out = current + blend * (prev_low - cur_low)
  elif mode in {"lowpass-pixel", "low-pass-pixel", "lp-pixel"}:
    cur_low = blur_nchw(current.permute(2, 0, 1).unsqueeze(0), TEMPORAL_KERNEL).squeeze(0).permute(1, 2, 0)
    prev_low = blur_nchw(previous.permute(2, 0, 1).unsqueeze(0), TEMPORAL_KERNEL).squeeze(0).permute(1, 2, 0)
    diff = (cur_low - prev_low).abs().mean(dim=2, keepdim=True)
    weight = blend * temporal_factor_from_diff(diff, low, high, min_factor, max_factor)
    out = current + weight * (prev_low - cur_low)
  elif mode in {"lowpass-frame", "low-pass-frame", "lp-frame"}:
    cur_low = blur_nchw(current.permute(2, 0, 1).unsqueeze(0), TEMPORAL_KERNEL).squeeze(0).permute(1, 2, 0)
    prev_low = blur_nchw(previous.permute(2, 0, 1).unsqueeze(0), TEMPORAL_KERNEL).squeeze(0).permute(1, 2, 0)
    diff = (cur_low - prev_low).abs().mean()
    weight = float(blend * temporal_factor_from_diff(diff, low, high, min_factor, max_factor).item())
    out = current + weight * (prev_low - cur_low)
  elif mode in {"motion", "motion-fixed", "global-motion"}:
    dy, dx = estimate_global_shift(current, previous, motion_radius, motion_downsample)
    shifted = shift_rgb(previous, current, dy, dx)
    out = current * (1.0 - blend) + shifted * blend
  elif mode in {"motion-lowpass", "motion-low-pass", "motion-lp"}:
    dy, dx = estimate_global_shift(current, previous, motion_radius, motion_downsample)
    shifted = shift_rgb(previous, current, dy, dx)
    cur_low = blur_nchw(current.permute(2, 0, 1).unsqueeze(0), TEMPORAL_KERNEL).squeeze(0).permute(1, 2, 0)
    prev_low = blur_nchw(shifted.permute(2, 0, 1).unsqueeze(0), TEMPORAL_KERNEL).squeeze(0).permute(1, 2, 0)
    out = current + blend * (prev_low - cur_low)
  elif mode in {"motion-frame", "motion-gated", "global-motion-gated"}:
    dy, dx = estimate_global_shift(current, previous, motion_radius, motion_downsample)
    shifted = shift_rgb(previous, current, dy, dx)
    diff = (current - shifted).abs().mean()
    weight = float(blend * temporal_factor_from_diff(diff, low, high, min_factor, max_factor).item())
    out = current * (1.0 - weight) + shifted * weight
  elif mode == "luma":
    cur_y = luma_plane(current).unsqueeze(2)
    prev_y = luma_plane(previous).unsqueeze(2)
    out = current + blend * (prev_y - cur_y)
  elif mode == "chroma":
    mixed = current * (1.0 - blend) + previous * blend
    cur_y = luma_plane(current).unsqueeze(2)
    mixed_y = luma_plane(mixed).unsqueeze(2)
    out = mixed + (cur_y - mixed_y)
  else:
    raise SystemExit(f"unknown temporal mode for segment {segment_idx}: {mode}")

  return out.clamp(0, 255).round().to(torch.uint8)


def iter_segment_rgb(path: Path):
  container = av.open(str(path))
  stream = container.streams.video[0]
  try:
    for frame in container.decode(stream):
      if frame.format.name != "yuv420p":
        frame = frame.reformat(format="yuv420p")
      yield yuv420_to_rgb(frame)
  finally:
    container.close()


def inflate_segments(segment_dir: Path, dst: Path) -> int:
  target_w, target_h = camera_size
  segment_paths = sorted(segment_dir.glob("*.ivf"))
  if not segment_paths:
    raise SystemExit(f"no .ivf segments found in {segment_dir}")

  frames = 0
  prev = None
  sidechannel = load_sidechannel(segment_dir)
  latent_luma = load_latent_luma_bundle(segment_dir)
  postfilter = load_postfilter_bundle(segment_dir)
  final_segment_idx = len(segment_paths) - 1
  asym_mode = pair_asym_mode(segment_dir).replace("_", "-")
  if asym_mode in {"", "none"}:
    asym_mode = "none"
  if asym_mode not in {"none", "odd-duplicate", "odd-prevblend", "odd-motion", "split-even-lowq"}:
    raise SystemExit(f"unknown pair asymmetry mode: {asym_mode}")
  previous_odd = None
  with dst.open("wb") as file:
    if asym_mode == "split-even-lowq":
      split_indices = sorted(
        int(path.stem.split("_", 1)[0])
        for path in segment_dir.glob("*_odd.ivf")
      )
      if not split_indices:
        raise SystemExit(f"split-even-lowq mode found no *_odd.ivf segments in {segment_dir}")
      final_segment_idx = split_indices[-1]
      for segment_idx in split_indices:
        even_path = segment_dir / f"{segment_idx:03d}_even.ivf"
        odd_path = segment_dir / f"{segment_idx:03d}_odd.ivf"
        if not even_path.exists() or not odd_path.exists():
          raise SystemExit(f"missing split pair segment: {even_path} / {odd_path}")
        for pair_in_segment, (even_rgb, odd_rgb) in enumerate(zip_longest(iter_segment_rgb(even_path), iter_segment_rgb(odd_path))):
          if even_rgb is None or odd_rgb is None:
            raise SystemExit(f"split pair frame count mismatch in segment {segment_idx}")
          for rgb, frame_in_segment in ((even_rgb, pair_in_segment * 2), (odd_rgb, pair_in_segment * 2 + 1)):
            h, w, _ = rgb.shape
            if h != target_h or w != target_w:
              rgb = resize_rgb(rgb, target_w, target_h, segment_idx)
            rgb = postprocess_rgb(rgb, segment_idx)
            rgb = temporal_blend_rgb(rgb, prev, segment_idx, frame_in_segment, final_segment_idx)
            rgb = color_correct_rgb(rgb, segment_idx, frame_in_segment)
            out = apply_sidechannel_rgb(rgb, sidechannel, frames)
            out = apply_latent_luma_rgb(out, latent_luma, frames)
            out = apply_postfilter_rgb(out, postfilter)
            file.write(out.contiguous().numpy().tobytes())
            prev = out if SIDECHANNEL_FEEDBACK else rgb
            frames += 1
      return frames

    for segment_idx, segment_path in enumerate(segment_paths):
      if "_" in segment_path.stem:
        continue
      for decoded_frame_in_segment, rgb in enumerate(iter_segment_rgb(segment_path)):
        frame_in_segment = decoded_frame_in_segment * 2 + 1 if asym_mode != "none" else decoded_frame_in_segment
        h, w, _ = rgb.shape
        if h != target_h or w != target_w:
          rgb = resize_rgb(rgb, target_w, target_h, segment_idx)
        rgb = postprocess_rgb(rgb, segment_idx)
        rgb = temporal_blend_rgb(rgb, prev, segment_idx, frame_in_segment, final_segment_idx)
        rgb = color_correct_rgb(rgb, segment_idx, frame_in_segment)
        out = apply_sidechannel_rgb(rgb, sidechannel, frames)
        out = apply_latent_luma_rgb(out, latent_luma, frames)
        out = apply_postfilter_rgb(out, postfilter)
        if asym_mode == "none":
          file.write(out.contiguous().numpy().tobytes())
          prev = out if SIDECHANNEL_FEEDBACK else rgb
          frames += 1
        else:
          even = reconstruct_pair_even_frame(out, previous_odd, asym_mode)
          file.write(even.contiguous().numpy().tobytes())
          file.write(out.contiguous().numpy().tobytes())
          previous_odd = out
          prev = out if SIDECHANNEL_FEEDBACK else rgb
          frames += 2
  return frames


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("usage: inflate.py <segment-dir> <output.raw>", file=sys.stderr)
    raise SystemExit(2)
  n = inflate_segments(Path(sys.argv[1]), Path(sys.argv[2]))
  print(f"saved {n} frames")
