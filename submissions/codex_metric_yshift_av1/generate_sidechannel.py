#!/usr/bin/env python
import argparse
import collections
import itertools
import os
import struct
import sys
from pathlib import Path

import av
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inflate
from frame_utils import seq_len, yuv420_to_rgb


MAGIC = b"SC01"
MODE_YMEAN = 1
MODE_RGBMEAN = 2
MODE_YBANDS = 3
MODE_YPLANE = 4
MODE_YGRID = 5
MODE_Y_SAT = 6
MODE_Y_SHIFT = 7
HEADER = struct.Struct("<4sBBIf")
SAT_RAW_STEP = 0.01


def configure_torch_threads() -> None:
  threads_text = os.environ.get("CODEX_TORCH_THREADS") or os.environ.get("OMP_NUM_THREADS")
  if not threads_text:
    return
  threads = max(1, int(threads_text))
  torch.set_num_threads(threads)
  try:
    torch.set_num_interop_threads(max(1, min(4, threads)))
  except RuntimeError:
    pass


def iter_source_rgb(path: Path):
  container = av.open(str(path))
  stream = container.streams.video[0]
  try:
    for frame in container.decode(stream):
      if frame.format.name != "yuv420p":
        frame = frame.reformat(format="yuv420p")
      yield yuv420_to_rgb(frame)
  finally:
    container.close()


def band_slices(height: int, bands: int):
  for band in range(bands):
    y0 = (height * band) // bands
    y1 = (height * (band + 1)) // bands
    yield y0, y1


def parse_ygrid_shape(mode: str) -> tuple[int, int] | None:
  if not mode.startswith("ygrid"):
    return None
  tail = mode.removeprefix("ygrid").replace("x", ",")
  parts = [item for item in tail.split(",") if item]
  if not parts:
    return 8, 6
  if len(parts) != 2:
    raise SystemExit(f"expected ygridCOLSxROWS mode, got: {mode}")
  cols = int(parts[0])
  rows = int(parts[1])
  if rows <= 0 or cols <= 0 or rows * cols > 255:
    raise SystemExit(f"unsupported ygrid shape: {cols}x{rows}")
  return cols, rows


def quantize(values: list[float], step: float) -> bytes:
  out = bytearray()
  for value in values:
    q = int(round(value / step))
    q = max(-127, min(127, q))
    out.append(q & 0xFF)
  return bytes(out)


def correction_values(mode: str, gt: torch.Tensor, pred: torch.Tensor, gain: float) -> tuple[int, list[float]]:
  diff = gt.float() - pred.float()
  if mode == "ymean":
    value = inflate.luma_plane(diff).mean().item() * gain
    return MODE_YMEAN, [value]
  if mode == "rgbmean":
    values = diff.mean(dim=(0, 1)).tolist()
    return MODE_RGBMEAN, [value * gain for value in values]
  if mode.startswith("ybands"):
    bands = int(mode.removeprefix("ybands") or "3")
    if bands <= 0 or bands > 32:
      raise SystemExit(f"unsupported ybands count: {bands}")
    luma = inflate.luma_plane(diff)
    values = []
    for y0, y1 in band_slices(luma.shape[0], bands):
      values.append(luma[y0:y1].mean().item() * gain)
    return MODE_YBANDS, values
  grid_shape = parse_ygrid_shape(mode)
  if grid_shape:
    cols, rows = grid_shape
    luma = inflate.luma_plane(diff)
    grid = F.interpolate(
      luma.view(1, 1, *luma.shape),
      size=(rows, cols),
      mode="area",
    ).flatten()
    return MODE_YGRID, [value.item() * gain for value in grid]
  raise SystemExit(f"unsupported sidechannel mode: {mode}")


def metric_device(name: str) -> torch.device:
  if name != "auto":
    return torch.device(name)
  if torch.cuda.is_available():
    return torch.device("cuda", 0)
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def sidechannel_pose_weight() -> float:
  return float(os.environ.get("CODEX_SIDECHANNEL_POSE_WEIGHT", "6.3"))


def sidechannel_seg_weight() -> float:
  return float(os.environ.get("CODEX_SIDECHANNEL_SEG_WEIGHT", "100.0"))


def sidechannel_score_mode() -> str:
  return os.environ.get("CODEX_SIDECHANNEL_SCORE_MODE", "linear").strip().lower()


def combined_metric_score(pose: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
  mode = sidechannel_score_mode()
  if mode == "linear":
    return sidechannel_pose_weight() * pose + sidechannel_seg_weight() * seg
  if mode == "exact":
    return torch.sqrt(torch.clamp(10.0 * pose, min=0.0)) + sidechannel_seg_weight() * seg
  raise SystemExit(f"unsupported CODEX_SIDECHANNEL_SCORE_MODE: {mode}")


def parse_candidates(text: str) -> list[float]:
  values = [float(item.strip()) for item in text.replace(";", ",").split(",") if item.strip()]
  if not values:
    raise SystemExit("metric sidechannel needs at least one candidate")
  return values


def parse_sat_candidates() -> list[int]:
  text = os.environ.get("CODEX_SIDECHANNEL_SAT_CANDIDATES", "-6,-3,0,3,6")
  values = [int(round(float(item.strip()))) for item in text.replace(";", ",").split(",") if item.strip()]
  if not values:
    raise SystemExit("metric y+sat sidechannel needs at least one saturation candidate")
  return values


def parse_shift_candidates() -> list[int]:
  text = os.environ.get("CODEX_SIDECHANNEL_SHIFT_CANDIDATES", "-2,-1,0,1,2")
  values = [int(round(float(item.strip()))) for item in text.replace(";", ",").split(",") if item.strip()]
  if not values:
    raise SystemExit("metric y+shift sidechannel needs at least one shift candidate")
  return values


def iter_pairs(gt_iter, pred_iter):
  while True:
    gt_pair = []
    pred_pair = []
    for _ in range(seq_len):
      try:
        gt = next(gt_iter)
        pred = next(pred_iter)
      except StopIteration:
        if gt_pair or pred_pair:
          raise SystemExit("source and prediction ended mid-sequence")
        return
      gt_pair.append(gt)
      pred_pair.append(pred)
    yield torch.stack(gt_pair), torch.stack(pred_pair)


def metric_y_values(
  src: Path,
  segment_dir: Path,
  mode: str,
  candidates: list[float],
  device: torch.device,
) -> tuple[int, int, list[float]]:
  from modules import DistortionNet, posenet_sd_path, segnet_sd_path

  shared = mode == "metric-y-shared"
  if mode not in {"metric-y", "metric-y-shared"}:
    raise SystemExit(f"unsupported metric sidechannel mode: {mode}")

  net = DistortionNet().eval().to(device=device)
  net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
  candidate_pairs = (
    [(value, value) for value in candidates]
    if shared
    else list(itertools.product(candidates, repeat=seq_len))
  )
  all_values = []
  pair_count = 0

  with torch.inference_mode():
    for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
      pair_count += 1
      gt_batch = gt_pair.unsqueeze(0).to(device=device)
      gt_pose, gt_seg = net(gt_batch)

      cand = pred_pair.unsqueeze(0).repeat(len(candidate_pairs), 1, 1, 1, 1).float()
      for idx, pair in enumerate(candidate_pairs):
        for frame_idx, bias in enumerate(pair):
          cand[idx, frame_idx] = cand[idx, frame_idx] + bias
      cand = cand.clamp(0, 255).round().to(torch.uint8).to(device=device)

      cand_pose, cand_seg = net(cand)
      pose = net.posenet.compute_distortion(
        {key: value.repeat(len(candidate_pairs), *([1] * (value.ndim - 1))) for key, value in gt_pose.items()},
        cand_pose,
      )
      seg = net.segnet.compute_distortion(
        gt_seg.repeat(len(candidate_pairs), *([1] * (gt_seg.ndim - 1))),
        cand_seg,
      )
      score = combined_metric_score(pose, seg)
      best_idx = int(score.argmin().item())
      all_values.extend(candidate_pairs[best_idx])
      progress_interval = write_sidechannel.progress_interval
      if progress_interval > 0 and pair_count % progress_interval == 0:
        print(f"metric-y searched {pair_count * seq_len} frames", flush=True)

  return MODE_YMEAN, 1, all_values


def metric_y_coordinate_values(
  src: Path,
  segment_dir: Path,
  candidates: list[float],
  device: torch.device,
  passes: int,
) -> tuple[int, int, list[float]]:
  from modules import DistortionNet, posenet_sd_path, segnet_sd_path

  net = DistortionNet().eval().to(device=device)
  net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
  all_values = []
  passes = max(1, passes)
  pair_count = 0

  with torch.inference_mode():
    for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
      pair_count += 1
      gt_batch = gt_pair.unsqueeze(0).to(device=device)
      gt_pose, gt_seg = net(gt_batch)

      base = pred_pair.float()
      current = torch.zeros((seq_len,), dtype=torch.float32)
      for _ in range(passes):
        for frame_idx in range(seq_len):
          cand = base.unsqueeze(0).repeat(len(candidates), 1, 1, 1, 1)
          cand = cand + current.view(1, seq_len, 1, 1, 1)
          for idx, value in enumerate(candidates):
            cand[idx, frame_idx] = base[frame_idx] + value
          score = metric_score(net, gt_pose, gt_seg, cand, device)
          current[frame_idx] = candidates[int(score.argmin().item())]

      all_values.extend(current.tolist())
      progress_interval = write_sidechannel.progress_interval
      if progress_interval > 0 and pair_count % progress_interval == 0:
        print(f"metric-y-coordinate searched {pair_count * seq_len} frames", flush=True)

  return MODE_YMEAN, 1, all_values


def metric_score_preprocessed(
  net,
  gt_pose,
  gt_seg,
  cand_pose_in: torch.Tensor,
  cand_seg_in: torch.Tensor,
) -> torch.Tensor:
  candidate_count = cand_pose_in.shape[0]
  cand_pose = net.posenet(cand_pose_in)
  cand_seg = net.segnet(cand_seg_in)
  pose = net.posenet.compute_distortion(
    {key: value.repeat(candidate_count, *([1] * (value.ndim - 1))) for key, value in gt_pose.items()},
    cand_pose,
  )
  seg = net.segnet.compute_distortion(
    gt_seg.repeat(candidate_count, *([1] * (gt_seg.ndim - 1))),
    cand_seg,
  )
  return combined_metric_score(pose, seg)


def add_preprocessed_luma_bias(
  pose_in: torch.Tensor,
  seg_in: torch.Tensor,
  frame_idx: int,
  bias: float,
) -> None:
  pose_start = frame_idx * 6
  pose_in[:, pose_start : pose_start + 4] = (pose_in[:, pose_start : pose_start + 4] + bias).clamp(0, 255)
  if frame_idx == seq_len - 1:
    seg_in[:] = (seg_in + bias).clamp(0, 255)


def add_preprocessed_pose_luma_bias(
  pose_in: torch.Tensor,
  frame_idx: int,
  bias: float,
) -> None:
  pose_start = frame_idx * 6
  pose_in[:, pose_start : pose_start + 4] = (pose_in[:, pose_start : pose_start + 4] + bias).clamp(0, 255)


def luma_plane_map(height: int, width: int, coeffs: torch.Tensor) -> torch.Tensor:
  y = torch.linspace(-1.0, 1.0, height, dtype=coeffs.dtype, device=coeffs.device).view(height, 1)
  x = torch.linspace(-1.0, 1.0, width, dtype=coeffs.dtype, device=coeffs.device).view(1, width)
  if coeffs.numel() == 3:
    return coeffs[0] + coeffs[1] * x + coeffs[2] * y
  if coeffs.numel() == 6:
    x2 = x * x - (1.0 / 3.0)
    y2 = y * y - (1.0 / 3.0)
    return coeffs[0] + coeffs[1] * x + coeffs[2] * y + coeffs[3] * (x * y) + coeffs[4] * x2 + coeffs[5] * y2
  raise SystemExit("luma basis sidechannel requires 3 or 6 coefficients")


def yplane_basis_channels(mode: str) -> int:
  return 6 if "ybasis6" in mode or "yquad" in mode else 3


def add_preprocessed_luma_plane(
  pose_in: torch.Tensor,
  seg_in: torch.Tensor,
  frame_idx: int,
  coeffs: torch.Tensor,
) -> None:
  pose_start = frame_idx * 6
  pose_map = luma_plane_map(pose_in.shape[-2], pose_in.shape[-1], coeffs)
  pose_in[:, pose_start : pose_start + 4] = (
    pose_in[:, pose_start : pose_start + 4] + pose_map.view(1, 1, *pose_map.shape)
  ).clamp(0, 255)
  if frame_idx == seq_len - 1:
    seg_map = luma_plane_map(seg_in.shape[-2], seg_in.shape[-1], coeffs)
    seg_in[:] = (seg_in + seg_map.view(1, 1, *seg_map.shape)).clamp(0, 255)


def add_preprocessed_pose_luma_plane(
  pose_in: torch.Tensor,
  frame_idx: int,
  coeffs: torch.Tensor,
) -> None:
  pose_start = frame_idx * 6
  pose_map = luma_plane_map(pose_in.shape[-2], pose_in.shape[-1], coeffs)
  pose_in[:, pose_start : pose_start + 4] = (
    pose_in[:, pose_start : pose_start + 4] + pose_map.view(1, 1, *pose_map.shape)
  ).clamp(0, 255)


def pose_preprocess_pair(posenet, pair: torch.Tensor, device: torch.device) -> torch.Tensor:
  x = pair.unsqueeze(0).permute(0, 1, 4, 2, 3).float().to(device=device)
  return posenet.preprocess_input(x)


def pose_metric_score(posenet, gt_pose, cand_pose_in: torch.Tensor) -> torch.Tensor:
  candidate_count = cand_pose_in.shape[0]
  cand_pose = posenet(cand_pose_in)
  return posenet.compute_distortion(
    {key: value.repeat(candidate_count, *([1] * (value.ndim - 1))) for key, value in gt_pose.items()},
    cand_pose,
  )


def metric_y_pose_fast_values(
  src: Path,
  segment_dir: Path,
  mode: str,
  candidates: list[float],
  device: torch.device,
) -> tuple[int, int, list[float]]:
  from safetensors.torch import load_file
  from modules import PoseNet, posenet_sd_path

  shared = mode == "metric-y-pose-shared-fast"
  if mode not in {"metric-y-pose-fast", "metric-y-pose-shared-fast"}:
    raise SystemExit(f"unsupported pose metric sidechannel mode: {mode}")

  posenet = PoseNet().eval().to(device=device)
  posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
  candidate_pairs = (
    [(value, value) for value in candidates]
    if shared
    else list(itertools.product(candidates, repeat=seq_len))
  )
  all_values = []
  pair_count = 0

  with torch.inference_mode():
    for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
      pair_count += 1
      gt_pose_in = pose_preprocess_pair(posenet, gt_pair, device)
      pred_pose_in = pose_preprocess_pair(posenet, pred_pair, device)
      gt_pose = posenet(gt_pose_in)

      cand_pose = pred_pose_in.repeat(len(candidate_pairs), 1, 1, 1)
      for idx, pair in enumerate(candidate_pairs):
        for frame_idx, bias in enumerate(pair):
          add_preprocessed_pose_luma_bias(cand_pose[idx : idx + 1], frame_idx, bias)
      score = pose_metric_score(posenet, gt_pose, cand_pose)
      best_idx = int(score.argmin().item())
      all_values.extend(candidate_pairs[best_idx])
      progress_interval = write_sidechannel.progress_interval
      if progress_interval > 0 and pair_count % progress_interval == 0:
        print(f"{mode} searched {pair_count * seq_len} frames", flush=True)

  return MODE_YMEAN, 1, all_values


def metric_y_pose_coordinate_fast_values(
  src: Path,
  segment_dir: Path,
  candidates: list[float],
  device: torch.device,
  passes: int,
) -> tuple[int, int, list[float]]:
  from safetensors.torch import load_file
  from modules import PoseNet, posenet_sd_path

  posenet = PoseNet().eval().to(device=device)
  posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
  all_values = []
  passes = max(1, passes)
  pair_count = 0

  with torch.inference_mode():
    for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
      pair_count += 1
      gt_pose_in = pose_preprocess_pair(posenet, gt_pair, device)
      pred_pose_in = pose_preprocess_pair(posenet, pred_pair, device)
      gt_pose = posenet(gt_pose_in)

      current = torch.zeros((seq_len,), dtype=torch.float32)
      for _ in range(passes):
        for frame_idx in range(seq_len):
          cand_pose = pred_pose_in.repeat(len(candidates), 1, 1, 1)
          for current_frame_idx, bias in enumerate(current.tolist()):
            add_preprocessed_pose_luma_bias(cand_pose, current_frame_idx, bias)
          for idx, value in enumerate(candidates):
            add_preprocessed_pose_luma_bias(
              cand_pose[idx : idx + 1],
              frame_idx,
              value - float(current[frame_idx].item()),
            )
          score = pose_metric_score(posenet, gt_pose, cand_pose)
          current[frame_idx] = candidates[int(score.argmin().item())]

      all_values.extend(current.tolist())
      progress_interval = write_sidechannel.progress_interval
      if progress_interval > 0 and pair_count % progress_interval == 0:
        print(f"metric-y-pose-coordinate-fast searched {pair_count * seq_len} frames", flush=True)

  return MODE_YMEAN, 1, all_values


def metric_yplane_coordinate_fast_values(
  src: Path,
  segment_dir: Path,
  mode: str,
  candidates: list[float],
  device: torch.device,
  passes: int,
) -> tuple[int, int, list[float]]:
  from modules import DistortionNet, posenet_sd_path, segnet_sd_path

  net = DistortionNet().eval().to(device=device)
  net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
  all_values = []
  passes = max(1, passes)
  basis_channels = yplane_basis_channels(mode)
  candidate_tensor = torch.tensor(candidates, dtype=torch.float32, device=device)
  pair_count = 0

  with torch.inference_mode():
    for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
      pair_count += 1
      gt_pose_in, gt_seg_in = net.preprocess_input(gt_pair.unsqueeze(0).to(device=device))
      pred_pose_in, pred_seg_in = net.preprocess_input(pred_pair.unsqueeze(0).to(device=device))
      gt_pose, gt_seg = net.posenet(gt_pose_in), net.segnet(gt_seg_in)

      current = torch.zeros((seq_len, basis_channels), dtype=torch.float32, device=device)
      for _ in range(passes):
        for frame_idx in range(seq_len):
          for coeff_idx in range(basis_channels):
            cand_pose = pred_pose_in.repeat(len(candidates), 1, 1, 1)
            cand_seg = pred_seg_in.repeat(len(candidates), 1, 1, 1)
            for current_frame_idx in range(seq_len):
              add_preprocessed_luma_plane(cand_pose, cand_seg, current_frame_idx, current[current_frame_idx])
            for idx, value in enumerate(candidate_tensor):
              candidate_coeffs = current[frame_idx].clone()
              candidate_coeffs[coeff_idx] = value
              add_preprocessed_luma_plane(
                cand_pose[idx : idx + 1],
                cand_seg[idx : idx + 1],
                frame_idx,
                candidate_coeffs - current[frame_idx],
              )
            score = metric_score_preprocessed(net, gt_pose, gt_seg, cand_pose, cand_seg)
            current[frame_idx, coeff_idx] = candidate_tensor[int(score.argmin().item())]

      for frame_idx in range(seq_len):
        all_values.extend(current[frame_idx].tolist())
      progress_interval = write_sidechannel.progress_interval
      if progress_interval > 0 and pair_count % progress_interval == 0:
        print(f"{mode} searched {pair_count * seq_len} frames", flush=True)

  return MODE_YPLANE, basis_channels, all_values


def metric_yplane_pose_coordinate_fast_values(
  src: Path,
  segment_dir: Path,
  mode: str,
  candidates: list[float],
  device: torch.device,
  passes: int,
) -> tuple[int, int, list[float]]:
  from safetensors.torch import load_file
  from modules import PoseNet, posenet_sd_path

  posenet = PoseNet().eval().to(device=device)
  posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
  all_values = []
  passes = max(1, passes)
  basis_channels = yplane_basis_channels(mode)
  candidate_tensor = torch.tensor(candidates, dtype=torch.float32, device=device)
  pair_count = 0

  with torch.inference_mode():
    for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
      pair_count += 1
      gt_pose_in = pose_preprocess_pair(posenet, gt_pair, device)
      pred_pose_in = pose_preprocess_pair(posenet, pred_pair, device)
      gt_pose = posenet(gt_pose_in)

      current = torch.zeros((seq_len, basis_channels), dtype=torch.float32, device=device)
      for _ in range(passes):
        for frame_idx in range(seq_len):
          for coeff_idx in range(basis_channels):
            cand_pose = pred_pose_in.repeat(len(candidates), 1, 1, 1)
            for current_frame_idx in range(seq_len):
              add_preprocessed_pose_luma_plane(cand_pose, current_frame_idx, current[current_frame_idx])
            for idx, value in enumerate(candidate_tensor):
              candidate_coeffs = current[frame_idx].clone()
              candidate_coeffs[coeff_idx] = value
              add_preprocessed_pose_luma_plane(
                cand_pose[idx : idx + 1],
                frame_idx,
                candidate_coeffs - current[frame_idx],
              )
            score = pose_metric_score(posenet, gt_pose, cand_pose)
            current[frame_idx, coeff_idx] = candidate_tensor[int(score.argmin().item())]

      for frame_idx in range(seq_len):
        all_values.extend(current[frame_idx].tolist())
      progress_interval = write_sidechannel.progress_interval
      if progress_interval > 0 and pair_count % progress_interval == 0:
        print(f"{mode} searched {pair_count * seq_len} frames", flush=True)

  return MODE_YPLANE, basis_channels, all_values


def add_preprocessed_yband_bias(
  pose_in: torch.Tensor,
  seg_in: torch.Tensor,
  frame_idx: int,
  band_idx: int,
  bands: int,
  bias: float,
) -> None:
  pose_h = pose_in.shape[-2]
  seg_h = seg_in.shape[-2]
  pose_y0 = (pose_h * band_idx) // bands
  pose_y1 = (pose_h * (band_idx + 1)) // bands
  seg_y0 = (seg_h * band_idx) // bands
  seg_y1 = (seg_h * (band_idx + 1)) // bands
  pose_start = frame_idx * 6
  pose_in[:, pose_start : pose_start + 4, pose_y0:pose_y1, :] = (
    pose_in[:, pose_start : pose_start + 4, pose_y0:pose_y1, :] + bias
  ).clamp(0, 255)
  if frame_idx == seq_len - 1:
    seg_in[:, :, seg_y0:seg_y1, :] = (seg_in[:, :, seg_y0:seg_y1, :] + bias).clamp(0, 255)


def set_preprocessed_yband_bias(
  pose_in: torch.Tensor,
  seg_in: torch.Tensor,
  base_pose_in: torch.Tensor,
  base_seg_in: torch.Tensor,
  frame_idx: int,
  band_idx: int,
  bands: int,
  bias: float,
) -> None:
  pose_h = pose_in.shape[-2]
  seg_h = seg_in.shape[-2]
  pose_y0 = (pose_h * band_idx) // bands
  pose_y1 = (pose_h * (band_idx + 1)) // bands
  seg_y0 = (seg_h * band_idx) // bands
  seg_y1 = (seg_h * (band_idx + 1)) // bands
  pose_start = frame_idx * 6
  pose_in[:, pose_start : pose_start + 4, pose_y0:pose_y1, :] = (
    base_pose_in[:, pose_start : pose_start + 4, pose_y0:pose_y1, :] + bias
  ).clamp(0, 255)
  if frame_idx == seq_len - 1:
    seg_in[:, :, seg_y0:seg_y1, :] = (
      base_seg_in[:, :, seg_y0:seg_y1, :] + bias
    ).clamp(0, 255)


def set_preprocessed_ygrid_correction(
  pose_in: torch.Tensor,
  seg_in: torch.Tensor,
  base_pose_in: torch.Tensor,
  base_seg_in: torch.Tensor,
  frame_idx: int,
  values: torch.Tensor,
  rows: int,
  cols: int,
) -> None:
  if values.ndim == 1:
    values = values.view(1, -1)
  if values.shape[1] != rows * cols:
    raise SystemExit(f"metric ygrid expected {rows * cols} values, got {values.shape[1]}")

  pose_start = frame_idx * 6
  pose_map = F.interpolate(
    values.view(values.shape[0], 1, rows, cols),
    size=(pose_in.shape[-2], pose_in.shape[-1]),
    mode="bilinear",
    align_corners=False,
  )
  pose_in[:, pose_start : pose_start + 4] = (
    base_pose_in[:, pose_start : pose_start + 4] + pose_map
  ).clamp(0, 255)

  if frame_idx == seq_len - 1:
    seg_map = F.interpolate(
      values.view(values.shape[0], 1, rows, cols),
      size=(seg_in.shape[-2], seg_in.shape[-1]),
      mode="bilinear",
      align_corners=False,
    )
    seg_in[:] = (base_seg_in + seg_map).clamp(0, 255)


def metric_y_fast_values(
  src: Path,
  segment_dir: Path,
  mode: str,
  candidates: list[float],
  device: torch.device,
) -> tuple[int, int, list[float]]:
  from modules import DistortionNet, posenet_sd_path, segnet_sd_path

  shared = mode == "metric-y-shared-fast"
  if mode not in {"metric-y-fast", "metric-y-shared-fast"}:
    raise SystemExit(f"unsupported fast metric sidechannel mode: {mode}")

  net = DistortionNet().eval().to(device=device)
  net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
  candidate_pairs = (
    [(value, value) for value in candidates]
    if shared
    else list(itertools.product(candidates, repeat=seq_len))
  )
  all_values = []
  pair_count = 0

  with torch.inference_mode():
    for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
      pair_count += 1
      gt_pose_in, gt_seg_in = net.preprocess_input(gt_pair.unsqueeze(0).to(device=device))
      pred_pose_in, pred_seg_in = net.preprocess_input(pred_pair.unsqueeze(0).to(device=device))
      gt_pose, gt_seg = net.posenet(gt_pose_in), net.segnet(gt_seg_in)

      cand_pose = pred_pose_in.repeat(len(candidate_pairs), 1, 1, 1)
      cand_seg = pred_seg_in.repeat(len(candidate_pairs), 1, 1, 1)
      for idx, pair in enumerate(candidate_pairs):
        for frame_idx, bias in enumerate(pair):
          add_preprocessed_luma_bias(cand_pose[idx : idx + 1], cand_seg[idx : idx + 1], frame_idx, bias)
      score = metric_score_preprocessed(net, gt_pose, gt_seg, cand_pose, cand_seg)
      best_idx = int(score.argmin().item())
      all_values.extend(candidate_pairs[best_idx])
      progress_interval = write_sidechannel.progress_interval
      if progress_interval > 0 and pair_count % progress_interval == 0:
        print(f"{mode} searched {pair_count * seq_len} frames", flush=True)

  return MODE_YMEAN, 1, all_values


def metric_y_coordinate_fast_values(
  src: Path,
  segment_dir: Path,
  candidates: list[float],
  device: torch.device,
  passes: int,
) -> tuple[int, int, list[float]]:
  from modules import DistortionNet, posenet_sd_path, segnet_sd_path

  net = DistortionNet().eval().to(device=device)
  net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
  all_values = []
  passes = max(1, passes)
  pair_count = 0

  with torch.inference_mode():
    for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
      pair_count += 1
      gt_pose_in, gt_seg_in = net.preprocess_input(gt_pair.unsqueeze(0).to(device=device))
      pred_pose_in, pred_seg_in = net.preprocess_input(pred_pair.unsqueeze(0).to(device=device))
      gt_pose, gt_seg = net.posenet(gt_pose_in), net.segnet(gt_seg_in)

      current = torch.zeros((seq_len,), dtype=torch.float32)
      for _ in range(passes):
        for frame_idx in range(seq_len):
          cand_pose = pred_pose_in.repeat(len(candidates), 1, 1, 1)
          cand_seg = pred_seg_in.repeat(len(candidates), 1, 1, 1)
          for current_frame_idx, bias in enumerate(current.tolist()):
            add_preprocessed_luma_bias(cand_pose, cand_seg, current_frame_idx, bias)
          for idx, value in enumerate(candidates):
            add_preprocessed_luma_bias(
              cand_pose[idx : idx + 1],
              cand_seg[idx : idx + 1],
              frame_idx,
              value - float(current[frame_idx].item()),
            )
          score = metric_score_preprocessed(net, gt_pose, gt_seg, cand_pose, cand_seg)
          current[frame_idx] = candidates[int(score.argmin().item())]

      all_values.extend(current.tolist())
      progress_interval = write_sidechannel.progress_interval
      if progress_interval > 0 and pair_count % progress_interval == 0:
        print(f"metric-y-coordinate-fast searched {pair_count * seq_len} frames", flush=True)

  return MODE_YMEAN, 1, all_values


def env_int(name: str, default: int) -> int:
  return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
  return float(os.environ.get(name, str(default)))


def apply_preprocessed_luma_biases(
  pose_in: torch.Tensor,
  seg_in: torch.Tensor,
  biases: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  cand_pose = pose_in.clone()
  cand_seg = seg_in.clone()
  for frame_idx in range(seq_len):
    bias = biases[frame_idx]
    pose_start = frame_idx * 6
    cand_pose[:, pose_start : pose_start + 4] = (
      cand_pose[:, pose_start : pose_start + 4] + bias
    ).clamp(0, 255)
    if frame_idx == seq_len - 1:
      cand_seg = (cand_seg + bias).clamp(0, 255)
  return cand_pose, cand_seg


def local_candidate_pairs(
  candidates: torch.Tensor,
  center_values: torch.Tensor,
  radius: int,
) -> list[tuple[float, ...]]:
  radius = max(0, radius)
  per_frame = []
  for value in center_values.tolist():
    idx = int((candidates - float(value)).abs().argmin().item())
    lo = max(0, idx - radius)
    hi = min(candidates.numel(), idx + radius + 1)
    per_frame.append([float(item) for item in candidates[lo:hi].tolist()])
  return [tuple(items) for items in itertools.product(*per_frame)]


def metric_y_gradient_values(
  src: Path,
  segment_dir: Path,
  candidates: list[float],
  device: torch.device,
) -> tuple[int, int, list[float]]:
  from modules import DistortionNet, posenet_sd_path, segnet_sd_path

  steps = max(1, env_int("CODEX_SIDECHANNEL_GRAD_STEPS", 12))
  lr = env_float("CODEX_SIDECHANNEL_GRAD_LR", 1.5)
  pose_weight = env_float("CODEX_SIDECHANNEL_GRAD_POSE_WEIGHT", 6.3)
  seg_ce_weight = env_float("CODEX_SIDECHANNEL_GRAD_SEG_CE_WEIGHT", 0.02)
  refine_radius = max(0, env_int("CODEX_SIDECHANNEL_GRAD_REFINE_RADIUS", 1))
  candidate_values = torch.tensor(sorted(candidates), dtype=torch.float32, device=device)
  low = float(candidate_values.min().item())
  high = float(candidate_values.max().item())

  net = DistortionNet().eval().to(device=device)
  net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
  for param in net.parameters():
    param.requires_grad_(False)

  all_values = []
  pair_count = 0

  for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
    pair_count += 1
    gt_pose_in, gt_seg_in = net.preprocess_input(gt_pair.unsqueeze(0).to(device=device))
    pred_pose_in, pred_seg_in = net.preprocess_input(pred_pair.unsqueeze(0).to(device=device))
    with torch.no_grad():
      gt_pose = net.posenet(gt_pose_in)
      gt_seg_logits = net.segnet(gt_seg_in)
      gt_seg_labels = gt_seg_logits.argmax(dim=1)
      gt_pose = {key: value.detach().clone() for key, value in gt_pose.items()}
      gt_seg_logits = gt_seg_logits.detach().clone()
      gt_seg_labels = gt_seg_labels.detach().clone()

    current = torch.zeros((seq_len,), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([current], lr=lr)
    for _ in range(steps):
      optimizer.zero_grad(set_to_none=True)
      biases = current.clamp(low, high)
      cand_pose, cand_seg = apply_preprocessed_luma_biases(pred_pose_in, pred_seg_in, biases)
      pred_pose = net.posenet(cand_pose)
      pose_loss = net.posenet.compute_distortion(gt_pose, pred_pose).mean()
      loss = pose_weight * pose_loss
      if seg_ce_weight != 0.0:
        pred_seg_logits = net.segnet(cand_seg)
        loss = loss + seg_ce_weight * F.cross_entropy(pred_seg_logits, gt_seg_labels)
      loss.backward()
      optimizer.step()
      with torch.no_grad():
        current.clamp_(low, high)

    with torch.inference_mode():
      snapped = candidate_values[(candidate_values.unsqueeze(1) - current.detach().view(1, -1)).abs().argmin(dim=0)]
      if refine_radius > 0:
        candidate_pairs = local_candidate_pairs(candidate_values, snapped, refine_radius)
        cand_pose = pred_pose_in.repeat(len(candidate_pairs), 1, 1, 1)
        cand_seg = pred_seg_in.repeat(len(candidate_pairs), 1, 1, 1)
        for idx, pair in enumerate(candidate_pairs):
          for frame_idx, bias in enumerate(pair):
            add_preprocessed_luma_bias(cand_pose[idx : idx + 1], cand_seg[idx : idx + 1], frame_idx, bias)
        score = metric_score_preprocessed(net, gt_pose, gt_seg_logits, cand_pose, cand_seg)
        snapped = torch.tensor(candidate_pairs[int(score.argmin().item())], dtype=torch.float32, device=device)

    all_values.extend(float(value) for value in snapped.tolist())
    progress_interval = write_sidechannel.progress_interval
    if progress_interval > 0 and pair_count % progress_interval == 0:
      print(f"metric-y-gradient searched {pair_count * seq_len} frames", flush=True)

  return MODE_YMEAN, 1, all_values


def metric_ybands_coordinate_fast_values(
  src: Path,
  segment_dir: Path,
  candidates: list[float],
  device: torch.device,
  passes: int,
  bands: int,
) -> tuple[int, int, list[float]]:
  from modules import DistortionNet, posenet_sd_path, segnet_sd_path

  net = DistortionNet().eval().to(device=device)
  net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
  all_values = []
  passes = max(1, passes)
  pair_count = 0

  with torch.inference_mode():
    for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
      pair_count += 1
      gt_pose_in, gt_seg_in = net.preprocess_input(gt_pair.unsqueeze(0).to(device=device))
      pred_pose_in, pred_seg_in = net.preprocess_input(pred_pair.unsqueeze(0).to(device=device))
      gt_pose, gt_seg = net.posenet(gt_pose_in), net.segnet(gt_seg_in)

      current = torch.zeros((seq_len, bands), dtype=torch.float32)
      working_pose = pred_pose_in.clone()
      working_seg = pred_seg_in.clone()
      for _ in range(passes):
        for frame_idx in range(seq_len):
          for band_idx in range(bands):
            cand_pose = working_pose.repeat(len(candidates), 1, 1, 1)
            cand_seg = working_seg.repeat(len(candidates), 1, 1, 1)
            for idx, value in enumerate(candidates):
              set_preprocessed_yband_bias(
                cand_pose[idx : idx + 1],
                cand_seg[idx : idx + 1],
                pred_pose_in,
                pred_seg_in,
                frame_idx,
                band_idx,
                bands,
                value,
              )
            score = metric_score_preprocessed(net, gt_pose, gt_seg, cand_pose, cand_seg)
            best = candidates[int(score.argmin().item())]
            current[frame_idx, band_idx] = best
            set_preprocessed_yband_bias(
              working_pose,
              working_seg,
              pred_pose_in,
              pred_seg_in,
              frame_idx,
              band_idx,
              bands,
              best,
            )

      for frame_idx in range(seq_len):
        all_values.extend(current[frame_idx].tolist())
      progress_interval = write_sidechannel.progress_interval
      if progress_interval > 0 and pair_count % progress_interval == 0:
        print(f"metric-ybands{bands}-fast searched {pair_count * seq_len} frames", flush=True)

  return MODE_YBANDS, bands, all_values


def parse_metric_ygrid(mode: str) -> tuple[int, int, bool] | None:
  prefix = "metric-ygrid"
  if not mode.startswith(prefix):
    return None
  tail = mode.removeprefix(prefix)
  fast = tail.endswith("-fast")
  tail = tail.removesuffix("-fast")
  tail = tail.removesuffix("-coordinate")
  tail = tail.removesuffix("-coord")
  if not tail:
    cols, rows = 4, 3
  else:
    normalized = tail.replace("x", ",")
    parts = [item for item in normalized.split(",") if item]
    if len(parts) != 2:
      raise SystemExit(f"expected metric-ygridCOLSxROWS mode, got: {mode}")
    cols = int(parts[0])
    rows = int(parts[1])
  if rows <= 0 or cols <= 0 or rows * cols > 64:
    raise SystemExit(f"unsupported metric ygrid shape: {cols}x{rows}")
  return cols, rows, fast


def metric_ygrid_coordinate_fast_values(
  src: Path,
  segment_dir: Path,
  candidates: list[float],
  device: torch.device,
  passes: int,
  cols: int,
  rows: int,
) -> tuple[int, int, list[float]]:
  from modules import DistortionNet, posenet_sd_path, segnet_sd_path

  net = DistortionNet().eval().to(device=device)
  net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
  all_values = []
  passes = max(1, passes)
  channels = rows * cols
  candidate_tensor = torch.tensor(candidates, dtype=torch.float32, device=device)
  pair_count = 0

  with torch.inference_mode():
    for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
      pair_count += 1
      gt_pose_in, gt_seg_in = net.preprocess_input(gt_pair.unsqueeze(0).to(device=device))
      pred_pose_in, pred_seg_in = net.preprocess_input(pred_pair.unsqueeze(0).to(device=device))
      gt_pose, gt_seg = net.posenet(gt_pose_in), net.segnet(gt_seg_in)

      current = torch.zeros((seq_len, channels), dtype=torch.float32, device=device)
      working_pose = pred_pose_in.clone()
      working_seg = pred_seg_in.clone()
      for _ in range(passes):
        for frame_idx in range(seq_len):
          for cell_idx in range(channels):
            cand_values = current[frame_idx].repeat(len(candidates), 1)
            cand_values[:, cell_idx] = candidate_tensor
            cand_pose = working_pose.repeat(len(candidates), 1, 1, 1)
            cand_seg = working_seg.repeat(len(candidates), 1, 1, 1)
            set_preprocessed_ygrid_correction(
              cand_pose,
              cand_seg,
              pred_pose_in,
              pred_seg_in,
              frame_idx,
              cand_values,
              rows,
              cols,
            )
            score = metric_score_preprocessed(net, gt_pose, gt_seg, cand_pose, cand_seg)
            current[frame_idx, cell_idx] = candidate_tensor[int(score.argmin().item())]
            set_preprocessed_ygrid_correction(
              working_pose,
              working_seg,
              pred_pose_in,
              pred_seg_in,
              frame_idx,
              current[frame_idx],
              rows,
              cols,
            )

      for frame_idx in range(seq_len):
        all_values.extend(current[frame_idx].tolist())
      progress_interval = write_sidechannel.progress_interval
      if progress_interval > 0 and pair_count % progress_interval == 0:
        print(f"metric-ygrid{cols}x{rows}-fast searched {pair_count * seq_len} frames", flush=True)

  return MODE_YGRID, channels, all_values


def metric_score(net, gt_pose, gt_seg, cand: torch.Tensor, device: torch.device) -> torch.Tensor:
  candidate_count = cand.shape[0]
  cand = cand.clamp(0, 255).round().to(torch.uint8).to(device=device)
  cand_pose, cand_seg = net(cand)
  pose = net.posenet.compute_distortion(
    {key: value.repeat(candidate_count, *([1] * (value.ndim - 1))) for key, value in gt_pose.items()},
    cand_pose,
  )
  seg = net.segnet.compute_distortion(
    gt_seg.repeat(candidate_count, *([1] * (gt_seg.ndim - 1))),
    cand_seg,
  )
  return combined_metric_score(pose, seg)


def apply_luma_sat_batch(
  cand: torch.Tensor,
  frame_idx: int,
  luma_values: torch.Tensor,
  sat_raw_values: torch.Tensor,
) -> None:
  frame = cand[:, frame_idx]
  frame = frame + luma_values.to(dtype=frame.dtype, device=frame.device).view(-1, 1, 1, 1)
  sat = 1.0 + SAT_RAW_STEP * sat_raw_values.to(dtype=frame.dtype, device=frame.device).view(-1, 1, 1, 1)
  coeffs = torch.tensor([0.299, 0.587, 0.114], dtype=frame.dtype, device=frame.device)
  y = (frame * coeffs.view(1, 1, 1, 3)).sum(dim=3, keepdim=True)
  cand[:, frame_idx] = y + sat * (frame - y)


def shift_frame(frame: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
  if dy == 0 and dx == 0:
    return frame
  h, w, _ = frame.shape
  src_y0 = max(0, -dy)
  src_y1 = min(h, h - dy)
  src_x0 = max(0, -dx)
  src_x1 = min(w, w - dx)
  dst_y0 = max(0, dy)
  dst_y1 = min(h, h + dy)
  dst_x0 = max(0, dx)
  dst_x1 = min(w, w + dx)
  out = frame.clone()
  if src_y1 > src_y0 and src_x1 > src_x0:
    out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
  return out


def apply_luma_shift_batch(
  cand: torch.Tensor,
  frame_idx: int,
  luma_values: torch.Tensor,
  dy_values: torch.Tensor,
  dx_values: torch.Tensor,
) -> None:
  for batch_idx in range(cand.shape[0]):
    frame = cand[batch_idx, frame_idx]
    dy = int(round(float(dy_values[batch_idx].item())))
    dx = int(round(float(dx_values[batch_idx].item())))
    frame = shift_frame(frame, dy, dx)
    cand[batch_idx, frame_idx] = frame + float(luma_values[batch_idx].item())


def metric_y_shift_coordinate_values(
  src: Path,
  segment_dir: Path,
  luma_candidates: list[float],
  shift_candidates: list[int],
  device: torch.device,
  passes: int,
) -> tuple[int, int, list[float]]:
  from modules import DistortionNet, posenet_sd_path, segnet_sd_path

  net = DistortionNet().eval().to(device=device)
  net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
  luma_tensor = torch.tensor(luma_candidates, dtype=torch.float32)
  shift_tensor = torch.tensor(shift_candidates, dtype=torch.float32)
  all_values = []
  passes = max(1, passes)
  pair_count = 0

  with torch.inference_mode():
    for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
      pair_count += 1
      gt_batch = gt_pair.unsqueeze(0).to(device=device)
      gt_pose, gt_seg = net(gt_batch)

      base = pred_pair.float()
      current_luma = torch.zeros((seq_len,), dtype=torch.float32)
      current_dy = torch.zeros((seq_len,), dtype=torch.float32)
      current_dx = torch.zeros((seq_len,), dtype=torch.float32)

      for _ in range(passes):
        for frame_idx in range(seq_len):
          cand = base.unsqueeze(0).repeat(len(luma_candidates), 1, 1, 1, 1)
          for current_frame_idx in range(seq_len):
            luma_values = current_luma[current_frame_idx].repeat(len(luma_candidates))
            dy_values = current_dy[current_frame_idx].repeat(len(luma_candidates))
            dx_values = current_dx[current_frame_idx].repeat(len(luma_candidates))
            if current_frame_idx == frame_idx:
              luma_values = luma_tensor
            apply_luma_shift_batch(cand, current_frame_idx, luma_values, dy_values, dx_values)
          score = metric_score(net, gt_pose, gt_seg, cand, device)
          current_luma[frame_idx] = luma_tensor[int(score.argmin().item())]

          cand = base.unsqueeze(0).repeat(len(shift_candidates), 1, 1, 1, 1)
          for current_frame_idx in range(seq_len):
            luma_values = current_luma[current_frame_idx].repeat(len(shift_candidates))
            dy_values = current_dy[current_frame_idx].repeat(len(shift_candidates))
            dx_values = current_dx[current_frame_idx].repeat(len(shift_candidates))
            if current_frame_idx == frame_idx:
              dy_values = shift_tensor
            apply_luma_shift_batch(cand, current_frame_idx, luma_values, dy_values, dx_values)
          score = metric_score(net, gt_pose, gt_seg, cand, device)
          current_dy[frame_idx] = shift_tensor[int(score.argmin().item())]

          cand = base.unsqueeze(0).repeat(len(shift_candidates), 1, 1, 1, 1)
          for current_frame_idx in range(seq_len):
            luma_values = current_luma[current_frame_idx].repeat(len(shift_candidates))
            dy_values = current_dy[current_frame_idx].repeat(len(shift_candidates))
            dx_values = current_dx[current_frame_idx].repeat(len(shift_candidates))
            if current_frame_idx == frame_idx:
              dx_values = shift_tensor
            apply_luma_shift_batch(cand, current_frame_idx, luma_values, dy_values, dx_values)
          score = metric_score(net, gt_pose, gt_seg, cand, device)
          current_dx[frame_idx] = shift_tensor[int(score.argmin().item())]

      for frame_idx in range(seq_len):
        all_values.extend(
          [
            current_luma[frame_idx].item(),
            current_dy[frame_idx].item(),
            current_dx[frame_idx].item(),
          ]
        )
      progress_interval = write_sidechannel.progress_interval
      if progress_interval > 0 and pair_count % progress_interval == 0:
        print(f"metric-y-shift searched {pair_count * seq_len} frames", flush=True)

  return MODE_Y_SHIFT, 3, all_values


def metric_y_sat_coordinate_values(
  src: Path,
  segment_dir: Path,
  luma_candidates: list[float],
  sat_candidates: list[int],
  device: torch.device,
  passes: int,
) -> tuple[int, int, list[float]]:
  from modules import DistortionNet, posenet_sd_path, segnet_sd_path

  net = DistortionNet().eval().to(device=device)
  net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
  luma_tensor = torch.tensor(luma_candidates, dtype=torch.float32)
  sat_tensor = torch.tensor(sat_candidates, dtype=torch.float32)
  all_values = []
  passes = max(1, passes)
  pair_count = 0

  with torch.inference_mode():
    for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
      pair_count += 1
      gt_batch = gt_pair.unsqueeze(0).to(device=device)
      gt_pose, gt_seg = net(gt_batch)

      base = pred_pair.float()
      current_luma = torch.zeros((seq_len,), dtype=torch.float32)
      current_sat = torch.zeros((seq_len,), dtype=torch.float32)

      for _ in range(passes):
        for frame_idx in range(seq_len):
          cand = base.unsqueeze(0).repeat(len(luma_candidates), 1, 1, 1, 1)
          for current_frame_idx in range(seq_len):
            luma_values = current_luma[current_frame_idx].repeat(len(luma_candidates))
            sat_values = current_sat[current_frame_idx].repeat(len(luma_candidates))
            if current_frame_idx == frame_idx:
              luma_values = luma_tensor
            apply_luma_sat_batch(cand, current_frame_idx, luma_values, sat_values)
          score = metric_score(net, gt_pose, gt_seg, cand, device)
          current_luma[frame_idx] = luma_tensor[int(score.argmin().item())]

          cand = base.unsqueeze(0).repeat(len(sat_candidates), 1, 1, 1, 1)
          for current_frame_idx in range(seq_len):
            luma_values = current_luma[current_frame_idx].repeat(len(sat_candidates))
            sat_values = current_sat[current_frame_idx].repeat(len(sat_candidates))
            if current_frame_idx == frame_idx:
              sat_values = sat_tensor
            apply_luma_sat_batch(cand, current_frame_idx, luma_values, sat_values)
          score = metric_score(net, gt_pose, gt_seg, cand, device)
          current_sat[frame_idx] = sat_tensor[int(score.argmin().item())]

      for frame_idx in range(seq_len):
        all_values.extend([current_luma[frame_idx].item(), current_sat[frame_idx].item()])
      progress_interval = write_sidechannel.progress_interval
      if progress_interval > 0 and pair_count % progress_interval == 0:
        print(f"metric-y-sat searched {pair_count * seq_len} frames", flush=True)

  return MODE_Y_SAT, 2, all_values


def metric_rgb_coordinate_values(
  src: Path,
  segment_dir: Path,
  candidates: list[float],
  device: torch.device,
  passes: int,
) -> tuple[int, int, list[float]]:
  from modules import DistortionNet, posenet_sd_path, segnet_sd_path

  net = DistortionNet().eval().to(device=device)
  net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
  all_values = []
  passes = max(1, passes)
  pair_count = 0

  with torch.inference_mode():
    for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
      pair_count += 1
      gt_batch = gt_pair.unsqueeze(0).to(device=device)
      gt_pose, gt_seg = net(gt_batch)

      base = pred_pair.float()
      current = torch.zeros((seq_len, 3), dtype=torch.float32)
      for _ in range(passes):
        for frame_idx in range(seq_len):
          for channel_idx in range(3):
            cand = base.unsqueeze(0).repeat(len(candidates), 1, 1, 1, 1)
            cand = cand + current.view(1, seq_len, 1, 1, 3)
            for idx, value in enumerate(candidates):
              cand[idx, frame_idx, :, :, channel_idx] = base[frame_idx, :, :, channel_idx] + value
            score = metric_score(net, gt_pose, gt_seg, cand, device)
            current[frame_idx, channel_idx] = candidates[int(score.argmin().item())]

      for frame_idx in range(seq_len):
        all_values.extend(current[frame_idx].tolist())
      progress_interval = write_sidechannel.progress_interval
      if progress_interval > 0 and pair_count % progress_interval == 0:
        print(f"metric-rgb-coordinate searched {pair_count * seq_len} frames", flush=True)

  return MODE_RGBMEAN, 3, all_values


def parse_metric_ybands(mode: str) -> tuple[int, bool] | None:
  prefixes = ("metric-ybands", "metric-yband")
  for prefix in prefixes:
    if not mode.startswith(prefix):
      continue
    tail = mode.removeprefix(prefix)
    fast = tail.endswith("-fast")
    tail = tail.removesuffix("-fast")
    tail = tail.removesuffix("-coordinate")
    tail = tail.removesuffix("-coord")
    bands = int(tail or "3")
    if bands <= 0 or bands > 32:
      raise SystemExit(f"unsupported metric ybands count: {bands}")
    return bands, fast
  return None


def add_band_biases(cand: torch.Tensor, current: torch.Tensor, band_edges: list[tuple[int, int]]) -> None:
  for frame_idx in range(seq_len):
    for band_idx, (y0, y1) in enumerate(band_edges):
      cand[:, frame_idx, y0:y1, :, :] += current[frame_idx, band_idx]


def metric_ybands_coordinate_values(
  src: Path,
  segment_dir: Path,
  candidates: list[float],
  device: torch.device,
  passes: int,
  bands: int,
) -> tuple[int, int, list[float]]:
  from modules import DistortionNet, posenet_sd_path, segnet_sd_path

  net = DistortionNet().eval().to(device=device)
  net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
  all_values = []
  passes = max(1, passes)
  pair_count = 0

  with torch.inference_mode():
    for gt_pair, pred_pair in iter_pairs(iter(iter_source_rgb(src)), iter(iter_baseline_frames(segment_dir))):
      pair_count += 1
      gt_batch = gt_pair.unsqueeze(0).to(device=device)
      gt_pose, gt_seg = net(gt_batch)

      base = pred_pair.float()
      band_edges = list(band_slices(base.shape[1], bands))
      current = torch.zeros((seq_len, bands), dtype=torch.float32)
      for _ in range(passes):
        for frame_idx in range(seq_len):
          for band_idx, (y0, y1) in enumerate(band_edges):
            cand = base.unsqueeze(0).repeat(len(candidates), 1, 1, 1, 1)
            add_band_biases(cand, current, band_edges)
            for idx, value in enumerate(candidates):
              cand[idx, frame_idx, y0:y1, :, :] += value - current[frame_idx, band_idx]
            score = metric_score(net, gt_pose, gt_seg, cand, device)
            current[frame_idx, band_idx] = candidates[int(score.argmin().item())]

      for frame_idx in range(seq_len):
        all_values.extend(current[frame_idx].tolist())
      progress_interval = write_sidechannel.progress_interval
      if progress_interval > 0 and pair_count % progress_interval == 0:
        print(f"metric-ybands{bands} searched {pair_count * seq_len} frames", flush=True)

  return MODE_YBANDS, bands, all_values


def iter_baseline_frames(segment_dir: Path):
  target_w, target_h = inflate.camera_size
  segment_paths = sorted(segment_dir.glob("*.ivf"))
  prev = None
  final_segment_idx = len(segment_paths) - 1
  for segment_idx, segment_path in enumerate(segment_paths):
    for frame_in_segment, rgb in enumerate(inflate.iter_segment_rgb(segment_path)):
      h, w, _ = rgb.shape
      if h != target_h or w != target_w:
        rgb = inflate.resize_rgb(rgb, target_w, target_h, segment_idx)
      rgb = inflate.postprocess_rgb(rgb, segment_idx)
      rgb = inflate.temporal_blend_rgb(rgb, prev, segment_idx, frame_in_segment, final_segment_idx)
      rgb = inflate.color_correct_rgb(rgb, segment_idx, frame_in_segment)
      prev = rgb
      yield rgb


def write_sidechannel(src: Path, segment_dir: Path, mode: str, gain: float, step: float) -> None:
  mode = mode.strip().lower()
  if not mode or mode == "none":
    return

  metric_ybands = parse_metric_ybands(mode)
  metric_ygrid = parse_metric_ygrid(mode)
  if mode in {
    "metric-y",
    "metric-y-shared",
    "metric-y-fast",
    "metric-y-shared-fast",
    "metric-y-coordinate",
    "metric-y-coord",
    "metric-y-coordinate-fast",
    "metric-y-coord-fast",
    "metric-y-gradient",
    "metric-y-grad",
    "metric-y-sat-coordinate",
    "metric-y-sat-coord",
    "metric-y-sat",
    "metric-y-shift-coordinate",
    "metric-y-shift-coord",
    "metric-y-shift",
    "metric-y-pose-fast",
    "metric-y-pose-shared-fast",
    "metric-y-pose-coordinate-fast",
    "metric-y-pose-coord-fast",
    "metric-yplane-fast",
    "metric-yplane-coordinate-fast",
    "metric-yplane-coord-fast",
    "metric-yplane-pose-fast",
    "metric-yplane-pose-coordinate-fast",
    "metric-yplane-pose-coord-fast",
    "metric-ybasis6-fast",
    "metric-ybasis6-coordinate-fast",
    "metric-ybasis6-coord-fast",
    "metric-ybasis6-pose-fast",
    "metric-ybasis6-pose-coordinate-fast",
    "metric-ybasis6-pose-coord-fast",
    "metric-yquad-fast",
    "metric-yquad-coordinate-fast",
    "metric-yquad-coord-fast",
    "metric-yquad-pose-fast",
    "metric-yquad-pose-coordinate-fast",
    "metric-yquad-pose-coord-fast",
    "metric-rgb-coordinate",
    "metric-rgb",
  } or metric_ybands or metric_ygrid:
    device = metric_device(write_sidechannel.metric_device)
    candidates = parse_candidates(write_sidechannel.candidates)
    if metric_ygrid:
      cols, rows, fast_ygrid = metric_ygrid
      if not fast_ygrid:
        raise SystemExit(f"metric ygrid currently requires -fast mode, got: {mode}")
      mode_id, channels, all_values = metric_ygrid_coordinate_fast_values(
        src,
        segment_dir,
        candidates,
        device,
        write_sidechannel.metric_passes,
        cols,
        rows,
      )
    elif metric_ybands:
      bands, fast_ybands = metric_ybands
      if fast_ybands:
        mode_id, channels, all_values = metric_ybands_coordinate_fast_values(
          src,
          segment_dir,
          candidates,
          device,
          write_sidechannel.metric_passes,
          bands,
        )
      else:
        mode_id, channels, all_values = metric_ybands_coordinate_values(
          src,
          segment_dir,
          candidates,
          device,
          write_sidechannel.metric_passes,
          bands,
        )
    elif mode in {"metric-y-fast", "metric-y-shared-fast"}:
      mode_id, channels, all_values = metric_y_fast_values(
        src,
        segment_dir,
        mode,
        candidates,
        device,
      )
    elif mode in {"metric-y-coordinate-fast", "metric-y-coord-fast"}:
      mode_id, channels, all_values = metric_y_coordinate_fast_values(
        src,
        segment_dir,
        candidates,
        device,
        write_sidechannel.metric_passes,
      )
    elif mode in {"metric-y-gradient", "metric-y-grad"}:
      mode_id, channels, all_values = metric_y_gradient_values(
        src,
        segment_dir,
        candidates,
        device,
      )
    elif mode in {"metric-y-sat-coordinate", "metric-y-sat-coord", "metric-y-sat"}:
      mode_id, channels, all_values = metric_y_sat_coordinate_values(
        src,
        segment_dir,
        candidates,
        parse_sat_candidates(),
        device,
        write_sidechannel.metric_passes,
      )
    elif mode in {"metric-y-shift-coordinate", "metric-y-shift-coord", "metric-y-shift"}:
      mode_id, channels, all_values = metric_y_shift_coordinate_values(
        src,
        segment_dir,
        candidates,
        parse_shift_candidates(),
        device,
        write_sidechannel.metric_passes,
      )
    elif mode in {"metric-y-pose-fast", "metric-y-pose-shared-fast"}:
      mode_id, channels, all_values = metric_y_pose_fast_values(
        src,
        segment_dir,
        mode,
        candidates,
        device,
      )
    elif mode in {"metric-y-pose-coordinate-fast", "metric-y-pose-coord-fast"}:
      mode_id, channels, all_values = metric_y_pose_coordinate_fast_values(
        src,
        segment_dir,
        candidates,
        device,
        write_sidechannel.metric_passes,
      )
    elif mode in {
      "metric-yplane-fast",
      "metric-yplane-coordinate-fast",
      "metric-yplane-coord-fast",
      "metric-ybasis6-fast",
      "metric-ybasis6-coordinate-fast",
      "metric-ybasis6-coord-fast",
      "metric-yquad-fast",
      "metric-yquad-coordinate-fast",
      "metric-yquad-coord-fast",
    }:
      mode_id, channels, all_values = metric_yplane_coordinate_fast_values(
        src,
        segment_dir,
        mode,
        candidates,
        device,
        write_sidechannel.metric_passes,
      )
    elif mode in {
      "metric-yplane-pose-fast",
      "metric-yplane-pose-coordinate-fast",
      "metric-yplane-pose-coord-fast",
      "metric-ybasis6-pose-fast",
      "metric-ybasis6-pose-coordinate-fast",
      "metric-ybasis6-pose-coord-fast",
      "metric-yquad-pose-fast",
      "metric-yquad-pose-coordinate-fast",
      "metric-yquad-pose-coord-fast",
    }:
      mode_id, channels, all_values = metric_yplane_pose_coordinate_fast_values(
        src,
        segment_dir,
        mode,
        candidates,
        device,
        write_sidechannel.metric_passes,
      )
    elif mode in {"metric-y-coordinate", "metric-y-coord"}:
      mode_id, channels, all_values = metric_y_coordinate_values(
        src,
        segment_dir,
        candidates,
        device,
        write_sidechannel.metric_passes,
      )
    elif mode in {"metric-rgb-coordinate", "metric-rgb"}:
      mode_id, channels, all_values = metric_rgb_coordinate_values(
        src,
        segment_dir,
        candidates,
        device,
        write_sidechannel.metric_passes,
      )
    else:
      mode_id, channels, all_values = metric_y_values(src, segment_dir, mode, candidates, device)
    frame_count = len(all_values)
    if channels:
      frame_count //= channels
    if mode_id in {MODE_Y_SAT, MODE_Y_SHIFT}:
      payload_bytes = bytearray()
      for idx, value in enumerate(all_values):
        if idx % channels == 0:
          q = int(round(value / step))
        else:
          q = int(round(value))
        q = max(-127, min(127, q))
        payload_bytes.append(q & 0xFF)
      payload = bytes(payload_bytes)
    else:
      payload = quantize(all_values, step)
    header = HEADER.pack(MAGIC, mode_id, channels, frame_count, step)
    (segment_dir / "sidechannel.bin").write_bytes(header + payload)
    counts = ", ".join(
      f"{value:g}:{count}" for value, count in sorted(collections.Counter(all_values).items())
    )
    print(
      f"sidechannel {mode}: {frame_count} frames, {channels} channel(s), "
      f"{len(header) + len(payload)} bytes raw, device={device}, values={counts}"
    )
    return

  all_values = []
  mode_id = None
  channels = None
  frame_count = 0

  for gt, pred in zip(iter_source_rgb(src), iter_baseline_frames(segment_dir), strict=True):
    current_mode_id, values = correction_values(mode, gt, pred, gain)
    if mode_id is None:
      mode_id = current_mode_id
      channels = len(values)
    elif current_mode_id != mode_id or len(values) != channels:
      raise SystemExit("sidechannel mode changed while generating metadata")
    all_values.extend(values)
    frame_count += 1

  if mode_id is None or channels is None:
    raise SystemExit(f"no frames available for sidechannel: {src}")

  payload = quantize(all_values, step)
  header = HEADER.pack(MAGIC, mode_id, channels, frame_count, step)
  (segment_dir / "sidechannel.bin").write_bytes(header + payload)
  print(
    f"sidechannel {mode}: {frame_count} frames, {channels} channel(s), "
    f"{len(header) + len(payload)} bytes raw"
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", required=True)
  parser.add_argument("--gain", type=float, default=1.0)
  parser.add_argument("--step", type=float, default=0.5)
  parser.add_argument("--candidates", default="-4,-2,0,2,4")
  parser.add_argument("--metric-device", default="auto")
  parser.add_argument("--metric-passes", type=int, default=1)
  parser.add_argument("--progress-interval", type=int, default=100)
  parser.add_argument("source", type=Path)
  parser.add_argument("segment_dir", type=Path)
  args = parser.parse_args()
  write_sidechannel.candidates = args.candidates
  write_sidechannel.metric_device = args.metric_device
  write_sidechannel.metric_passes = args.metric_passes
  write_sidechannel.progress_interval = args.progress_interval
  configure_torch_threads()
  write_sidechannel(args.source, args.segment_dir, args.mode, args.gain, args.step)


if __name__ == "__main__":
  main()
