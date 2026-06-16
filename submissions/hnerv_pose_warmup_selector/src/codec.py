#!/usr/bin/env python
"""Self-contained decode side of the HNeRV codec (for inflation).

Reads an extracted archive/ (meta.json + weights.br) and rebuilds the HNeRV model. Vendored
into the submission so inflation has no dependency on the training code at repo root.
"""
import json
import numpy as np
import torch
import brotli
from pathlib import Path

from .model import HNeRV

# Architecture presets (must match the trainer that produced the archive).
PRESETS = {
  "sc4_5x": dict(seed_hw=(4, 5), channels=(4, 48, 40, 32, 24, 16), strides=(3, 3, 3, 3, 2)),
  "sc6_5x": dict(seed_hw=(4, 5), channels=(6, 56, 48, 40, 32, 16), strides=(3, 3, 3, 3, 2)),
  "big":   dict(seed_hw=(11, 14), channels=(16, 128, 96, 64, 32), strides=(3, 3, 3, 3)),
}


def build_model_from_config(cfg) -> HNeRV:
  if "preset" in cfg:
    return HNeRV(n_frames=cfg["n_frames"], **PRESETS[cfg["preset"]])
  raise ValueError(f"unsupported config (expected a preset): {cfg}")


def _dequantize(q: np.ndarray, scale: float, shape) -> torch.Tensor:
  return torch.from_numpy(q.astype(np.float32) * scale).reshape(shape)


def decode(archive_dir) -> HNeRV:
  """Load an extracted archive/ dir (meta.json + weights.br) into a ready HNeRV."""
  archive_dir = Path(archive_dir)
  meta = json.loads((archive_dir / "meta.json").read_text())
  blob = brotli.decompress((archive_dir / "weights.br").read_bytes())
  model = build_model_from_config(meta["config"])
  new_sd = {}
  for e in meta["tensors"]:
    raw = blob[e["offset"]: e["offset"] + e["nbytes"]]
    q = np.frombuffer(raw, dtype=np.int8)
    new_sd[e["name"]] = _dequantize(q, e["scale"], e["shape"])
  model.load_state_dict(new_sd)
  model.eval()
  return model
