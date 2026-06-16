#!/usr/bin/env python
"""Inflate the HNeRV archive into a .raw frame dump.

Self-contained: rebuilds the trained HNeRV decoder from the extracted archive/ (meta.json +
weights.br), renders every frame, applies the per-pair perturbation selector if present, and
writes flat uint8 RGB (N, 874, 1164, 3) — what the evaluator expects. The SRC video is unused
(the archive is the full compressed representation). Runs on CPU (~1 min) or GPU.

Usage: python -m submissions.hnerv.inflate <archive_dir> <dst.raw>
"""
import sys
import torch
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.codec import decode          # noqa: E402
from src.selector import apply_mode   # noqa: E402


def pick_device():
  if torch.cuda.is_available(): return torch.device("cuda")
  if torch.backends.mps.is_available(): return torch.device("mps")
  return torch.device("cpu")


def _load_selector(archive_dir):
  sel = Path(archive_dir) / "selector.bin"
  return list(sel.read_bytes()) if sel.exists() else None


def inflate(archive_dir: str, dst: str, batch: int = 16):
  device = pick_device()
  model = decode(archive_dir).to(device)
  n = model.embeddings.shape[0]
  modes = _load_selector(archive_dir)

  if modes is None:
    with torch.inference_mode(), open(dst, "wb") as f:
      for start in range(0, n, batch):
        idx = torch.arange(start, min(start + batch, n), device=device)
        x = model.render_native(idx).clamp(0, 255).round().to(torch.uint8)
        f.write(x.permute(0, 2, 3, 1).contiguous().cpu().numpy().tobytes())
    return n

  # selector path: render per pair, apply the stored transform, write both frames.
  n_pairs = n // 2
  with torch.inference_mode(), open(dst, "wb") as f:
    for p in range(n_pairs):
      idx = torch.tensor([2 * p, 2 * p + 1], device=device)
      pair = model.render_native(idx).clamp(0, 255)
      pair = apply_mode(pair, modes[p] if p < len(modes) else 0)
      x = pair.round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
      f.write(x.tobytes())
    if n % 2:
      idx = torch.tensor([n - 1], device=device)
      x = model.render_native(idx).clamp(0, 255).round().to(torch.uint8)
      f.write(x.permute(0, 2, 3, 1).contiguous().cpu().numpy().tobytes())
  return n


if __name__ == "__main__":
  archive_dir, dst = sys.argv[1], sys.argv[2]
  n = inflate(archive_dir, dst)
  print(f"rendered {n} frames -> {dst}")
