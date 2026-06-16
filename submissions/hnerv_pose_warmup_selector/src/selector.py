#!/usr/bin/env python
"""Per-pair perturbation selector — a post-hoc, decoder-agnostic distortion reducer.

Idea (from the #1 leaderboard solution, reimplemented): a decoded frame pair has a tiny
systematic error the scorer nets are sensitive to. For each pair we try a palette of ~31
cheap transforms, keep the one that MOST reduces that pair's SegNet+PoseNet distortion, and
store just its index (~5 bits/pair ≈ <1KB total). The decoder applies the chosen transform
before writing the .raw. Works on top of any model with no retraining.

The transforms are deliberately tiny: small luma/chroma biases, single-pixel rolls, a faint
chroma checkerboard. They nudge the rendered frame back toward where the nets agree with GT.
"""
import torch

# A transform is a callable (pair[2,3,H,W] float) -> pair[2,3,H,W] float. Applied per pair,
# always clamped to [0,255]. Mode 0 is identity. The palette is fixed/deterministic so the
# decoder reconstructs it from the stored index alone.


def _clamp(x):
  return x.clamp(0, 255)


def _luma_bias(delta, which=0):
  def f(p):
    out = p.clone()
    out[which] = _clamp(out[which] + delta)
    return out
  return f


def _rgb_bias(dr, dg, db, which=0):
  def f(p):
    out = p.clone()
    out[which, 0] = _clamp(out[which, 0] + dr)
    out[which, 1] = _clamp(out[which, 1] + dg)
    out[which, 2] = _clamp(out[which, 2] + db)
    return out
  return f


def _roll(dy, dx, which=0):
  def f(p):
    out = p.clone()
    out[which] = torch.roll(out[which], shifts=(dy, dx), dims=(-2, -1))
    return out
  return f


def _blue_tile(amp, which=0):
  """Add a faint 8x8 ±1 checkerboard to blue, subtract from red (mirrors the winner's _blue_tile)."""
  def f(p):
    out = p.clone()
    H, W = out.shape[-2], out.shape[-1]
    yy = torch.arange(H).view(H, 1)
    xx = torch.arange(W).view(1, W)
    checker = (((yy // 8) + (xx // 8)) % 2).float() * 2 - 1  # ±1 per 8x8 block
    checker = checker.to(out.device) * amp
    out[which, 2] = _clamp(out[which, 2] + checker)
    out[which, 0] = _clamp(out[which, 0] - checker)
    return out
  return f


def build_palette():
  """The fixed transform palette. Index 0 = identity. ~31 modes total."""
  palette = [lambda p: p.clone()]  # 0: identity
  # uniform luma bias on frame 0 and frame 1
  for d in (-4, -2, -1, 1, 2, 4):
    palette.append(_luma_bias(d, which=0))
  for d in (-2, -1, 1, 2):
    palette.append(_luma_bias(d, which=1))
  # per-channel RGB nudges on frame 0
  for dr, dg, db in [(0, -1, 1), (0, 1, -1), (1, 0, -1), (-1, 0, 1), (2, -1, -1), (-2, 1, 1)]:
    palette.append(_rgb_bias(dr, dg, db, which=0))
  # single-pixel rolls on frame 0 and frame 1
  for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
    palette.append(_roll(dy, dx, which=0))
  for dy, dx in [(0, 1), (1, 0)]:
    palette.append(_roll(dy, dx, which=1))
  # faint chroma checkerboard
  for amp in (1, 2, 3):
    palette.append(_blue_tile(amp, which=0))
  for amp in (1, 2):
    palette.append(_blue_tile(amp, which=1))
  return palette


PALETTE = build_palette()
N_MODES = len(PALETTE)


def apply_mode(pair, mode):
  """pair: (2,3,H,W) float [0,255]. Returns transformed pair."""
  return _clamp(PALETTE[mode](pair))
