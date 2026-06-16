#!/usr/bin/env python
"""HNeRV-style decoder for the comma compression challenge.

A per-frame embedding is decoded by a shared conv-upsample stack into a frame.
The whole model (quantized embeddings + decoder weights) IS the compressed archive.

Design constraints (see DESIGN.md, validated by probe.py 2026-06-14):
- Tiny per-frame embeddings (most capacity in the shared decoder).
- Decode at NEAR-NATIVE internal resolution (>=0.75x of 1164x874). Probe proved that
  decoding at 512x384 and upscaling costs ~0.31 distortion — the GT path keeps native
  high-freq detail a low-res recon never had. Byte budget is met by weight quantization +
  entropy coding (Stage 4), NOT by shrinking resolution.
- Everything differentiable so SegNet/PoseNet distortion can be backpropped (Stage 3).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# native output the .raw must contain
NATIVE_W, NATIVE_H = 1164, 874


class UpBlock(nn.Module):
  """Upsample by `stride` via PixelShuffle, then GroupNorm + conv + activation.

  PixelShuffle is parameter-cheaper than ConvTranspose for the same upsampling and
  avoids checkerboard artifacts that could create spurious class edges for SegNet.

  GroupNorm is ESSENTIAL: without it, stacked conv+GELU layers compound the activation
  scale (~5-10x per block) until the head saturates the output sigmoid to dead-zero and
  training collapses. GroupNorm (batch-independent, good for overfitting one video) keeps
  activations bounded so the network actually learns.
  """
  def __init__(self, in_ch, out_ch, stride=2, groups=8):
    super().__init__()
    self.stride = stride
    self.conv = nn.Conv2d(in_ch, out_ch * stride * stride, kernel_size=3, padding=1)
    # GroupNorm needs num_groups to divide out_ch; pick the largest divisor <= groups.
    ng = next(g for g in range(min(groups, out_ch), 0, -1) if out_ch % g == 0)
    self.norm = nn.GroupNorm(num_groups=ng, num_channels=out_ch)
    self.act = nn.GELU()

  def forward(self, x):
    x = self.conv(x)
    x = F.pixel_shuffle(x, self.stride)
    x = self.norm(x)
    return self.act(x)


class HNeRVDecoder(nn.Module):
  """Maps a per-frame SPATIAL embedding seed[B, seed_ch, h0, w0] -> frame[B, 3, H_int, W_int].

  Unlike a flat E-vector + Linear (which bottlenecks all frame content through E scalars),
  the embedding is itself a small feature map. This is the real HNeRV design and is what
  lets the model actually FIT 1200 distinct high-res frames. The seed stays small+compressible
  (spatially smooth, temporally correlated) so it entropy-codes well.

  seed_hw: spatial size of the embedding seed (h0, w0).
  channels: channel width at each stage, len(channels) == len(strides)+1. channels[0]=seed_ch.
  strides: per-stage upsampling factors; product * seed_hw == internal resolution.
  """
  def __init__(self, seed_hw=(11, 14), channels=(64, 48, 32, 16, 8),
               strides=(3, 3, 3), internal_hw=None):
    super().__init__()
    assert len(channels) == len(strides) + 1, "channels must be one longer than strides"
    self.seed_hw = seed_hw
    self.seed_ch = channels[0]
    self.strides = strides

    blocks = []
    for i, s in enumerate(strides):
      blocks.append(UpBlock(channels[i], channels[i + 1], stride=s))
    self.blocks = nn.ModuleList(blocks)
    self.head = nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1)
    # Start the output near the (dark) data mean: frames avg ~20/255≈0.08, and
    # sigmoid(-2.4)≈0.08. Small head weights + this bias => sane initial output, no
    # dead-sigmoid collapse before GroupNorm has a chance to keep activations bounded.
    nn.init.zeros_(self.head.weight)
    nn.init.constant_(self.head.bias, -2.4)

    h0, w0 = seed_hw
    sh = h0 * math.prod(strides)
    sw = w0 * math.prod(strides)
    self.internal_hw = internal_hw or (sh, sw)

  def forward(self, seed):
    """seed: (B, seed_ch, h0, w0)."""
    x = seed
    for blk in self.blocks:
      x = blk(x)
    x = self.head(x)                     # (B, 3, sh, sw), logits-ish
    return x

  def render_native(self, seed):
    """Decode and resize to native (1164x874) RGB in [0,255] float. For inflate."""
    x = self.forward(seed)
    x = torch.sigmoid(x) * 255.0
    x = F.interpolate(x, size=(NATIVE_H, NATIVE_W), mode="bilinear", align_corners=False)
    return x  # (B, 3, NATIVE_H, NATIVE_W)

  def param_count(self):
    return sum(p.numel() for p in self.parameters())


class HNeRV(nn.Module):
  """Full model: per-frame SPATIAL embedding table + shared decoder.

  embeddings: (n_frames, seed_ch, h0, w0) — one small feature map per frame.
  """
  def __init__(self, n_frames, seed_hw=(11, 14), channels=(64, 48, 32, 16, 8),
               strides=(3, 3, 3, 3), **dec_kwargs):
    super().__init__()
    seed_ch = channels[0]
    h0, w0 = seed_hw
    self.embeddings = nn.Parameter(torch.randn(n_frames, seed_ch, h0, w0) * 0.1)
    self.decoder = HNeRVDecoder(seed_hw=seed_hw, channels=channels, strides=strides, **dec_kwargs)

  def forward(self, idx):
    return self.decoder(self.embeddings[idx])

  def render_native(self, idx):
    return self.decoder.render_native(self.embeddings[idx])

  def budget_report(self):
    dec = self.decoder.param_count()
    emb = self.embeddings.numel()
    n_frames = self.embeddings.shape[0]
    return {
      "decoder_params": dec,
      "embedding_params": emb,
      "embed_per_frame": emb // n_frames,
      "total_params": dec + emb,
      "bytes_at_4bit": (dec + emb) * 0.5,
      "internal_hw": self.decoder.internal_hw,
    }


if __name__ == "__main__":
  # sanity: full-native-res config with spatial embeddings, print the budget
  m = HNeRV(n_frames=1200, seed_hw=(11, 14), channels=(64, 48, 32, 16, 8),
            strides=(3, 3, 3, 3))
  rep = m.budget_report()
  for k, v in rep.items():
    print(f"  {k}: {v}")
  x = m.render_native(torch.arange(4))
  print(f"  render_native(4 frames) -> {tuple(x.shape)}  range[{x.min():.1f},{x.max():.1f}]")
