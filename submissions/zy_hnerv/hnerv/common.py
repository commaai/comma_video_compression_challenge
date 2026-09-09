"""Shared pieces: repo bootstrap, differentiable yuv6 patch, resize operator, model, codec."""
from __future__ import annotations
import io, math, os, struct, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Portable paths.  Default layout: this package sits in <repo>/hnerv/, so the
# challenge repo is one level up.  Override either with an env var.
HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("COMMA_REPO", HERE.parent))
WORK = Path(os.environ.get("COMMA_WORK", HERE))
CACHE = WORK / "cache"
sys.path.insert(0, str(REPO))

EVAL_H, EVAL_W = 384, 512          # what SegNet/PoseNet actually consume
CAM_H, CAM_W = 874, 1164           # what the .raw file must contain
N_PAIRS_EXPECTED = 600


# --------------------------------------------------------------------------
# differentiable rgb_to_yuv6
#
# frame_utils.rgb_to_yuv6 is @torch.no_grad() + in-place clamp_, so PoseNet's
# preprocess_input silently kills the gradient path.  Patch both module refs
# before anything imports them.
# --------------------------------------------------------------------------
def _rgb_to_yuv6_diff(rgb_chw: torch.Tensor) -> torch.Tensor:
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, : 2 * H2, : 2 * W2]
    R, G, B = rgb[..., 0, :, :], rgb[..., 1, :, :], rgb[..., 2, :, :]
    Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
    U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2]
             + U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
    V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2]
             + V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
    return torch.stack([Y[..., 0::2, 0::2], Y[..., 1::2, 0::2],
                        Y[..., 0::2, 1::2], Y[..., 1::2, 1::2],
                        U_sub, V_sub], dim=-3)


def load_distortion_net(device="cpu", differentiable=True):
    import frame_utils, modules
    if differentiable:
        frame_utils.rgb_to_yuv6 = _rgb_to_yuv6_diff
        modules.rgb_to_yuv6 = _rgb_to_yuv6_diff
    from modules import DistortionNet, segnet_sd_path, posenet_sd_path
    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# --------------------------------------------------------------------------
# the bicubic-up / bilinear-down round trip, as two small dense matrices
#
# inflate writes 874x1164, the harness bilinear-downsamples back to 384x512.
# Doing that literally costs a 3M-pixel bicubic every step.  Both resamplers
# are separable and linear, so the composite is  X -> Ah @ X @ Aw^T  with
# Ah (384x384) and Aw (512x512).  Two small matmuls instead of 3M pixels.
# --------------------------------------------------------------------------
def _axis_operator(n_small: int, n_big: int, other: int = 8) -> torch.Tensor:
    """Column j of the returned matrix is the round trip applied to basis vector j."""
    eye = torch.eye(n_small, dtype=torch.float32).view(n_small, 1, n_small, 1)
    eye = eye.expand(n_small, 1, n_small, other).contiguous()
    up = F.interpolate(eye, size=(n_big, other), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(n_small, other), mode="bilinear", align_corners=False)
    # constant along the 'other' axis, so any column works
    return down[:, 0, :, other // 2].clone()          # (n_small_in, n_small_out)


def build_roundtrip(cache=True):
    p = CACHE / "roundtrip.pt"
    if cache and p.exists():
        d = torch.load(p)
        return d["Ah"], d["Aw"]
    Ah = _axis_operator(EVAL_H, CAM_H).T.contiguous()   # (out_h, in_h)
    Aw = _axis_operator(EVAL_W, CAM_W).T.contiguous()   # (out_w, in_w)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"Ah": Ah, "Aw": Aw}, p)
    return Ah, Aw


def apply_roundtrip(x, Ah, Aw):
    """x: (..., H, W) -> same shape, matching inflate-then-harness resampling."""
    return torch.einsum("ij,...jk,lk->...il", Ah, x, Aw)


# --------------------------------------------------------------------------
# decoder
# --------------------------------------------------------------------------
class Decoder(nn.Module):
    """HNeRV-style: latent -> small grid -> pixelshuffle stages -> RGB frame pair.

    Three changes from the reference, all paid for by the profiler:

    * base grid 3x4 with 7 stages instead of 6x8 with 6.  The stem Linear is
      latent_dim x (C0 * base_h * base_w) and was the single largest parameter
      block (245K of 435K at latent 128); a 4x smaller grid shrinks it 4x for
      negligible extra compute, since the added stage runs at 3x4.
    * no full-resolution `refine` block.  Two 384x512 convs plus a sin cost
      ~40% of decoder time for a residual polish I cannot afford.
    * a shared learned low-res `base` image added to the RGB logits, so the
      static parts of a dashcam scene (hood, dark sky, road geometry) do not
      have to be learned through the whole stack.
    """

    def __init__(self, latent_dim=96, base_channels=40, n_stages=7,
                 taper=(1.0, 1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.3),
                 base_grid=(3, 4), base_img=(48, 64)):
        super().__init__()
        C = base_channels
        self.latent_dim = latent_dim
        self.n_stages = n_stages
        self.base_h, self.base_w = base_grid
        self.channels = [max(4, int(round(C * t))) for t in taper[: n_stages + 1]]

        self.stem = nn.Linear(latent_dim, self.channels[0] * self.base_h * self.base_w)
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(n_stages):
            cin, cout = self.channels[i], self.channels[i + 1]
            self.blocks.append(nn.Conv2d(cin, cout * 4, 3, padding=1))
            self.skips.append(nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity())
        self.ps = nn.PixelShuffle(2)

        fc = self.channels[-1]
        self.base = nn.Parameter(torch.zeros(1, 3, *base_img))
        self.rgb_0 = nn.Conv2d(fc, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(fc, 3, 3, padding=1)

    @property
    def render_size(self):
        return self.base_h * 2 ** self.n_stages, self.base_w * 2 ** self.n_stages

    def forward(self, z, amp=False):
        """amp: run the expensive trunk in bf16 (AMX) but keep the RGB heads in
        fp32.  bf16 output would quantise pixels to ~1 LSB, which PoseNet — the
        sensitive term — cannot afford; the heads are cheap so this costs little."""
        B = z.shape[0]
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(amp)):
            x = torch.sin(self.stem(z).view(B, self.channels[0], self.base_h, self.base_w))
            for block, skip in zip(self.blocks, self.skips):
                idt = skip(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
                x = torch.sin(self.ps(block(x)) + idt)
        x = x.float()
        h, w = x.shape[-2:]
        base = F.interpolate(self.base, size=(h, w), mode="bilinear", align_corners=False)
        f0 = torch.sigmoid(self.rgb_0(x) + base) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x) + base) * 255.0
        out = torch.stack([f0, f1], dim=1)                     # (B,2,3,h,w)
        h, w = out.shape[-2:]
        if (h, w) != (EVAL_H, EVAL_W):
            out = F.interpolate(out.flatten(0, 1), size=(EVAL_H, EVAL_W),
                                mode="bilinear", align_corners=False).view(B, 2, 3, EVAL_H, EVAL_W)
        return out


# --------------------------------------------------------------------------
# codec: INT8 weights (zigzag + brotli), per-dim delta-coded uint8 latents
# --------------------------------------------------------------------------
def _zig8(a):
    a = a.astype(np.int32)
    return np.where(a >= 0, 2 * a, -2 * a - 1).astype(np.uint8)


def _unzig8(a):
    a = a.astype(np.int32)
    return np.where(a % 2 == 0, a // 2, -(a // 2) - 1).astype(np.int8)


def encode_weights(sd, n_levels=127):
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(sd)))
    for name, t in sd.items():
        t = t.detach().cpu().float()
        m = t.abs().max().item()
        scale = m / n_levels if m > 0 else 1.0
        q = (t / scale).round().clamp(-n_levels, n_levels).to(torch.int8).numpy().ravel()
        nb = name.encode()
        buf.write(struct.pack("<I", len(nb))); buf.write(nb)
        buf.write(struct.pack("<I", len(t.shape)))
        for s in t.shape:
            buf.write(struct.pack("<I", s))
        buf.write(struct.pack("<f", scale))
        buf.write(struct.pack("<I", q.size))
        buf.write(_zig8(q).tobytes())
    import brotli
    return brotli.compress(buf.getvalue(), quality=11)


def decode_weights(blob):
    import brotli
    buf = io.BytesIO(brotli.decompress(blob))
    n = struct.unpack("<I", buf.read(4))[0]
    sd = {}
    for _ in range(n):
        nl = struct.unpack("<I", buf.read(4))[0]
        name = buf.read(nl).decode()
        nd = struct.unpack("<I", buf.read(4))[0]
        shape = tuple(struct.unpack("<I", buf.read(4))[0] for _ in range(nd))
        scale = struct.unpack("<f", buf.read(4))[0]
        size = struct.unpack("<I", buf.read(4))[0]
        q = _unzig8(np.frombuffer(buf.read(size), dtype=np.uint8))
        sd[name] = torch.from_numpy(q.astype(np.float32).reshape(shape)) * scale
    return sd


def encode_latents(lat, levels=255):
    t = lat.detach().cpu().float()
    n, d = t.shape
    mins, maxs = t.min(0).values, t.max(0).values
    scales = ((maxs - mins) / levels).clamp(min=1e-10)
    q = ((t - mins) / scales).round().clamp(0, levels).to(torch.uint8).numpy()
    delta = np.empty_like(q, dtype=np.int16)
    delta[0] = q[0]
    delta[1:] = q[1:].astype(np.int16) - q[:-1].astype(np.int16)
    zz = np.where(delta >= 0, 2 * delta, -2 * delta - 1).astype(np.uint16)
    payload = struct.pack("<II", n, d)
    payload += mins.to(torch.float16).numpy().tobytes()
    payload += scales.to(torch.float16).numpy().tobytes()
    payload += (zz & 0xFF).astype(np.uint8).tobytes() + (zz >> 8).astype(np.uint8).tobytes()
    import brotli
    return brotli.compress(payload, quality=11)


def decode_latents(blob):
    import brotli
    buf = io.BytesIO(brotli.decompress(blob))
    n, d = struct.unpack("<II", buf.read(8))
    mins = torch.from_numpy(np.frombuffer(buf.read(d * 2), dtype=np.float16).copy()).float()
    scales = torch.from_numpy(np.frombuffer(buf.read(d * 2), dtype=np.float16).copy()).float()
    tot = n * d
    lo = np.frombuffer(buf.read(tot), dtype=np.uint8).astype(np.uint16)
    hi = np.frombuffer(buf.read(tot), dtype=np.uint8).astype(np.uint16)
    zz = ((hi << 8) | lo).reshape(n, d).astype(np.int32)
    delta = np.where(zz % 2 == 0, zz // 2, -(zz // 2) - 1).astype(np.int32)
    q = np.cumsum(delta, axis=0).astype(np.float32)
    return torch.from_numpy(q) * scales + mins


def build_archive(sd, latents, meta):
    import json, brotli
    mb = brotli.compress(json.dumps(meta).encode(), quality=11)
    wb = encode_weights(sd)
    lb = encode_latents(latents)
    out = io.BytesIO()
    for part in (mb, wb, lb):
        out.write(struct.pack("<I", len(part))); out.write(part)
    return out.getvalue()


def parse_archive(blob):
    import json, brotli
    buf = io.BytesIO(blob)
    parts = []
    for _ in range(3):
        n = struct.unpack("<I", buf.read(4))[0]
        parts.append(buf.read(n))
    meta = json.loads(brotli.decompress(parts[0]))
    return decode_weights(parts[1]), decode_latents(parts[2]), meta


def score_of(seg, pose, archive_bytes, total_bytes=37545489):
    return 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * archive_bytes / total_bytes
