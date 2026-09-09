#!/usr/bin/env python
"""Self-contained decoder: archive bytes -> raw uint8 RGB frames at 874x1164.

No dependency on the training tree; only torch/numpy/brotli.
"""
import io, json, struct, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(__import__("os").environ.get("INFLATE_THREADS", "4")))

EVAL_H, EVAL_W = 384, 512
CAM_H, CAM_W = 874, 1164


class Decoder(nn.Module):
    def __init__(self, latent_dim=96, base_channels=40, n_stages=7,
                 taper=(1.0, 1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.3),
                 base_grid=(3, 4), base_img=(48, 64)):
        super().__init__()
        C = base_channels
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

    def forward(self, z):
        B = z.shape[0]
        x = torch.sin(self.stem(z).view(B, self.channels[0], self.base_h, self.base_w))
        for block, skip in zip(self.blocks, self.skips):
            idt = skip(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
            x = torch.sin(self.ps(block(x)) + idt)
        h, w = x.shape[-2:]
        base = F.interpolate(self.base, size=(h, w), mode="bilinear", align_corners=False)
        f0 = torch.sigmoid(self.rgb_0(x) + base) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x) + base) * 255.0
        return torch.stack([f0, f1], dim=1)


def _unzig8(a):
    a = a.astype(np.int32)
    return np.where(a % 2 == 0, a // 2, -(a // 2) - 1).astype(np.int8)


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


def parse_archive(blob):
    import brotli
    buf = io.BytesIO(blob)
    parts = []
    for _ in range(3):
        n = struct.unpack("<I", buf.read(4))[0]
        parts.append(buf.read(n))
    meta = json.loads(brotli.decompress(parts[0]))
    return decode_weights(parts[1]), decode_latents(parts[2]), meta


@torch.inference_mode()
def main(archive_path, dst, batch=8):
    blob = Path(archive_path).read_bytes()
    sd, lat, meta = parse_archive(blob)
    dec = Decoder(latent_dim=meta["latent_dim"], base_channels=meta["base_channels"]).eval()
    dec.load_state_dict(sd)
    n_pairs = lat.shape[0]
    written = 0
    with open(dst, "wb") as f:
        for s in range(0, n_pairs, batch):
            z = lat[s:s + batch]
            out = dec(z)                                   # (b,2,3,384,512)
            b = out.shape[0]
            up = F.interpolate(out.reshape(b * 2, 3, EVAL_H, EVAL_W),
                               size=(CAM_H, CAM_W), mode="bicubic", align_corners=False)
            up = up.clamp(0, 255).round().to(torch.uint8)   # (b*2,3,H,W)
            f.write(up.permute(0, 2, 3, 1).contiguous().numpy().tobytes())
            written += b * 2
    print(f"wrote {written} frames to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
