# SPDX-License-Identifier: MIT
"""Self-contained codec for {decoder_sd, latents, meta}.

Decoder: per-tensor symmetric int8 + zigzag + brotli (the regime shown optimal in
WRITEUP.md). Latents: per-dim uint8 + temporal delta + zigzag + lo/hi split + brotli
(byte-identical scheme to PR #95's codec). Round-trip is exact for the int8 weights.
"""
import io
import json
import struct

import brotli
import numpy as np
import torch


# ---- latents (scheme from PR #95) -----------------------------------------
def encode_latents(latents):
    t = latents.detach().cpu().float()
    n, d = t.shape
    mins = t.min(dim=0).values
    maxs = t.max(dim=0).values
    scales = ((maxs - mins) / 254.0).clamp(min=1e-10)
    q = ((t - mins.unsqueeze(0)) / scales.unsqueeze(0)).round().clamp(0, 254).to(torch.uint8).numpy()
    delta = np.empty_like(q, dtype=np.int16)
    delta[0] = q[0]
    delta[1:] = q[1:].astype(np.int16) - q[:-1].astype(np.int16)
    zz = np.where(delta >= 0, 2 * delta, -2 * delta - 1).astype(np.uint16)
    payload = struct.pack("<II", n, d)
    payload += mins.to(torch.float16).numpy().tobytes()
    payload += scales.to(torch.float16).numpy().tobytes()
    payload += (zz & 0xFF).astype(np.uint8).tobytes()
    payload += (zz >> 8).astype(np.uint8).tobytes()
    return brotli.compress(payload, quality=11)


def decode_latents(blob):
    buf = io.BytesIO(brotli.decompress(blob))
    n, d = struct.unpack("<II", buf.read(8))
    mins = torch.from_numpy(np.frombuffer(buf.read(d * 2), dtype=np.float16).copy()).float()
    scales = torch.from_numpy(np.frombuffer(buf.read(d * 2), dtype=np.float16).copy()).float()
    total = n * d
    lo = np.frombuffer(buf.read(total), dtype=np.uint8).astype(np.uint16)
    hi = np.frombuffer(buf.read(total), dtype=np.uint8).astype(np.uint16)
    zz = ((hi << 8) | lo).reshape(n, d)
    delta = np.where(zz % 2 == 0, zz.astype(np.int32) // 2, -(zz.astype(np.int32) // 2) - 1).astype(np.int16)
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, n):
        q[i] = q[i - 1] + delta[i]
    q = q.astype(np.uint8)
    return torch.from_numpy(q.astype(np.float32)) * scales.unsqueeze(0) + mins.unsqueeze(0)


# ---- decoder (per-tensor int8 + zigzag + brotli) --------------------------
def encode_decoder(sd):
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(sd)))
    for name, t in sd.items():
        t = t.detach().cpu().float()
        m = t.abs().max().item()
        s = m / 127 if m > 0 else 1.0
        q = (t / s).round().clamp(-127, 127).to(torch.int8).numpy().flatten()
        nb = name.encode()
        buf.write(struct.pack("<I", len(nb))); buf.write(nb)
        buf.write(struct.pack("<I", t.ndim))
        for dim in t.shape:
            buf.write(struct.pack("<I", dim))
        buf.write(struct.pack("<f", s)); buf.write(struct.pack("<I", q.size))
        zz = np.where(q.astype(np.int32) >= 0, 2 * q, -2 * q - 1).astype(np.uint8)
        buf.write(zz.tobytes())
    return brotli.compress(buf.getvalue(), quality=11)


def decode_decoder(blob):
    raw = brotli.decompress(blob)
    buf = io.BytesIO(raw)
    n = struct.unpack("<I", buf.read(4))[0]
    sd = {}
    for _ in range(n):
        nl = struct.unpack("<I", buf.read(4))[0]
        name = buf.read(nl).decode()
        nd = struct.unpack("<I", buf.read(4))[0]
        shape = tuple(struct.unpack("<I", buf.read(4))[0] for _ in range(nd))
        s = struct.unpack("<f", buf.read(4))[0]
        size = struct.unpack("<I", buf.read(4))[0]
        zz = np.frombuffer(buf.read(size), dtype=np.uint8).astype(np.int32)
        q = np.where(zz % 2 == 0, zz // 2, -(zz // 2) - 1).astype(np.int8)
        sd[name] = torch.from_numpy(q.astype(np.float32).reshape(shape)) * s
    return sd


# ---- top-level payload -----------------------------------------------------
def build_payload(decoder_sd, latents, meta):
    meta_b = json.dumps(meta).encode()
    dec = encode_decoder(decoder_sd)
    lat = encode_latents(latents)
    out = io.BytesIO()
    out.write(struct.pack("<I", len(meta_b))); out.write(meta_b)
    out.write(struct.pack("<I", len(dec))); out.write(dec)
    out.write(struct.pack("<I", len(lat))); out.write(lat)
    return out.getvalue()


def parse_payload(data):
    buf = io.BytesIO(data)
    ml = struct.unpack("<I", buf.read(4))[0]
    meta = json.loads(buf.read(ml))
    dl = struct.unpack("<I", buf.read(4))[0]
    decoder_sd = decode_decoder(buf.read(dl))
    ll = struct.unpack("<I", buf.read(4))[0]
    latents = decode_latents(buf.read(ll))
    return decoder_sd, latents, meta
