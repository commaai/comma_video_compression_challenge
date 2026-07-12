#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Reproduce archive.zip from a fine-tuned checkpoint (decoder + latents).

Quantizes the decoder per-tensor at a configurable bit-width and the latents to
8-bit per-dim, then entropy-codes with the codec_ctx container (no FEC6 selector,
no sidecar). Reuses the merged PR #112 modules from submissions/rhnerv_comma/
(codec_ctx, codec, model) rather than vendoring copies.

Usage:
  python compress.py <decoder.pt> <latents.pt> <out_archive.zip> [bits.json]

bits.json: optional {tensor_idx: bits}; default all 8-bit. The bit-width layout
must be embedded implicitly by the produced symbol ranges (the codec reconstructs
tensors from the stored scales, independent of bits), so inflate needs no bit map.
"""
from __future__ import annotations
import io, json, sys, zipfile
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RHNERV = HERE.parent / "rhnerv_comma"   # merged PR #112 — reused, not vendored
sys.path.insert(0, str(RHNERV))
sys.path.insert(0, str(HERE))
import codec_ctx  # noqa: E402
from codec import (  # noqa: E402
    LATENT_DIM, BASE_CHANNELS, EVAL_SIZE,
    DECODER_STORAGE_ORDER, DECODER_STREAM_ENDS, CONV4_STORAGE_PERMS,
    DECODER_BYTE_MAPS, LATENT_DIM_ORDER,
)
from model import HNeRVDecoder  # noqa: E402


def _sd_items(state_dict):
    names = list(HNeRVDecoder(LATENT_DIM, BASE_CHANNELS, EVAL_SIZE).state_dict().keys())
    return [(n, state_dict[n]) for n in names]


def _zigzag_encode(q):
    q = q.astype(np.int64)
    return np.where(q >= 0, 2 * q, -2 * q - 1).astype(np.uint8)


def _encode_mapped_u8(q, byte_map):
    q = q.astype(np.int64)
    if byte_map == "zig":
        return _zigzag_encode(q)
    if byte_map == "negzig":
        return _zigzag_encode(-q)
    if byte_map == "off":
        return (q + 128).astype(np.uint8)
    if byte_map == "twos":
        return (q & 0xFF).astype(np.uint8)
    raise ValueError(byte_map)


def _levels(bits):
    return (1 << (bits - 1)) - 1


def _build_decoder_raws(state_dict, bits_map):
    items = _sd_items(state_dict)
    per = {}
    for idx in DECODER_STORAGE_ORDER:
        name, tensor = items[idx]
        w = tensor.detach().cpu().float().numpy()
        L = _levels(bits_map.get(idx, 8))
        ma = float(np.abs(w).max())
        scale = float(np.float16(ma / L if ma > 0 else 1.0)) or float(np.float16(1.0))
        q = np.clip(np.rint(w / scale), -L, L).astype(np.int64)
        if w.ndim == 4:
            q = np.transpose(q, CONV4_STORAGE_PERMS[idx]).reshape(-1)
        else:
            q = q.reshape(-1)
        stored = _encode_mapped_u8(q, DECODER_BYTE_MAPS.get(idx, "zig"))
        per[idx] = stored.tobytes() + np.float16(scale).tobytes()
    raws, start = [], 0
    for end in DECODER_STREAM_ENDS:
        raws.append(b"".join(per[DECODER_STORAGE_ORDER[p]] for p in range(start, end)))
        start = end
    return raws


def _build_latent_raw(latents):
    lat = latents.detach().cpu().float().numpy()
    mins = lat.min(axis=0)
    rng = lat.max(axis=0) - mins
    scales = np.where(rng > 0, rng / 255.0, 1.0)
    mins16, scales16 = np.float16(mins), np.float16(scales)
    q = np.clip(np.rint((lat - mins16.astype(np.float32)) / scales16.astype(np.float32)),
                0, 255).astype(np.uint8)
    q_ord = q[:, list(LATENT_DIM_ORDER)].T.copy()
    delta = np.empty_like(q_ord)
    delta[:, 0] = q_ord[:, 0]
    step = q_ord[:, 1:].astype(np.int16) - q_ord[:, :-1].astype(np.int16)
    delta[:, 1:] = ((step + 128) & 255).astype(np.uint8)
    return mins16.tobytes() + scales16.tobytes() + delta.reshape(-1).tobytes()


def build_archive(state_dict, latents, bits_map=None):
    bits_map = bits_map or {}
    dec_sec = codec_ctx.encode_decoder_section(_build_decoder_raws(state_dict, bits_map))
    lat_sec = codec_ctx.encode_latent_section(_build_latent_raw(latents))
    member = codec_ctx.pack_container(
        dec_sec, lat_sec, b"",
        (codec_ctx.CODER_CTX, codec_ctx.CODER_CTX, codec_ctx.CODER_PASSTHROUGH))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("x", member)
    return buf.getvalue()


def main():
    if len(sys.argv) < 4:
        sys.exit("Usage: python compress.py <decoder.pt> <latents.pt> <out.zip> [bits.json]")
    sd = torch.load(sys.argv[1], map_location="cpu")
    lat = torch.load(sys.argv[2], map_location="cpu")
    if isinstance(lat, torch.nn.Parameter):
        lat = lat.data
    bits_map = None
    if len(sys.argv) > 4:
        bits_map = {int(k): int(v) for k, v in json.loads(Path(sys.argv[4]).read_text()).items()}
    archive = build_archive(sd, lat, bits_map)
    Path(sys.argv[3]).write_bytes(archive)
    print(f"wrote {sys.argv[3]} ({len(archive):,} B)")


if __name__ == "__main__":
    main()
