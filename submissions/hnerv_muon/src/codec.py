"""Compression codec for the HNeRV decoder + per-frame-pair latents.

Decoder path: per-tensor symmetric INT8 quantization → zigzag → concat with
shape/scale metadata → brotli (quality 11). We previously used a hybrid that
added per-tensor categorical AC (via the constriction crate) for big tensors —
that was ~217 bytes smaller (~+0.0001 to score) but added a Python dependency
and embedded a feature-level compression choice into the inflation path. Pure
brotli is the simpler, more transparent default.

Latent path: per-dim min/max scaling to [0, 254] (uint8), then 1st-order
temporal delta, zigzag to uint16, split into lo/hi byte streams (lo brotli's
well, hi is mostly zero). Beats plain brotli by ~240 bytes on our latents.

Round-trip verified bit-exact.
"""
import io
import struct
import numpy as np
import torch
import brotli


N_QUANT = 127


# ============================================================================
# Quantization
# ============================================================================

def quantize_state_dict(sd, n_quant=N_QUANT):
    """Per-tensor symmetric INT8 quant. Returns {name: (int8_flat_array, scale, shape)}."""
    out = {}
    for name, tensor in sd.items():
        t = tensor.detach().cpu().float()
        m = t.abs().max().item()
        scale = m / n_quant if m > 0 else 1.0
        q = (t / scale).round().clamp(-n_quant, n_quant).to(torch.int8).numpy().flatten()
        out[name] = (q, scale, tuple(tensor.shape))
    return out


def zigzag_encode_i8(arr_i8):
    arr = arr_i8.astype(np.int32)
    return np.where(arr >= 0, 2 * arr, -2 * arr - 1).astype(np.uint8)


def zigzag_decode_u8(arr_u8):
    arr = arr_u8.astype(np.int32)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


# ============================================================================
# Decoder weights: pure brotli on the entire INT8-quantized state dict.
# We previously used a hybrid (per-tensor categorical AC for big tensors + brotli for small)
# but switched to pure brotli for simplicity — it's only ~217 bytes worse on our
# trained weights (~+0.0001 to score) and removes the constriction dependency.
# ============================================================================

def encode_decoder(q_sd):
    """Encode quantized state dict to compressed bytes via zigzag + brotli."""
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(q_sd)))
    for name, (q, scale, shape) in q_sd.items():
        nb = name.encode('utf-8')
        buf.write(struct.pack("<I", len(nb))); buf.write(nb)
        buf.write(struct.pack("<I", len(shape)))
        for s in shape: buf.write(struct.pack("<I", s))
        buf.write(struct.pack("<f", scale))
        buf.write(struct.pack("<I", q.size))
        buf.write(zigzag_encode_i8(q).tobytes())
    return brotli.compress(buf.getvalue(), quality=11)


def decode_decoder(data):
    """Inverse of encode_decoder. Returns {name: torch.Tensor (float32, dequantized)}."""
    raw = brotli.decompress(data)
    buf = io.BytesIO(raw)
    n = struct.unpack("<I", buf.read(4))[0]
    sd = {}
    for _ in range(n):
        nl = struct.unpack("<I", buf.read(4))[0]
        name = buf.read(nl).decode('utf-8')
        nd = struct.unpack("<I", buf.read(4))[0]
        shape = tuple(struct.unpack("<I", buf.read(4))[0] for _ in range(nd))
        scale = struct.unpack("<f", buf.read(4))[0]
        size = struct.unpack("<I", buf.read(4))[0]
        zz = np.frombuffer(buf.read(size), dtype=np.uint8)
        q = zigzag_decode_u8(zz)
        sd[name] = torch.from_numpy(q.astype(np.float32).reshape(shape)) * scale
    return sd


# ============================================================================
# Latents: delta + zigzag + brotli (lo/hi byte split)
# ============================================================================

def encode_latents(latents: torch.Tensor):
    """Encode (n_pairs, latent_dim) float tensor to bytes.

    Per-dim asymmetric UINT8 (min/max scaling to [0,254]) + 1st-order temporal
    delta + zigzag to uint16 + lo/hi byte split (lo brotli's well, hi mostly 0).
    """
    t = latents.detach().cpu().float()
    n, d = t.shape
    mins = t.min(dim=0).values
    maxs = t.max(dim=0).values
    scales = ((maxs - mins) / 254.0).clamp(min=1e-10)
    q = ((t - mins.unsqueeze(0)) / scales.unsqueeze(0)).round().clamp(0, 254).to(torch.uint8).numpy()
    delta = np.empty_like(q, dtype=np.int16)
    delta[0] = q[0]
    delta[1:] = q[1:].astype(np.int16) - q[:-1].astype(np.int16)
    delta_zz = np.where(delta >= 0, 2 * delta, -2 * delta - 1).astype(np.uint16)
    lo = (delta_zz & 0xFF).astype(np.uint8).tobytes()
    hi = (delta_zz >> 8).astype(np.uint8).tobytes()
    payload = struct.pack("<II", n, d)
    payload += mins.to(torch.float16).numpy().tobytes()
    payload += scales.to(torch.float16).numpy().tobytes()
    payload += lo + hi
    return payload  # caller wraps in brotli


def decode_latents(raw):
    buf = io.BytesIO(raw)
    n, d = struct.unpack("<II", buf.read(8))
    mins = torch.from_numpy(np.frombuffer(buf.read(d * 2), dtype=np.float16).copy()).float()
    scales = torch.from_numpy(np.frombuffer(buf.read(d * 2), dtype=np.float16).copy()).float()
    total = n * d
    lo = np.frombuffer(buf.read(total), dtype=np.uint8).astype(np.uint16)
    hi = np.frombuffer(buf.read(total), dtype=np.uint8).astype(np.uint16)
    delta_zz = ((hi << 8) | lo).reshape(n, d)
    delta = np.where(delta_zz % 2 == 0, delta_zz.astype(np.int32) // 2,
                     -(delta_zz.astype(np.int32) // 2) - 1).astype(np.int16)
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, n):
        q[i] = q[i - 1] + delta[i]
    q = q.astype(np.uint8)
    return torch.from_numpy(q.astype(np.float32)) * scales.unsqueeze(0) + mins.unsqueeze(0)


# ============================================================================
# Top-level archive: meta + decoder + latents
# ============================================================================

def build_archive(decoder_state_dict, latents, meta_dict):
    """Build the final archive blob.

    Layout:
      [meta_brotli_len:u32] [meta_brotli]
      [decoder_blob_len:u32] [decoder_blob]
      [latents_brotli_len:u32] [latents_brotli]
    """
    import json
    meta_raw = json.dumps(meta_dict).encode('utf-8')
    meta_brotli = brotli.compress(meta_raw, quality=11)

    q_sd = quantize_state_dict(decoder_state_dict)
    decoder_blob = encode_decoder(q_sd)

    latents_payload = encode_latents(latents)
    latents_brotli = brotli.compress(latents_payload, quality=11)

    out = io.BytesIO()
    out.write(struct.pack("<I", len(meta_brotli))); out.write(meta_brotli)
    out.write(struct.pack("<I", len(decoder_blob))); out.write(decoder_blob)
    out.write(struct.pack("<I", len(latents_brotli))); out.write(latents_brotli)
    return out.getvalue()


def parse_archive(archive_bytes):
    """Inverse of build_archive. Returns (decoder_sd, latents_tensor, meta_dict)."""
    import json
    buf = io.BytesIO(archive_bytes)
    meta_len = struct.unpack("<I", buf.read(4))[0]
    meta = json.loads(brotli.decompress(buf.read(meta_len)))
    dec_len = struct.unpack("<I", buf.read(4))[0]
    decoder_sd = decode_decoder(buf.read(dec_len))
    lat_len = struct.unpack("<I", buf.read(4))[0]
    latents = decode_latents(brotli.decompress(buf.read(lat_len)))
    return decoder_sd, latents, meta
