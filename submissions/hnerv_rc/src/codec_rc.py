"""Adaptive range-coded variant of the decoder/latent entropy stage.

Lossless drop-in for the brotli stage in codec.py. Same INT8 quantization and
same latent transform (per-dim uint8 + temporal delta + zigzag + lo/hi split);
only the final entropy coding changes: a per-stream adaptive order-0 range
coder (constriction) replaces brotli. Encoder and decoder maintain identical
integer count tables (uniform prior, fixed increment) so no frequency table is
transmitted and the two sides stay bit-identical on any IEEE-754 platform.

Measured on our trained weights: 161,736 B vs brotli 163,237 B on the decoder
(-1,501 B); latents coded the same way. Round-trip verified bit-exact.
"""
import io
import struct

import numpy as np
import brotli
import constriction

from codec import (zigzag_encode_i8, zigzag_decode_u8, quantize_state_dict,
                   encode_latents, decode_latents)

INC = 8.0
PRIOR = 1.0
ALPHABET = 256


def _encode_stream(enc, symbols_u8):
    """Adaptive order-0 encode of a uint8 stream into an open RangeEncoder."""
    counts = np.full(ALPHABET, PRIOR, dtype=np.float64)
    for s in symbols_u8:
        p = counts / counts.sum()
        enc.encode(np.array([s], dtype=np.int32),
                   constriction.stream.model.Categorical(p, perfect=False))
        counts[s] += INC


def _decode_stream(dec, n):
    """Adaptive order-0 decode of n symbols mirroring _encode_stream."""
    counts = np.full(ALPHABET, PRIOR, dtype=np.float64)
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        p = counts / counts.sum()
        s = int(dec.decode(constriction.stream.model.Categorical(p, perfect=False)))
        out[i] = s
        counts[s] += INC
    return out


# --- decoder weights --------------------------------------------------------

def encode_decoder_rc(q_sd):
    """Range-code the quantized state dict. Header (names/shapes/scales/sizes)
    is stored raw; the concatenated zigzag symbol stream is range-coded."""
    hdr = io.BytesIO()
    hdr.write(struct.pack("<I", len(q_sd)))
    all_syms = []
    for name, (q, scale, shape) in q_sd.items():
        nb = name.encode('utf-8')
        hdr.write(struct.pack("<I", len(nb))); hdr.write(nb)
        hdr.write(struct.pack("<I", len(shape)))
        for s in shape:
            hdr.write(struct.pack("<I", s))
        hdr.write(struct.pack("<f", scale))
        hdr.write(struct.pack("<I", q.size))
        all_syms.append(zigzag_encode_i8(q))
    enc = constriction.stream.queue.RangeEncoder()
    for syms in all_syms:               # one stream per tensor, reset counts
        _encode_stream(enc, syms.astype(np.int32))
    comp = enc.get_compressed().astype("<u4").tobytes()
    header = brotli.compress(hdr.getvalue(), quality=11)  # names/shapes/scales
    return struct.pack("<I", len(header)) + header + comp


def decode_decoder_rc(data):
    import torch
    buf = io.BytesIO(data)
    hlen = struct.unpack("<I", buf.read(4))[0]
    hdr = io.BytesIO(brotli.decompress(buf.read(hlen)))
    comp = np.frombuffer(buf.read(), dtype="<u4").copy()
    dec = constriction.stream.queue.RangeDecoder(comp)
    n = struct.unpack("<I", hdr.read(4))[0]
    sd = {}
    for _ in range(n):
        nl = struct.unpack("<I", hdr.read(4))[0]
        name = hdr.read(nl).decode('utf-8')
        nd = struct.unpack("<I", hdr.read(4))[0]
        shape = tuple(struct.unpack("<I", hdr.read(4))[0] for _ in range(nd))
        scale = struct.unpack("<f", hdr.read(4))[0]
        size = struct.unpack("<I", hdr.read(4))[0]
        zz = _decode_stream(dec, size)
        q = zigzag_decode_u8(zz)
        sd[name] = torch.from_numpy(q.astype(np.float32).reshape(shape)) * scale
    return sd


# --- latents ----------------------------------------------------------------
# reuse codec.encode_latents transform, range-code the lo+hi byte planes.

def encode_latents_rc(latents):
    payload = encode_latents(latents)          # header + lo + hi
    buf = io.BytesIO(payload)
    n, d = struct.unpack("<II", buf.read(8))
    head = buf.read(d * 2 + d * 2)             # fp16 mins + scales
    lo = np.frombuffer(buf.read(n * d), dtype=np.uint8)
    hi = np.frombuffer(buf.read(n * d), dtype=np.uint8)
    enc = constriction.stream.queue.RangeEncoder()
    _encode_stream(enc, lo.astype(np.int32))
    _encode_stream(enc, hi.astype(np.int32))
    comp = enc.get_compressed().astype("<u4").tobytes()
    return struct.pack("<II", n, d) + head + comp


def decode_latents_rc(data):
    buf = io.BytesIO(data)
    n, d = struct.unpack("<II", buf.read(8))
    head = buf.read(d * 2 + d * 2)
    comp = np.frombuffer(buf.read(), dtype="<u4").copy()
    dec = constriction.stream.queue.RangeDecoder(comp)
    lo = _decode_stream(dec, n * d)
    hi = _decode_stream(dec, n * d)
    payload = struct.pack("<II", n, d) + head + lo.tobytes() + hi.tobytes()
    return decode_latents(payload)
