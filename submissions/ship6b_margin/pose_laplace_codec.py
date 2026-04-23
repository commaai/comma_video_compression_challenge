"""Encoder + decoder for delta+Laplace+range-coded pose_targets.bin.

Drop-in replacement for `pose_lzma_codec.py`. Same input quantization (fp16
dim 0 + per-dim int8 rest, bit-identical to `encode_pose_mixed.quantize_rest_int8`).
Wire format replaces the lzma-wrapped payload with per-dim arithmetic coding
against a parametric Laplace PMF over zigzag-mapped first-order deltas.

Measured savings on the shipped 600-pair corpus:
  current pose_lzma wire format:  3,416 B
  new pose_laplace wire format:   ~3,122 B  (-~294 B raw)
  projected score delta:          ~-0.000196

Same quantized dim 0 fp16 values + same int8 rest values are reconstructed
by the decoder, so pose_dist is bit-identical to the c4-split parent.

Wire format (little-endian):
  uint16 n_pairs
  uint16 n_dims
  float32 mins[n_dims-1]
  float32 scales[n_dims-1]
  uint16 dim0_first_u16           # raw fp16 bit pattern of dim0[0]
  uint8  rest_firsts[n_dims-1]    # first int8 per rest dim
  for each dim d in 0..n_dims-1:
      float16 laplace_b           # PMF scale param
      uint16  alphabet_size       # 2*max_abs_delta + 1
      uint16  coded_len
      bytes   coded_bytes
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from range_coder import RangeDecoder, RangeEncoder, cdfs_from_freqs

PMF_SCALE = 1 << 14


def _zigzag(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int32)
    return ((x << 1) ^ (x >> 31)).astype(np.uint32)


def _unzigzag(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.int64)
    return (u >> 1) ^ -(u & 1)


def _laplace_pmf_freqs(b: float, max_abs: int) -> list[int]:
    """Laplace PMF freq table using platform-independent decimal arithmetic.

    The original np.exp implementation used platform-specific libm (glibc vs
    Apple vForce), causing borderline freq values to round differently on x86
    vs ARM. decimal.Decimal.exp() is implemented by mpdecimal (bundled in
    CPython) and computes correctly-rounded results for the specified precision,
    independent of CPU/OS. This eliminates cross-platform pose codec divergence.
    """
    from decimal import Decimal, getcontext

    ctx = getcontext()
    ctx.prec = 50

    b_dec = Decimal(str(b))
    if b_dec < Decimal("1e-6"):
        b_dec = Decimal("1e-6")
    alpha = 2 * max_abs + 1

    # Compute raw Laplace probs in original k-space using Decimal exp.
    probs = []
    for k in range(-max_abs, max_abs + 1):
        log_p = -Decimal(abs(k)) / b_dec
        probs.append(log_p.exp())
    total_p = sum(probs)

    # Map to zigzag order (standard: k>=0 → 2k, k<0 → -2k-1).
    zz_probs = [Decimal(0)] * alpha
    for i, k in enumerate(range(-max_abs, max_abs + 1)):
        if k >= 0:
            zz = k << 1
        else:
            zz = ((-k) << 1) - 1
        if 0 <= zz < alpha:
            zz_probs[zz] += probs[i]

    scale = Decimal(PMF_SCALE)
    freqs = [max(1, int((p / total_p * scale).to_integral_value())) for p in zz_probs]

    # Adjust residual so freqs sum to exactly PMF_SCALE.
    # When most symbols are at floor (1), a single pass may not absorb the
    # full residual. Cycle until resolved — guaranteed to terminate because
    # the peak entry can always absorb the remainder.
    diff = PMF_SCALE - sum(freqs)
    if diff != 0:
        zz_float = [float(p) for p in zz_probs]
        order = sorted(range(alpha), key=lambda i: -zz_float[i])
        step = 1 if diff > 0 else -1
        i = 0
        while diff != 0:
            idx = order[i % len(order)]
            if step < 0 and freqs[idx] <= 1:
                i += 1
                continue
            freqs[idx] += step
            diff -= step
            i += 1
    return freqs


def _rc_encode(symbols: np.ndarray, freqs: list[int]) -> bytes:
    cdf = cdfs_from_freqs(freqs)
    total = cdf[-1]
    enc = RangeEncoder()
    for s in symbols:
        s = int(s)
        enc.encode_symbol(cdf[s], cdf[s + 1], total)
    return enc.finish()


def _rc_decode(data: bytes, n: int, freqs: list[int]) -> np.ndarray:
    cdf = cdfs_from_freqs(freqs)
    total = cdf[-1]
    dec = RangeDecoder(data)
    out = np.zeros(n, dtype=np.int64)
    n_syms = len(freqs)
    for t in range(n):
        target = dec.decode_target(total)
        lo, hi = 0, n_syms
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if cdf[mid] <= target:
                lo = mid
            else:
                hi = mid
        out[t] = lo
        dec.advance(cdf[lo], cdf[lo + 1], total)
    return out


def _laplace_mle_scale(ks: np.ndarray) -> float:
    med = float(np.median(ks))
    return max(0.25, float(np.mean(np.abs(ks - med))))


def encode_pose_laplace_blob(
    n_pairs: int,
    n_dims: int,
    dim0_fp16: np.ndarray,
    q: np.ndarray,
    mins: np.ndarray,
    scales: np.ndarray,
) -> bytes:
    """Encode the pose blob in the delta+Laplace wire format."""
    if dim0_fp16.dtype != np.float16:
        raise TypeError(f"dim0_fp16 must be float16, got {dim0_fp16.dtype}")
    if q.dtype != np.uint8:
        raise TypeError(f"q must be uint8, got {q.dtype}")
    if q.shape != (n_pairs, n_dims - 1):
        raise ValueError(f"q shape {q.shape} != expected ({n_pairs}, {n_dims - 1})")
    if mins.shape != (n_dims - 1,) or scales.shape != (n_dims - 1,):
        raise ValueError("mins/scales shape mismatch")

    out = bytearray()
    out += struct.pack("<HH", n_pairs, n_dims)
    out += mins.astype("<f4").tobytes()
    out += scales.astype("<f4").tobytes()

    dim0_u16 = dim0_fp16.view(np.uint16).astype(np.int32)
    out += struct.pack("<H", int(dim0_u16[0]))

    rest_firsts = q[0, :].astype(np.uint8).tobytes()
    out += rest_firsts  # n_dims-1 bytes

    # Build n-1 delta streams per dim.
    streams: list[np.ndarray] = []
    alphabets: list[int] = []
    d0_delta = np.diff(dim0_u16)
    streams.append(_zigzag(d0_delta))
    alphabets.append(2 * int(np.max(np.abs(d0_delta))) + 1 if d0_delta.size else 1)
    for d in range(n_dims - 1):
        col = q[:, d].astype(np.int32)
        delta = np.diff(col)
        streams.append(_zigzag(delta))
        alphabets.append(2 * int(np.max(np.abs(delta))) + 1 if delta.size else 1)

    for zz, alpha in zip(streams, alphabets):
        max_abs = (alpha - 1) // 2
        ks = _unzigzag(zz).astype(np.float64)
        b_f32 = _laplace_mle_scale(ks)
        # Round to fp16 BEFORE building the PMF so encoder and decoder see the
        # exact same b value (decoder will reconstruct fp32 from the shipped
        # fp16). Without this, fp32→fp16→fp32 drift produces different integer
        # PMF freqs and the arithmetic coder round-trip diverges.
        b_fp16 = np.array([b_f32], dtype=np.float16)
        b = float(b_fp16[0])
        freqs = _laplace_pmf_freqs(b, max_abs)
        bs = _rc_encode(zz, freqs)
        out += b_fp16.tobytes()
        out += struct.pack("<H", alpha)
        out += struct.pack("<H", len(bs))
        out += bs

    return bytes(out)


def decode_pose_laplace(path: Path) -> tuple[np.ndarray, int]:
    """Inverse of encode_pose_laplace_blob — returns (pose_vecs_fp32, n_dims).

    Matches `decode_pose_lzma`'s return contract so inflate.py's call site is
    a 1-line swap.
    """
    blob = Path(path).read_bytes()
    pos = 0
    n_pairs, n_dims = struct.unpack_from("<HH", blob, pos)
    pos += 4
    n_rest = n_dims - 1

    mins = np.frombuffer(blob, dtype="<f4", count=n_rest, offset=pos).astype(np.float32)
    pos += n_rest * 4
    scales = np.frombuffer(blob, dtype="<f4", count=n_rest, offset=pos).astype(np.float32)
    pos += n_rest * 4

    (dim0_first_u16,) = struct.unpack_from("<H", blob, pos)
    pos += 2

    rest_firsts = np.frombuffer(blob, dtype=np.uint8, count=n_rest, offset=pos).copy()
    pos += n_rest

    streams_decoded: list[np.ndarray] = []
    for _ in range(n_dims):
        b = float(np.frombuffer(blob[pos : pos + 2], dtype="<f2")[0])
        pos += 2
        (alpha,) = struct.unpack_from("<H", blob, pos)
        pos += 2
        (coded_len,) = struct.unpack_from("<H", blob, pos)
        pos += 2
        coded = blob[pos : pos + coded_len]
        pos += coded_len
        max_abs = (alpha - 1) // 2
        freqs = _laplace_pmf_freqs(b, max_abs)
        zz = _rc_decode(coded, n_pairs - 1, freqs)
        streams_decoded.append(_unzigzag(zz).astype(np.int64))

    # Reconstruct dim 0 fp16 via uint16 cumsum.
    dim0_u16 = np.empty(n_pairs, dtype=np.int64)
    dim0_u16[0] = dim0_first_u16
    dim0_u16[1:] = dim0_first_u16 + np.cumsum(streams_decoded[0])
    dim0_u16 = dim0_u16.astype(np.uint16)
    dim0_fp16 = dim0_u16.view(np.float16).astype(np.float32)

    # Reconstruct rest int8 via per-dim cumsum.
    rest_deq = np.empty((n_pairs, n_rest), dtype=np.float32)
    for d in range(n_rest):
        col_int = np.empty(n_pairs, dtype=np.int64)
        col_int[0] = int(rest_firsts[d])
        col_int[1:] = col_int[0] + np.cumsum(streams_decoded[1 + d])
        col_uint8 = col_int.astype(np.uint8)  # wraps naturally if cumsum stays in [0,255]
        rest_deq[:, d] = col_uint8.astype(np.float32) * scales[d] + mins[d]

    pose_vecs = np.concatenate([dim0_fp16[:, None], rest_deq], axis=1).astype(np.float32)
    return pose_vecs, n_dims
