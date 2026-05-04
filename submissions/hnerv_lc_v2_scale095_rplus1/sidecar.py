"""Latent-correction sidecar for hnerv_repack_latent.

Wire format (single blob, brotli'd):
  u16 n_pairs
  per pair: u8 dim_idx (0..27, or 255 = no correction), i8 delta_quantized (real = i8 * DELTA_SCALE)

At inflate time, for each pair p:
  if dim_idx[p] != 255:
      latents[p, dim_idx[p]] += delta_quantized[p] * DELTA_SCALE
"""
import struct
import numpy as np

DELTA_SCALE = 0.0095  # int8 quant: real_delta = i8 * 0.01 (range ±1.27)


def encode_corrections(out_dim, out_delta_q):
    """out_dim, out_delta_q: int8 arrays of length 600. dim=0 + delta_q=0 means 'no correction'.
    Returns brotli-compressed blob."""
    import brotli
    n = len(out_dim)
    assert len(out_delta_q) == n
    # Mark 'no correction' as dim=255 (since dim 0 is valid)
    dim_packed = np.where(out_delta_q == 0, 255, out_dim).astype(np.uint8)
    payload = struct.pack('<H', n) + np.stack([dim_packed, out_delta_q.astype(np.int8).view(np.uint8)], axis=1).tobytes()
    return brotli.compress(payload, quality=11)


def decode_corrections(blob):
    """Returns (dim_arr (n, int8), delta_q_arr (n, int8)). dim==255 means no correction."""
    import brotli
    raw = brotli.decompress(blob)
    n = struct.unpack_from('<H', raw, 0)[0]
    arr = np.frombuffer(raw[2:2 + 2*n], dtype=np.uint8).reshape(n, 2)
    dim = arr[:, 0]  # uint8 with 255 sentinel
    delta_q = arr[:, 1].view(np.int8)  # signed
    return dim, delta_q


def apply_corrections(latents_tensor, dim_arr, delta_q_arr, scale=DELTA_SCALE):
    """In-place add correction to latents_tensor (n, latent_dim). dim==255 means no-op."""
    import torch
    n = latents_tensor.shape[0]
    for p in range(n):
        d = int(dim_arr[p])
        if d == 255:
            continue
        latents_tensor[p, d] = latents_tensor[p, d] + float(delta_q_arr[p]) * scale
    return latents_tensor
