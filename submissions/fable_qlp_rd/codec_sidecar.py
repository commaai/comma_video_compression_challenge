# SPDX-License-Identifier: MIT
"""Latent sidecar decoding for the 607-byte PR #101 enum-rank format.

Ported from `submissions/hnerv_fec6_fixed_huffman_k16/src/codec_sidecar.py`
(PR #110, @adpena, MIT; decode grammar by PR #101, @SajayR). This submission
carries the #101 sidecar VERBATIM (607 bytes, enum-ranked canonical Huffman
with inferred no-op positions), so only that format's decode path is kept.
All retained function bodies are verbatim fec6; the legacy/raw/Brotli sidecar
formats were dropped, which removes the inflate-time `brotli` dependency.
"""

from functools import lru_cache
import math

import numpy as np
import torch


N_PAIRS = 600
LATENT_DIM = 28
SIDECAR_DELTAS_X100 = np.array(
    [-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10],
    dtype=np.int8,
)
SIDECAR_HUFF_ENUM_LEN = 607
SIDECAR_NOOP_INFER_RANK_LEN = 3
SIDECAR_DIM_PACKED_LEN = 359
SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN = 5
SIDECAR_HUFF_MIN_LEN = 2
SIDECAR_HUFF_MAX_LEN = 8
SIDECAR_HUFF_KRAFT_TOTAL = 1 << SIDECAR_HUFF_MAX_LEN


def decode_canonical_huffman_all(data, lengths):
    """Decode a complete canonical Huffman stream and reject dangling bits."""
    decode = {}
    code = 0
    prev_len = 0
    for sym, length in sorted(
        ((sym, int(length)) for sym, length in enumerate(lengths) if length),
        key=lambda x: (x[1], x[0]),
    ):
        code <<= length - prev_len
        decode[(length, code)] = sym
        code += 1
        prev_len = length

    out = []
    cur = 0
    cur_len = 0
    for byte in data:
        for shift in range(7, -1, -1):
            cur = (cur << 1) | ((byte >> shift) & 1)
            cur_len += 1
            sym = decode.get((cur_len, cur))
            if sym is not None:
                out.append(sym)
                cur = 0
                cur_len = 0
    if cur_len:
        raise ValueError("truncated Huffman sidecar")
    return np.array(out, dtype=np.uint8)


@lru_cache(None)
def huff_length_vector_count(pos, remaining):
    """Count remaining valid canonical length vectors for rank decoding."""
    if pos == len(SIDECAR_DELTAS_X100):
        return int(remaining == 0)
    total = 0
    for length in range(SIDECAR_HUFF_MIN_LEN, SIDECAR_HUFF_MAX_LEN + 1):
        weight = 1 << (SIDECAR_HUFF_MAX_LEN - length)
        if remaining >= weight:
            total += huff_length_vector_count(pos + 1, remaining - weight)
    return total


def decode_huff_length_rank(rank):
    """Recover a canonical Huffman length vector from its colex rank."""
    if rank >= huff_length_vector_count(0, SIDECAR_HUFF_KRAFT_TOTAL):
        raise ValueError("bad Huffman length-vector rank")
    lengths = np.empty(len(SIDECAR_DELTAS_X100), dtype=np.uint8)
    remaining = SIDECAR_HUFF_KRAFT_TOTAL
    for pos in range(lengths.size):
        for length in range(SIDECAR_HUFF_MIN_LEN, SIDECAR_HUFF_MAX_LEN + 1):
            weight = 1 << (SIDECAR_HUFF_MAX_LEN - length)
            if remaining < weight:
                continue
            block = huff_length_vector_count(pos + 1, remaining - weight)
            if rank >= block:
                rank -= block
            else:
                lengths[pos] = length
                remaining -= weight
                break
        else:
            raise ValueError("bad Huffman length-vector rank")
    if remaining or rank:
        raise ValueError("bad Huffman length-vector rank")
    return lengths


def decode_combination_colex(rank, n, k):
    """Decode a k-of-n combination from colexicographic rank."""
    if rank >= math.comb(n, k):
        raise ValueError("bad combination rank")
    combo = [0] * k
    x = n
    for i in range(k, 0, -1):
        x -= 1
        while math.comb(x, i) > rank:
            x -= 1
        combo[i - 1] = x
        rank -= math.comb(x, i)
    if rank:
        raise ValueError("bad combination rank")
    return np.array(combo, dtype=np.int64)


def _packed_dims(value, n_valid, *, error_message):
    """Recover little-endian base-LATENT_DIM sidecar dimensions."""
    dims_valid = np.empty(n_valid, dtype=np.int64)
    for i in range(n_valid):
        value, dims_valid[i] = divmod(value, LATENT_DIM)
    if value:
        raise ValueError(error_message)
    return dims_valid


def _vectors_from_valid(valid_mask, dims_valid, delta_valid):
    """Materialize full per-pair sidecar vectors from valid pair entries."""
    dims = np.full(N_PAIRS, 255, dtype=np.int64)
    codes = np.zeros(N_PAIRS, dtype=np.float32)
    dims[valid_mask] = dims_valid
    codes[valid_mask] = SIDECAR_DELTAS_X100[delta_valid].astype(np.float32)
    return dims, codes


def _decode_enum_rank_sidecar(raw, arr_size):
    """Decode enum-ranked Huffman sidecar with inferred no-op positions."""
    dim_end = SIDECAR_DIM_PACKED_LEN
    rank_end = dim_end + SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN
    length_rank = int.from_bytes(raw[dim_end:rank_end], "little")
    lengths = decode_huff_length_rank(length_rank)

    noop_rank_start = arr_size - SIDECAR_NOOP_INFER_RANK_LEN
    delta_valid = decode_canonical_huffman_all(
        raw[rank_end:noop_rank_start], lengths
    ).astype(np.int64)
    n_valid = delta_valid.size
    noop_count = N_PAIRS - n_valid
    if noop_count < 0:
        raise ValueError("bad compact Huffman sidecar length")

    noop_rank = int.from_bytes(raw[noop_rank_start:], "little")
    noop_pos = decode_combination_colex(noop_rank, N_PAIRS, noop_count)
    valid_mask = np.ones(N_PAIRS, dtype=bool)
    valid_mask[noop_pos] = False
    if int(valid_mask.sum()) != n_valid:
        raise ValueError("bad compact Huffman sidecar no-op count")

    dims_valid = _packed_dims(
        int.from_bytes(raw[:dim_end], "little"),
        n_valid,
        error_message="bad compact Huffman sidecar dimensions",
    )
    return _vectors_from_valid(valid_mask, dims_valid, delta_valid)


def apply_latent_sidecar(latents, data):
    """Apply archive-local latent corrections without changing decoder math."""
    if not data:
        return latents
    if len(data) != SIDECAR_HUFF_ENUM_LEN:
        raise ValueError("this submission only carries the 607-byte "
                         "enum-rank sidecar format")
    dims, codes = _decode_enum_rank_sidecar(data, len(data))
    valid = dims != 255
    if np.any(dims[valid] >= LATENT_DIM):
        raise ValueError("bad latent sidecar dimension")
    if valid.any():
        row = torch.from_numpy(np.nonzero(valid)[0])
        col = torch.from_numpy(dims[valid])
        delta = torch.from_numpy(codes[valid] / 100.0).to(latents.dtype)
        latents = latents.clone()
        latents[row, col] += delta
    return latents
