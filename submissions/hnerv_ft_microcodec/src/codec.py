"""Compact inflater-side codec for PR #98's fine-tuned HNeRV payload.

This stores the fixed model schema in code and keeps all video-specific payload
inside archive.zip:

  decoder: concatenated Brotli streams of q-bytes + fp16 scale per tensor
  latents: raw LZMA(fp16 min/scale per dim + centered temporal-delta uint8 latent codes)
  sidecar: Brotli((u8 dim, i8 delta_x100) per frame pair)
"""
import io
import lzma
import math
from functools import lru_cache

import brotli
import numpy as np
import torch

from model import HNeRVDecoder


DECODER_BLOB_LEN = 162_164
LATENT_BLOB_LEN = 15_387
N_PAIRS = 600
LATENT_DIM = 28
BASE_CHANNELS = 36
EVAL_SIZE = (384, 512)
LATENT_LZMA_FILTERS = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 4096, "lc": 3, "lp": 0, "pb": 0}
]

DECODER_STORAGE_ORDER = (
    14, 22, 7, 6, 19, 10, 25, 4, 20, 9, 12, 15, 5, 11,
    18, 1, 21, 3, 27, 13, 2, 26, 24, 17, 16, 23, 8, 0,
)
DECODER_STREAM_ENDS = (1, 2, 22, 23, 26, 27, 28)

CONV4_STORAGE_PERMS = {
    2: (3, 0, 2, 1),
    4: (3, 0, 2, 1),
    6: (0, 1, 2, 3),
    8: (3, 0, 1, 2),
    10: (3, 0, 2, 1),
    12: (3, 0, 1, 2),
    14: (1, 0, 2, 3),
    16: (3, 0, 2, 1),
    18: (1, 0, 2, 3),
    20: (0, 3, 2, 1),
    22: (0, 3, 2, 1),
    24: (0, 2, 3, 1),
    26: (0, 1, 3, 2),
}
CONV4_INVERSE_PERMS = {
    idx: tuple(np.argsort(perm)) for idx, perm in CONV4_STORAGE_PERMS.items()
}

DECODER_BYTE_MAPS = {
    9: "negzig",
    14: "negzig",
    20: "twos",
    27: "off",
}

LATENT_DIM_ORDER = (
    26, 0, 17, 15, 10, 24, 20, 12, 14, 21, 22, 18, 4, 11,
    3, 7, 16, 2, 6, 8, 19, 23, 5, 9, 1, 13, 27, 25,
)
SIDECAR_DELTAS_X100 = np.array(
    [-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10],
    dtype=np.int8,
)
SIDECAR_BASE = 1 + LATENT_DIM * len(SIDECAR_DELTAS_X100)
SIDECAR_PACKED_LEN = 661
SIDECAR_SPLIT_LEN = 656
SIDECAR_HUFF_LEN = 614
SIDECAR_HUFF_ENUM_LEN = 607
SIDECAR_HUFF_COMB_LEN = 609
SIDECAR_NOOP_RANK_PREFIX_LEN = 4
SIDECAR_NOOP_INFER_RANK_LEN = 3
SIDECAR_NOOP_TABLE_LEN = 7
SIDECAR_DIM_PACKED_LEN = 359
SIDECAR_DELTA_HUFF_LENGTHS_LEN = 8
SIDECAR_DELTA_HUFF3_LENGTHS_LEN = 6
SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN = 5
SIDECAR_HUFF_MIN_LEN = 2
SIDECAR_HUFF_MAX_LEN = 8
SIDECAR_HUFF_KRAFT_TOTAL = 1 << SIDECAR_HUFF_MAX_LEN


def unpack_nibbles(data, n):
    arr = np.frombuffer(data, dtype=np.uint8)
    unpacked = np.empty(arr.size * 2, dtype=np.uint8)
    unpacked[0::2] = arr & 15
    unpacked[1::2] = arr >> 4
    return unpacked[:n]


def unpack_3bit_lengths(data, n, offset):
    out = np.empty(n, dtype=np.uint8)
    bit_pos = 0
    for i in range(n):
        value = 0
        for _ in range(3):
            byte = data[bit_pos // 8]
            value = (value << 1) | ((byte >> (7 - (bit_pos % 8))) & 1)
            bit_pos += 1
        out[i] = value + offset
    return out


def decode_canonical_huffman(data, lengths, n_symbols):
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

    out = np.empty(n_symbols, dtype=np.uint8)
    out_pos = 0
    cur = 0
    cur_len = 0
    for byte in data:
        for shift in range(7, -1, -1):
            cur = (cur << 1) | ((byte >> shift) & 1)
            cur_len += 1
            sym = decode.get((cur_len, cur))
            if sym is not None:
                out[out_pos] = sym
                out_pos += 1
                if out_pos == n_symbols:
                    return out
                cur = 0
                cur_len = 0
    raise ValueError("truncated Huffman sidecar")


def decode_canonical_huffman_all(data, lengths):
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
    if pos == len(SIDECAR_DELTAS_X100):
        return int(remaining == 0)
    total = 0
    for length in range(SIDECAR_HUFF_MIN_LEN, SIDECAR_HUFF_MAX_LEN + 1):
        weight = 1 << (SIDECAR_HUFF_MAX_LEN - length)
        if remaining >= weight:
            total += huff_length_vector_count(pos + 1, remaining - weight)
    return total


def decode_huff_length_rank(rank):
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


def zigzag_decode_u8(arr_u8):
    arr = arr_u8.astype(np.int32)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


def decode_mapped_u8(arr_u8, byte_map):
    if byte_map == "zig":
        return zigzag_decode_u8(arr_u8)
    if byte_map == "negzig":
        return (-zigzag_decode_u8(arr_u8).astype(np.int16)).astype(np.int8)
    if byte_map == "off":
        return (arr_u8.astype(np.int16) - 128).astype(np.int8)
    if byte_map == "twos":
        return arr_u8.view(np.int8)
    raise ValueError(f"unknown decoder byte map: {byte_map}")


def decompress_brotli_streams(data, n_streams):
    outputs = []
    pos = 0
    for _ in range(n_streams):
        dec = brotli.Decompressor()
        chunks = []
        while pos < len(data) and not dec.is_finished():
            chunks.append(dec.process(data[pos:pos + 1]))
            pos += 1
        if not dec.is_finished():
            raise ValueError("truncated compact decoder payload")
        outputs.append(b"".join(chunks))
    if pos != len(data):
        raise ValueError("trailing compact decoder payload")
    return b"".join(outputs)


def decode_decoder_compact(data):
    raw = decompress_brotli_streams(data, len(DECODER_STREAM_ENDS))
    probe = HNeRVDecoder(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        eval_size=EVAL_SIZE,
    )
    items = list(probe.state_dict().items())
    pos = 0
    sd = {}

    for idx in DECODER_STORAGE_ORDER:
        name, tensor = items[idx]
        shape = tuple(tensor.shape)
        numel = int(tensor.numel())
        zz = np.frombuffer(raw, dtype=np.uint8, count=numel, offset=pos)
        pos += numel
        scale = np.frombuffer(raw, dtype=np.float16, count=1, offset=pos)[0]
        pos += 2

        q = decode_mapped_u8(zz, DECODER_BYTE_MAPS.get(idx, "zig"))
        if len(shape) == 4:
            storage_perm = CONV4_STORAGE_PERMS[idx]
            inverse_perm = CONV4_INVERSE_PERMS[idx]
            stored_shape = tuple(shape[i] for i in storage_perm)
            q = q.reshape(stored_shape)
            q = np.transpose(q, inverse_perm).copy()
        else:
            q = q.reshape(shape)
        sd[name] = torch.from_numpy(q.astype(np.float32)) * float(scale)

    if pos != len(raw):
        raise ValueError("trailing or truncated compact decoder payload")
    return sd


def decode_latents_compact(data):
    raw = lzma.decompress(data, format=lzma.FORMAT_RAW, filters=LATENT_LZMA_FILTERS)
    buf = io.BytesIO(raw)
    mins = torch.from_numpy(
        np.frombuffer(buf.read(LATENT_DIM * 2), dtype=np.float16).copy()
    ).float()
    scales = torch.from_numpy(
        np.frombuffer(buf.read(LATENT_DIM * 2), dtype=np.float16).copy()
    ).float()
    stored = np.frombuffer(buf.read(N_PAIRS * LATENT_DIM), dtype=np.uint8)
    if stored.size != N_PAIRS * LATENT_DIM:
        raise ValueError("truncated compact latent payload")
    delta_ordered = stored.reshape(LATENT_DIM, N_PAIRS)
    q_ordered = delta_ordered.copy()
    q_ordered[:, 1:] = np.cumsum(
        ((delta_ordered[:, 1:].astype(np.int16) - 128) & 255),
        axis=1,
        dtype=np.uint16,
    ).astype(np.uint8) + delta_ordered[:, :1]
    q_ordered = q_ordered.T.copy()
    q = np.empty((N_PAIRS, LATENT_DIM), dtype=np.uint8)
    q[:, LATENT_DIM_ORDER] = q_ordered
    return torch.from_numpy(q.astype(np.float32)) * scales.unsqueeze(0) + mins.unsqueeze(0)


def apply_latent_sidecar(latents, data):
    if not data:
        return latents
    raw = data
    if len(raw) not in (
        SIDECAR_HUFF_ENUM_LEN, SIDECAR_HUFF_COMB_LEN, SIDECAR_HUFF_LEN,
        SIDECAR_SPLIT_LEN, SIDECAR_PACKED_LEN, N_PAIRS, N_PAIRS * 2,
    ):
        raw = brotli.decompress(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size == SIDECAR_HUFF_ENUM_LEN:
        dim_end = SIDECAR_DIM_PACKED_LEN
        rank_end = dim_end + SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN
        length_rank = int.from_bytes(raw[dim_end:rank_end], "little")
        lengths = decode_huff_length_rank(length_rank)

        noop_rank_start = arr.size - SIDECAR_NOOP_INFER_RANK_LEN
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

        value = int.from_bytes(raw[:dim_end], "little")
        dims_valid = np.empty(n_valid, dtype=np.int64)
        for i in range(n_valid):
            value, dims_valid[i] = divmod(value, LATENT_DIM)
        if value:
            raise ValueError("bad compact Huffman sidecar dimensions")

        dims = np.full(N_PAIRS, 255, dtype=np.int64)
        codes = np.zeros(N_PAIRS, dtype=np.float32)
        dims[valid_mask] = dims_valid
        codes[valid_mask] = SIDECAR_DELTAS_X100[delta_valid].astype(np.float32)
    elif arr.size == SIDECAR_HUFF_COMB_LEN:
        noop_count = raw[0]
        noop_rank = int.from_bytes(raw[1:SIDECAR_NOOP_RANK_PREFIX_LEN], "little")
        noop_pos = decode_combination_colex(noop_rank, N_PAIRS, noop_count)
        valid_mask = np.ones(N_PAIRS, dtype=bool)
        valid_mask[noop_pos] = False
        n_valid = int(valid_mask.sum())

        dim_start = SIDECAR_NOOP_RANK_PREFIX_LEN
        dim_end = dim_start + SIDECAR_DIM_PACKED_LEN
        value = int.from_bytes(raw[dim_start:dim_end], "little")
        dims_valid = np.empty(n_valid, dtype=np.int64)
        for i in range(n_valid):
            value, dims_valid[i] = divmod(value, LATENT_DIM)
        if value:
            raise ValueError("bad compact Huffman sidecar dimensions")

        len_start = dim_end
        len_end = len_start + SIDECAR_DELTA_HUFF3_LENGTHS_LEN
        lengths = unpack_3bit_lengths(
            raw[len_start:len_end], len(SIDECAR_DELTAS_X100), 2
        )
        delta_valid = decode_canonical_huffman(
            raw[len_end:], lengths, n_valid
        ).astype(np.int64)

        dims = np.full(N_PAIRS, 255, dtype=np.int64)
        codes = np.zeros(N_PAIRS, dtype=np.float32)
        dims[valid_mask] = dims_valid
        codes[valid_mask] = SIDECAR_DELTAS_X100[delta_valid].astype(np.float32)
    elif arr.size in (SIDECAR_HUFF_LEN, SIDECAR_SPLIT_LEN):
        noop_count = raw[0]
        noop_pos = np.frombuffer(
            raw[1:1 + 2 * noop_count], dtype="<u2"
        ).astype(np.int64)
        if noop_count * 2 + 1 != SIDECAR_NOOP_TABLE_LEN:
            raise ValueError("bad split sidecar no-op table")
        valid_mask = np.ones(N_PAIRS, dtype=bool)
        valid_mask[noop_pos] = False
        n_valid = int(valid_mask.sum())

        dim_start = SIDECAR_NOOP_TABLE_LEN
        dim_end = dim_start + SIDECAR_DIM_PACKED_LEN
        value = int.from_bytes(raw[dim_start:dim_end], "little")
        dims_valid = np.empty(n_valid, dtype=np.int64)
        for i in range(n_valid):
            value, dims_valid[i] = divmod(value, LATENT_DIM)
        if value:
            raise ValueError("bad split sidecar dimensions")

        if arr.size == SIDECAR_HUFF_LEN:
            len_start = dim_end
            len_end = len_start + SIDECAR_DELTA_HUFF_LENGTHS_LEN
            lengths = unpack_nibbles(raw[len_start:len_end], len(SIDECAR_DELTAS_X100))
            delta_valid = decode_canonical_huffman(
                raw[len_end:], lengths, n_valid
            ).astype(np.int64)
        else:
            packed_delta = brotli.decompress(raw[dim_end:])
            delta_valid = unpack_nibbles(packed_delta, n_valid).astype(np.int64)

        dims = np.full(N_PAIRS, 255, dtype=np.int64)
        codes = np.zeros(N_PAIRS, dtype=np.float32)
        dims[valid_mask] = dims_valid
        codes[valid_mask] = SIDECAR_DELTAS_X100[delta_valid].astype(np.float32)
    elif arr.size == SIDECAR_PACKED_LEN:
        value = int.from_bytes(raw, "little")
        choices = np.empty(N_PAIRS, dtype=np.int64)
        for i in range(N_PAIRS):
            value, choices[i] = divmod(value, SIDECAR_BASE)
        if value:
            raise ValueError("bad packed latent sidecar")
        valid = choices != 0
        idx = choices[valid] - 1
        dims = np.full(N_PAIRS, 255, dtype=np.int64)
        codes = np.zeros(N_PAIRS, dtype=np.float32)
        dims[valid] = idx // len(SIDECAR_DELTAS_X100)
        codes[valid] = SIDECAR_DELTAS_X100[idx % len(SIDECAR_DELTAS_X100)].astype(np.float32)
    elif arr.size == N_PAIRS:
        choices = arr.astype(np.int64)
        valid = choices != 0
        idx = choices[valid] - 1
        dims = np.full(N_PAIRS, 255, dtype=np.int64)
        codes = np.zeros(N_PAIRS, dtype=np.float32)
        dims[valid] = idx // len(SIDECAR_DELTAS_X100)
        codes[valid] = SIDECAR_DELTAS_X100[idx % len(SIDECAR_DELTAS_X100)].astype(np.float32)
    elif arr.size == N_PAIRS * 2:
        pairs = arr.reshape(N_PAIRS, 2)
        dims = pairs[:, 0].astype(np.int64)
        codes = pairs[:, 1].view(np.int8).astype(np.float32)
    else:
        raise ValueError("bad latent sidecar length")
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


def parse_archive(archive_bytes):
    decoder_blob = archive_bytes[:DECODER_BLOB_LEN]
    latent_blob = archive_bytes[DECODER_BLOB_LEN:DECODER_BLOB_LEN + LATENT_BLOB_LEN]
    sidecar_blob = archive_bytes[DECODER_BLOB_LEN + LATENT_BLOB_LEN:]
    if not decoder_blob or not latent_blob:
        raise ValueError("bad compact archive")
    meta = {
        "n_pairs": N_PAIRS,
        "latent_dim": LATENT_DIM,
        "base_channels": BASE_CHANNELS,
        "eval_size": list(EVAL_SIZE),
    }
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    return decode_decoder_compact(decoder_blob), latents, meta
