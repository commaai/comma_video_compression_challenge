# SPDX-License-Identifier: MIT
"""Inflater-side codec for the PR #101 HNeRV-microcodec payload, raw-stream
variant.

Ported from `submissions/hnerv_fec6_fixed_huffman_k16/src/codec.py` (PR #110,
@adpena, MIT), which itself reuses PR #101's (@SajayR) decode grammar. The
ONLY functional change vs the fec6 original: the entropy-coding layer has been
swapped out. `decode_decoder_compact` takes the RAW concatenated decoder
stream bytes directly (already entropy-decoded by `codec_ctx`) instead of a
Brotli blob, and `decode_latents_compact` takes the raw latent payload
directly instead of an LZMA blob. Everything downstream of the entropy layer
-- storage order, conv4 storage permutations, byte maps, scale handling,
latent delta/cumsum reconstruction, dim reordering, fp32 math -- is verbatim
fec6/#101, so the reconstructed tensors are bit-identical.

All schema constants live in code (uncounted by the rate metric); the
archive carries only video-specific payload bytes.
"""
import io

import numpy as np
import torch

from model import HNeRVDecoder


N_PAIRS = 600
LATENT_DIM = 28
BASE_CHANNELS = 36
EVAL_SIZE = (384, 512)
DECODER_RAW_LEN = 229_014   # sum of the 7 raw decoder streams
LATENT_RAW_LEN = 16_912     # 2*2*28 fp16 header + 600*28 codes

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


def zigzag_decode_u8(arr_u8):
    """Map unsigned zigzag symbols back to signed int8 residuals."""
    arr = arr_u8.astype(np.int32)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


def decode_mapped_u8(arr_u8, byte_map):
    """Decode one stored uint8 tensor stream using its declared byte map."""
    if byte_map == "zig":
        return zigzag_decode_u8(arr_u8)
    if byte_map == "negzig":
        return (-zigzag_decode_u8(arr_u8).astype(np.int16)).astype(np.int8)
    if byte_map == "off":
        return (arr_u8.astype(np.int16) - 128).astype(np.int8)
    if byte_map == "twos":
        return arr_u8.view(np.int8)
    raise ValueError(f"unknown decoder byte map: {byte_map}")


def decode_decoder_compact(raw):
    """Decode the compact HNeRV state_dict from RAW concatenated stream bytes.

    `raw` is the byte concatenation of the 7 decoder streams that
    `codec_ctx.unpack_container` returns (per-tensor q-bytes + fp16 scale, in
    DECODER_STORAGE_ORDER). Identical to fec6's decode_decoder_compact after
    its Brotli step."""
    if len(raw) != DECODER_RAW_LEN:
        raise ValueError("bad raw decoder stream length")
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


def decode_latents_compact(raw):
    """Decode RAW per-pair latent payload bytes into float tensors.

    `raw` is what fec6's decode_latents_compact sees after its LZMA step:
    fp16 mins + fp16 scales + centered temporal-delta uint8 codes."""
    if len(raw) != LATENT_RAW_LEN:
        raise ValueError("bad raw latent payload length")
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
