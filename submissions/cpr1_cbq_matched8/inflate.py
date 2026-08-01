#!/usr/bin/env python3
"""Inflate exact semantic tokens through a compact semantic and pose renderer."""

from __future__ import annotations

import lzma
import math
import struct
import sys
import time
from pathlib import Path

import constriction
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from carrier_codec import MAGIC as COMPACT_CARRIER_MAGIC
from carrier_codec import decode_compact_carrier
from hpac_integer import IntegerHPAC
from hpac_integer_sparse import SparseIntegerHPAC
from integer_model_io import deserialize_integer_model


N = 600
NUM_CLASSES = 5
EVAL_H, EVAL_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
SEMANTIC_WIDTH = 96
SEMANTIC_WIDTH_BY_PAYLOAD_BYTES = {30748: 80, 40252: 96}
SEMANTIC_BLOCKS = 4
SEMANTIC_FRAME_DIM = 8
CARRIER_DIM = 12
CARRIER_H, CARRIER_W = 24, 32
CARRIER_AMPLITUDE = 64.0
CARRIER_COEFF_BITS = 12
CARRIER_COEFF_DELTA_ZIGZAG = True
HPAC_PATCH = 64
HPAC_DELTA = 2
HPAC_CHANNELS = 64
HPAC_FILM_DIM = 8
HPAC_LOGIT_PRECISION = 8
HPAC_TARGET_MODE = "raw"
HPAC_CODER_PERFECT = False
HPAC_HIERARCHICAL = False
HPAC_PACKED_SCHEMA = (
    ("conv_a.weight_q", (64, 7, 23), "i1"),
    ("conv_a.weight_scale", (64,), "<f2"),
    ("conv_b1.weight_q", (64, 1, 14), "i1"),
    ("conv_b1.weight_scale", (64,), "<f2"),
    ("conv_b2.weight_q", (64, 1, 5), "i1"),
    ("conv_b2.weight_scale", (64,), "<f2"),
    ("conv_past.weight_q", (64, 5, 3, 3), "i1"),
    ("conv_past.weight_scale", (64,), "<f2"),
    ("film_gen.weight_q", (128, 8), "i1"),
    ("film_gen.weight_scale", (128,), "<f2"),
    ("head.weight_q", (5, 64, 1, 1), "i1"),
    ("head.weight_scale", (5,), "<f2"),
    ("spm.dw.weight_q", (64, 1, 3, 3), "i1"),
    ("spm.dw.weight_scale", (64,), "<f2"),
    ("spm.pw.weight_q", (64, 64, 1, 1), "i1"),
    ("spm.pw.weight_scale", (64,), "<f2"),
    ("frame_embed.weight_q", (600, 8), "i1"),
    ("frame_embed.weight_scale", (8,), "<f2"),
    ("film_gen.bias", (128,), "<f2"),
    ("conv_a.bias", (64,), "<f2"),
    ("gn_a.scale", (64,), "<f2"),
    ("gn_a.shift", (64,), "<f2"),
    ("conv_b1.bias", (64,), "<f2"),
    ("gn_b1.scale", (64,), "<f2"),
    ("gn_b1.shift", (64,), "<f2"),
    ("conv_b2.bias", (64,), "<f2"),
    ("gn_b2.scale", (64,), "<f2"),
    ("gn_b2.shift", (64,), "<f2"),
    ("conv_past.bias", (64,), "<f2"),
    ("spm.norm.scale", (64,), "<f2"),
    ("spm.norm.shift", (64,), "<f2"),
    ("spm.dw.bias", (64,), "<f2"),
    ("spm.pw.bias", (64,), "<f2"),
    ("head.bias", (5,), "<f2"),
)


def unpack_signed_int4(blob: memoryview, count: int):
    byte_count = (count + 1) // 2
    packed = np.frombuffer(blob[:byte_count], dtype=np.uint8)
    values = np.empty(byte_count * 2, dtype=np.int8)
    values[0::2] = packed & 0xF
    values[1::2] = packed >> 4
    values[values >= 8] -= 16
    return torch.from_numpy(values[:count].copy()), blob[byte_count:]


def unpack_signed_bits(blob: memoryview, count: int, bits: int):
    if not 2 <= bits <= 8:
        raise ValueError("signed bit unpacker supports 2 through 8 bits")
    byte_count = (count * bits + 7) // 8
    packed = np.frombuffer(blob[:byte_count], dtype=np.uint8)
    bitstream = np.unpackbits(packed, bitorder="little")[:count * bits]
    bitstream = bitstream.reshape(count, bits).astype(np.int16, copy=False)
    shifts = (1 << np.arange(bits, dtype=np.int16))[None]
    unsigned = (bitstream * shifts).sum(axis=1, dtype=np.int16)
    sign = 1 << (bits - 1)
    values = np.where(unsigned >= sign, unsigned - (1 << bits), unsigned)
    return torch.from_numpy(values.astype(np.int8, copy=False)), blob[byte_count:]


def unpack_signed_int12(blob: memoryview, count: int):
    byte_count = ((count + 1) // 2) * 3
    packed = np.frombuffer(blob[:byte_count], dtype=np.uint8).astype(np.uint16)
    values = np.empty((byte_count // 3) * 2, dtype=np.int16)
    values[0::2] = packed[0::3] | ((packed[1::3] & 0xF) << 8)
    values[1::2] = (packed[1::3] >> 4) | (packed[2::3] << 4)
    values[values >= 2048] -= 4096
    return torch.from_numpy(values[:count].copy()), blob[byte_count:]


class TokenBlock(nn.Module):
    def __init__(self, width: int, frame_dim: int, dilation: int):
        super().__init__()
        self.dw = nn.Conv2d(
            width, width, 3, padding=dilation, dilation=dilation, groups=width
        )
        self.pw = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(max(1, width // 8), width)
        self.film = nn.Linear(frame_dim, 2 * width)

    def forward(self, value: torch.Tensor, frame: torch.Tensor):
        residual = self.norm(self.pw(self.dw(value)))
        scale, shift = self.film(frame).chunk(2, dim=1)
        residual = residual * (1.0 + scale[:, :, None, None])
        residual = residual + shift[:, :, None, None]
        return value + F.gelu(residual)


class SemanticTokenRenderer(nn.Module):
    def __init__(self, width: int = SEMANTIC_WIDTH):
        super().__init__()
        self.token_embed = nn.Embedding(NUM_CLASSES, width)
        self.frame_embed = nn.Embedding(N, SEMANTIC_FRAME_DIM)
        self.coord_mix = nn.Conv2d(width + 4, width, 1)
        dilations = (1, 1, 2, 4)
        self.blocks = nn.ModuleList([
            TokenBlock(width, SEMANTIC_FRAME_DIM, dilation)
            for dilation in dilations
        ])
        self.head = nn.Conv2d(width, 3, 3, padding=1)

    @staticmethod
    def coordinates(batch: int, device: torch.device, dtype: torch.dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, EVAL_H, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, EVAL_W, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([xx, yy, xx.square(), yy.square()], dim=0)
        return coords.unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, tokens: torch.Tensor, pair_idx: torch.Tensor):
        value = self.token_embed(tokens).permute(0, 3, 1, 2)
        value = self.coord_mix(torch.cat([
            value, self.coordinates(value.shape[0], value.device, value.dtype)
        ], dim=1))
        frame = self.frame_embed(pair_idx)
        for block in self.blocks:
            value = block(value, frame)
        return torch.sigmoid(self.head(F.gelu(value))) * 255.0


def unpack_semantic(blob: bytes, template: dict[str, torch.Tensor]):
    remaining = memoryview(blob)
    restored = {}
    for name, value in template.items():
        shape = tuple(value.shape)
        count = value.numel()
        if value.ndim < 2:
            byte_count = count * 2
            array = np.frombuffer(remaining[:byte_count], dtype="<f2").copy()
            restored[name] = torch.from_numpy(array).reshape(shape).float()
            remaining = remaining[byte_count:]
            continue
        is_embedding = name.endswith("embed.weight")
        scale_count = shape[-1] if is_embedding else shape[0]
        scale_bytes = scale_count * 2
        scales = torch.from_numpy(
            np.frombuffer(remaining[:scale_bytes], dtype="<f2").copy()
        ).float()
        remaining = remaining[scale_bytes:]
        codes, remaining = unpack_signed_int4(remaining, count)
        scale_shape = [1] * len(shape)
        scale_shape[-1 if is_embedding else 0] = scale_count
        restored[name] = codes.reshape(shape).float() * scales.reshape(scale_shape)
    if remaining:
        raise ValueError("semantic payload has trailing bytes")
    return restored


def unpack_semantic_pose(raw: bytes):
    if len(raw) < 8:
        raise ValueError("truncated semantic-pose payload")
    semantic_bytes = int.from_bytes(raw[:4], byteorder="little")
    carrier_bytes = int.from_bytes(raw[4:8], byteorder="little")
    if len(raw) != 8 + semantic_bytes + carrier_bytes:
        raise ValueError("semantic-pose payload length mismatch")
    semantic_blob = raw[8:8 + semantic_bytes]
    carrier_blob = memoryview(raw[8 + semantic_bytes:])
    try:
        semantic_width = SEMANTIC_WIDTH_BY_PAYLOAD_BYTES[semantic_bytes]
    except KeyError as error:
        raise ValueError(
            f"unsupported semantic payload size: {semantic_bytes} bytes"
        ) from error
    semantic = SemanticTokenRenderer(semantic_width)
    semantic.load_state_dict(unpack_semantic(semantic_blob, semantic.state_dict()))

    scale_bytes = CARRIER_DIM * 4
    basis_count = CARRIER_DIM * 3 * CARRIER_H * CARRIER_W
    coeff_count = N * CARRIER_DIM
    if carrier_blob[:4] == COMPACT_CARRIER_MAGIC:
        (
            basis_scales_array,
            basis_codes_array,
            coeff_scales_array,
            encoded_coefficients,
        ) = decode_compact_carrier(
            carrier_blob,
            basis_count=basis_count,
            frames=N,
            dimensions=CARRIER_DIM,
        )
        basis_scales = torch.from_numpy(basis_scales_array)
        basis_codes = torch.from_numpy(
            basis_codes_array.astype(np.int8, copy=False)
        )
        coeff_scales = torch.from_numpy(coeff_scales_array)
        coeff_codes = torch.from_numpy(
            encoded_coefficients.astype(np.int32, copy=False)
        )
    else:
        basis_scales = torch.from_numpy(
            np.frombuffer(carrier_blob[:scale_bytes], dtype="<f4").copy()
        )
        carrier_blob = carrier_blob[scale_bytes:]
        coeff_code_bytes = ((coeff_count + 1) // 2) * 3
        basis_code_bytes = carrier_bytes - 2 * scale_bytes - coeff_code_bytes
        if basis_code_bytes <= 0 or (basis_code_bytes * 8) % basis_count:
            raise ValueError("carrier payload has an invalid packed basis length")
        basis_bits = basis_code_bytes * 8 // basis_count
        if not 4 <= basis_bits <= 8:
            raise ValueError(f"unsupported carrier basis precision: {basis_bits}")
        basis_codes, carrier_blob = unpack_signed_bits(
            carrier_blob, basis_count, basis_bits
        )
        coeff_scales = torch.from_numpy(
            np.frombuffer(carrier_blob[:scale_bytes], dtype="<f4").copy()
        )
        carrier_blob = carrier_blob[scale_bytes:]
        if CARRIER_COEFF_BITS != 12:
            raise ValueError("submission payload expects signed int12 coefficients")
        coeff_codes, carrier_blob = unpack_signed_int12(carrier_blob, coeff_count)
        if carrier_blob:
            raise ValueError("carrier payload has trailing bytes")
        coeff_codes = coeff_codes.reshape(N, CARRIER_DIM).to(torch.int32) & 0xFFF
    if CARRIER_COEFF_DELTA_ZIGZAG:
        delta = (coeff_codes >> 1) ^ -(coeff_codes & 1)
        coeff_codes = torch.cumsum(delta, dim=0) & 0xFFF
        coeff_codes = torch.where(
            coeff_codes >= 0x800, coeff_codes - 0x1000, coeff_codes
        )
    basis = basis_codes.reshape(CARRIER_DIM, 3, CARRIER_H, CARRIER_W).float()
    basis = basis * basis_scales[:, None, None, None]
    coeff = coeff_codes.reshape(N, CARRIER_DIM).float() * coeff_scales[None]
    return semantic, basis, coeff


def patch_group_mask(kernel: int, delta: int, type_: str):
    mask = torch.zeros(kernel, kernel)
    center = (kernel - 1) // 2
    for row in range(kernel):
        for col in range(kernel):
            group_offset = col - center + delta * (row - center)
            if group_offset < 0 or (type_ == "B" and group_offset == 0):
                mask[row, col] = 1.0
    return mask


class MaskedConv2dPG(nn.Module):
    def __init__(self, c_in, c_out, kernel, padding=0, dilation=1,
                 groups=1, type_="B", delta=2):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in // groups, kernel, kernel))
        self.bias = nn.Parameter(torch.empty(c_out))
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.register_buffer(
            "mask", patch_group_mask(kernel, delta, type_).view(1, 1, kernel, kernel),
            persistent=False,
        )

    def forward(self, value):
        return F.conv2d(
            value, self.weight * self.mask, self.bias,
            padding=self.padding, dilation=self.dilation, groups=self.groups,
        )


class ChannelNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels))
        self.shift = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, value):
        mean = value.mean(dim=1, keepdim=True)
        variance = value.var(dim=1, keepdim=True, unbiased=False)
        value = (value - mean) / torch.sqrt(variance + self.eps)
        return value * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)


class CausalSPM(nn.Module):
    def __init__(self, channels: int, patch: int):
        super().__init__()
        self.P = patch
        self.norm = ChannelNorm2d(channels)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)

    def forward(self, value):
        batch, channels, height, width = value.shape
        patch_rows, patch_cols = height // self.P, width // self.P
        pooled = value.view(
            batch, channels, patch_rows, self.P, patch_cols, self.P
        ).mean(dim=(3, 5))
        pooled = F.gelu(self.dw(self.norm(pooled)))
        pooled = self.pw(pooled)
        expanded = pooled.unsqueeze(3).unsqueeze(5).expand(
            batch, channels, patch_rows, self.P, patch_cols, self.P
        ).contiguous()
        return expanded.view(batch, channels, height, width)


class HPACMini(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = NUM_CLASSES
        self.P = HPAC_PATCH
        self.delta = HPAC_DELTA
        self.ch = HPAC_CHANNELS
        self.frame_embed = nn.Embedding(N, HPAC_FILM_DIM)
        self.film_gen = nn.Linear(HPAC_FILM_DIM, HPAC_CHANNELS * 2)
        self.conv_a = MaskedConv2dPG(
            NUM_CLASSES + 2, HPAC_CHANNELS, 7, padding=3,
            type_="A", delta=HPAC_DELTA,
        )
        self.gn_a = ChannelNorm2d(HPAC_CHANNELS)
        self.conv_b1 = MaskedConv2dPG(
            HPAC_CHANNELS, HPAC_CHANNELS, 5, padding=4, dilation=2,
            groups=HPAC_CHANNELS, type_="B", delta=HPAC_DELTA,
        )
        self.gn_b1 = ChannelNorm2d(HPAC_CHANNELS)
        self.conv_b2 = MaskedConv2dPG(
            HPAC_CHANNELS, HPAC_CHANNELS, 3, padding=4, dilation=4,
            groups=HPAC_CHANNELS, type_="B", delta=HPAC_DELTA,
        )
        self.gn_b2 = ChannelNorm2d(HPAC_CHANNELS)
        self.conv_past = nn.Conv2d(NUM_CLASSES, HPAC_CHANNELS, 3, padding=1)
        self.spm = CausalSPM(HPAC_CHANNELS, HPAC_PATCH)
        self.head = nn.Conv2d(HPAC_CHANNELS, NUM_CLASSES, 1)
        self.register_buffer("_coord_cache", torch.zeros(0), persistent=False)
        self._cached_P = -1

    def _patch_coord_grid(self, batch: int, device: torch.device):
        if self._cached_P != self.P or self._coord_cache.numel() == 0:
            ys = torch.linspace(-1.0, 1.0, self.P, device=device).view(
                1, 1, self.P, 1
            ).expand(1, 1, self.P, self.P)
            xs = torch.linspace(-1.0, 1.0, self.P, device=device).view(
                1, 1, 1, self.P
            ).expand(1, 1, self.P, self.P)
            self._coord_cache = torch.cat([ys, xs], dim=1)
            self._cached_P = self.P
        return self._coord_cache.expand(batch, -1, -1, -1)

    def _to_patches(self, value):
        batch, channels, height, width = value.shape
        patch_rows, patch_cols = height // self.P, width // self.P
        value = value.view(
            batch, channels, patch_rows, self.P, patch_cols, self.P
        ).permute(0, 2, 4, 1, 3, 5).contiguous()
        return value.view(batch * patch_rows * patch_cols, channels, self.P, self.P)

    def _from_patches(self, value, batch: int, patch_rows: int, patch_cols: int):
        channels = value.shape[1]
        value = value.view(
            batch, patch_rows, patch_cols, channels, self.P, self.P
        ).permute(0, 3, 1, 4, 2, 5).contiguous()
        return value.view(
            batch, channels, patch_rows * self.P, patch_cols * self.P
        )


def reconstruct_hpac_state(packed: dict[str, torch.Tensor]):
    state = {}
    bases = sorted({
        key[:-len(".weight_q")] for key in packed
        if key.endswith(".weight_q") and key != "frame_embed.weight_q"
    })
    skipped = set()
    for base in bases:
        codes = packed[base + ".weight_q"].float()
        masked_specs = {
            "conv_a": (7, "A"),
            "conv_b1": (5, "B"),
            "conv_b2": (3, "B"),
        }
        if base in masked_specs:
            kernel, type_ = masked_specs[base]
            mask = patch_group_mask(kernel, HPAC_DELTA, type_).bool().flatten()
            full = torch.zeros(codes.shape[0], codes.shape[1], kernel * kernel)
            full[:, :, mask] = codes
            codes = full.reshape(codes.shape[0], codes.shape[1], kernel, kernel)
        scales = packed[base + ".weight_scale"].float()
        scale_shape = [1] * codes.ndim
        scale_shape[0] = -1
        state[base + ".weight"] = codes * scales.view(*scale_shape)
        skipped.update({base + ".weight_q", base + ".weight_scale"})
    state["frame_embed.weight"] = (
        packed["frame_embed.weight_q"].float()
        * packed["frame_embed.weight_scale"].float()[None]
    )
    skipped.update({"frame_embed.weight_q", "frame_embed.weight_scale"})
    for key, value in packed.items():
        if key not in skipped:
            state[key] = value.float() if torch.is_floating_point(value) else value
    return state


def deserialize_hpac_packed(raw: bytes):
    packed = {}
    offset = 0
    for name, shape, dtype in HPAC_PACKED_SCHEMA:
        np_dtype = np.dtype(dtype)
        count = math.prod(shape)
        byte_count = count * np_dtype.itemsize
        array = np.frombuffer(raw, dtype=np_dtype, count=count, offset=offset).copy()
        packed[name] = torch.from_numpy(array.reshape(shape))
        offset += byte_count
    if offset != len(raw):
        raise ValueError("packed HPAC blob has trailing bytes")
    return packed


def load_hpac(raw: bytes, device: torch.device):
    model = IntegerHPAC(
        num_pairs=N,
        num_classes=NUM_CLASSES,
        patch=HPAC_PATCH,
        delta=HPAC_DELTA,
        channels=HPAC_CHANNELS,
        frame_dim=HPAC_FILM_DIM,
        norm_mode="none",
        activation="relu",
        use_frame_scale=True,
        weight_bound=127,
        activation_bound=127,
        use_weight_scales=True,
        weight_exponent_min=-6,
        use_spm=True,
        use_norm_gates=False,
    ).eval()
    deserialize_integer_model(model, raw)
    return model.to(device)


def group_masks(device: torch.device):
    rows = torch.arange(HPAC_PATCH, device=device).view(HPAC_PATCH, 1)
    cols = torch.arange(HPAC_PATCH, device=device).view(1, HPAC_PATCH)
    grid = cols + HPAC_DELTA * rows
    patch_rows, patch_cols = EVAL_H // HPAC_PATCH, EVAL_W // HPAC_PATCH
    masks = []
    for group in range((1 + HPAC_DELTA) * HPAC_PATCH - HPAC_DELTA):
        local = grid == group
        full = local[None, None].expand(
            patch_rows, patch_cols, HPAC_PATCH, HPAC_PATCH
        )
        masks.append(full.permute(0, 2, 1, 3).reshape(EVAL_H, EVAL_W))
    return masks


def prepare_frame_context(model: HPACMini, idx, previous_raw):
    batch, height, width = previous_raw.shape
    patch_count = (height // model.P) * (width // model.P)
    film = model.film_gen(model.frame_embed(idx))
    scale, shift = film.chunk(2, dim=1)
    scale = scale.view(batch, 1, model.ch, 1, 1).expand(
        batch, patch_count, model.ch, 1, 1
    ).reshape(batch * patch_count, model.ch, 1, 1)
    shift = shift.view(batch, 1, model.ch, 1, 1).expand(
        batch, patch_count, model.ch, 1, 1
    ).reshape(batch * patch_count, model.ch, 1, 1)
    previous_one_hot = F.one_hot(
        previous_raw, num_classes=NUM_CLASSES
    ).permute(0, 3, 1, 2).float()
    past_full = model.conv_past(previous_one_hot)
    past = model._to_patches(past_full)
    spm = model._to_patches(model.spm(past_full))
    return scale, shift, past, spm


def cached_context_logits(model: HPACMini, current, context):
    batch, height, width = current.shape
    patch_rows, patch_cols = height // model.P, width // model.P
    patch_count = patch_rows * patch_cols
    one_hot = F.one_hot(current, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
    patches = model._to_patches(one_hot)
    coords = model._patch_coord_grid(batch * patch_count, current.device)
    hidden = model.gn_a(model.conv_a(torch.cat([patches, coords], dim=1)))
    scale, shift, past, spm = context
    hidden = F.gelu(hidden * (1.0 + scale) + shift)
    hidden = hidden + past
    hidden = hidden + spm
    hidden = F.gelu(model.gn_b1(model.conv_b1(hidden)))
    hidden = F.gelu(model.gn_b2(model.conv_b2(hidden)))
    return model._from_patches(model.head(hidden), batch, patch_rows, patch_cols)


def probability_table(selected_logits: torch.Tensor):
    quantized = selected_logits.mul(HPAC_LOGIT_PRECISION).round().clamp(
        -32768, 32767
    ).to(torch.int16)
    logits = quantized.cpu().numpy().astype(np.float64) / HPAC_LOGIT_PRECISION
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def hierarchical_decode(decoder, families, table: np.ndarray):
    top = table.argmax(axis=1).astype(np.int32)
    rows = np.arange(len(top))
    hit_probability = table[rows, top]
    binary_table = np.stack([1.0 - hit_probability, hit_probability], axis=1)
    hits = decoder.decode(families[0], binary_table.astype(np.float32))
    symbols = top.copy()
    miss_rows = np.flatnonzero(hits == 0)
    if len(miss_rows) == 0:
        return symbols
    classes = np.arange(NUM_CLASSES, dtype=np.int32)
    remaining = np.stack([classes[classes != top[row]] for row in miss_rows])
    miss_table = np.take_along_axis(table[miss_rows], remaining, axis=1)
    miss_symbols = decoder.decode(families[1], miss_table.astype(np.float32))
    symbols[miss_rows] = remaining[np.arange(len(miss_rows)), miss_symbols]
    return symbols


@torch.no_grad()
def decode_tokens(model: IntegerHPAC, blob: bytes, device: torch.device):
    if len(blob) % np.dtype("<u4").itemsize:
        raise ValueError("HPAC token payload length is not a multiple of four")
    decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(blob, dtype="<u4"))
    family = constriction.stream.model.Categorical(perfect=HPAC_CODER_PERFECT)
    hierarchical_families = (
        constriction.stream.model.Categorical(perfect=HPAC_CODER_PERFECT),
        constriction.stream.model.Categorical(perfect=HPAC_CODER_PERFECT),
    )
    masks = group_masks(device)
    sparse = SparseIntegerHPAC(model, EVAL_H, EVAL_W)
    tokens = torch.empty((N, EVAL_H, EVAL_W), dtype=torch.uint8)
    previous_raw = torch.zeros((1, EVAL_H, EVAL_W), dtype=torch.long, device=device)
    started = time.time()
    for frame in range(N):
        idx = torch.tensor([frame], dtype=torch.long, device=device)
        current = torch.zeros_like(previous_raw)
        context = model.prepare_frame_context(idx, previous_raw)
        for group, mask in enumerate(masks):
            selected = sparse.selected_logits(current, context, group)
            table = probability_table(selected)
            symbols = (
                hierarchical_decode(decoder, hierarchical_families, table)
                if HPAC_HIERARCHICAL
                else decoder.decode(family, table)
            )
            current[0, mask] = torch.from_numpy(symbols.astype(np.int64)).to(device)
        previous_raw = (
            current.clone()
            if HPAC_TARGET_MODE == "raw"
            else (current + previous_raw) % NUM_CLASSES
        )
        tokens[frame].copy_(previous_raw[0].to(torch.uint8).cpu())
        if frame == 0 or (frame + 1) % 50 == 0:
            print(
                f"decoded tokens {frame + 1}/{N} in {time.time() - started:.1f}s",
                flush=True,
            )
    return tokens


def normalized_basis(raw_basis: torch.Tensor):
    basis = F.interpolate(
        raw_basis, size=(EVAL_H, EVAL_W), mode="bicubic", align_corners=False
    )
    basis = basis - basis.mean(dim=(1, 2, 3), keepdim=True)
    rms = basis.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-5)
    return basis / rms


@torch.no_grad()
def render_video(semantic, basis, coeff, tokens, destination: Path, device):
    semantic = semantic.eval().to(device)
    basis = normalized_basis(basis.to(device))
    coeff = coeff.to(device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(
        destination, mode="w+", dtype=np.uint8,
        shape=(N * 2, CAMERA_H, CAMERA_W, 3),
    )
    started = time.time()
    semantic_batch = 8 if device.type == "cuda" else 1
    for start in range(0, N, semantic_batch):
        end = min(start + semantic_batch, N)
        idx = torch.arange(start, end, device=device)
        current_tokens = tokens[start:end].long().to(device)
        master_eval = semantic(current_tokens, idx)
        master = F.interpolate(
            master_eval, size=(CAMERA_H, CAMERA_W),
            mode="bilinear", align_corners=False,
        ).clamp(0.0, 255.0).round()
        master_np = master.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for offset in range(end - start):
            output[2 * (start + offset) + 1] = master_np[offset]
        if end % 50 == 0 or end == N:
            print(f"rendered masters {end}/{N} in {time.time() - started:.1f}s", flush=True)

    pose_batch = 64 if device.type == "cuda" else 1
    for start in range(0, N, pose_batch):
        end = min(start + pose_batch, N)
        carrier = torch.einsum("bk,kchw->bchw", coeff[start:end], basis)
        carrier = carrier / math.sqrt(CARRIER_DIM)
        slave_eval = (127.5 + CARRIER_AMPLITUDE * carrier).clamp(
            0.0, 255.0
        ).round()
        slave = F.interpolate(
            slave_eval, size=(CAMERA_H, CAMERA_W),
            mode="bicubic", align_corners=False,
        ).clamp(0.0, 255.0).round()
        slave_np = slave.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for offset in range(end - start):
            output[2 * (start + offset)] = slave_np[offset]
        if end % 64 == 0 or end == N:
            print(f"rendered carriers {end}/{N} in {time.time() - started:.1f}s", flush=True)
    output.flush()


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: inflate.py <archive-dir> <base> <destination.raw>")
    data_dir = Path(sys.argv[1])
    base = sys.argv[2]
    destination = Path(sys.argv[3])
    if base != "0":
        raise ValueError(f"unsupported public video: {base}")
    if not torch.cuda.is_available():
        raise RuntimeError("semantic_pose_landslide requires the official GPU rail")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    payload = (data_dir / "p").read_bytes()
    if len(payload) < 4:
        raise ValueError("combined payload is truncated before the model length")
    models_bytes = struct.unpack_from("<I", payload)[0]
    if len(payload) <= 4 + models_bytes:
        raise ValueError("combined payload is truncated")
    models_raw = lzma.decompress(payload[4:4 + models_bytes])
    if len(models_raw) < 8:
        raise ValueError("model bundle is truncated before semantic-pose lengths")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models_raw)
    semantic_pose_bytes = 8 + semantic_bytes + carrier_bytes
    if semantic_pose_bytes > len(models_raw):
        raise ValueError("semantic-pose lengths exceed the model bundle")
    semantic, basis, coeff = unpack_semantic_pose(
        models_raw[:semantic_pose_bytes]
    )
    hpac = load_hpac(models_raw[semantic_pose_bytes:], device)
    tokens = decode_tokens(hpac, payload[4 + models_bytes:], device)
    del hpac
    torch.cuda.empty_cache()
    render_video(semantic, basis, coeff, tokens, destination, device)
    print(f"wrote {destination} ({destination.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
