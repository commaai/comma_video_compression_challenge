"""Renderer primitives used by the F26 archive decoder."""

from __future__ import annotations

import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
from carrier_codec import MAGIC as COMPACT_CARRIER_MAGIC
from carrier_codec import decode_compact_carrier
from hpac_integer import IntegerHPAC
from integer_model_io import deserialize_integer_model
from ddm_mp2_semantic_receiver import unpack_variant_semantic_or_none
from torch import nn
from torch.nn import functional

N = 600
NUM_CLASSES = 5
EVAL_H, EVAL_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
SEMANTIC_WIDTH = 96
SEMANTIC_WIDTH_BY_PAYLOAD_BYTES = {30_748: 80, 40_252: 96}
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
    bitstream = np.unpackbits(packed, bitorder="little")[: count * bits]
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
        return value + functional.gelu(residual)


class SemanticTokenRenderer(nn.Module):
    def __init__(self, width: int = SEMANTIC_WIDTH):
        super().__init__()
        self.token_embed = nn.Embedding(NUM_CLASSES, width)
        self.frame_embed = nn.Embedding(N, SEMANTIC_FRAME_DIM)
        self.coord_mix = nn.Conv2d(width + 4, width, 1)
        self.blocks = nn.ModuleList(
            [
                TokenBlock(width, SEMANTIC_FRAME_DIM, dilation)
                for dilation in (1, 1, 2, 4)
            ]
        )
        self.head = nn.Conv2d(width, 3, 3, padding=1)

    @staticmethod
    def coordinates(batch: int, device: torch.device, dtype: torch.dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, EVAL_H, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, EVAL_W, device=device, dtype=dtype),
            indexing="ij",
        )
        coordinates = torch.stack([xx, yy, xx.square(), yy.square()], dim=0)
        return coordinates.unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, tokens: torch.Tensor, pair_indices: torch.Tensor):
        value = self.token_embed(tokens).permute(0, 3, 1, 2)
        value = self.coord_mix(
            torch.cat(
                [
                    value,
                    self.coordinates(value.shape[0], value.device, value.dtype),
                ],
                dim=1,
            )
        )
        frame = self.frame_embed(pair_indices)
        for block in self.blocks:
            value = block(value, frame)
        return torch.sigmoid(self.head(functional.gelu(value))) * 255.0


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
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", raw)
    if len(raw) != 8 + semantic_bytes + carrier_bytes:
        raise ValueError("semantic-pose payload length mismatch")
    semantic_blob = raw[8 : 8 + semantic_bytes]
    carrier_blob = memoryview(raw[8 + semantic_bytes :])
    tagged_state = None
    if semantic_blob.startswith((b"SD1M", b"SM3R")):
        semantic_width = SEMANTIC_WIDTH
    else:
        try:
            semantic_width = SEMANTIC_WIDTH_BY_PAYLOAD_BYTES[semantic_bytes]
        except KeyError as error:
            raise ValueError(
                f"unsupported semantic payload size: {semantic_bytes} bytes"
            ) from error
    semantic = SemanticTokenRenderer(semantic_width)
    tagged_state = unpack_variant_semantic_or_none(
        semantic_blob,
        semantic.state_dict(),
    )
    if tagged_state is None:
        tagged_state = unpack_semantic(semantic_blob, semantic.state_dict())
    semantic.load_state_dict(tagged_state, strict=True)

    scale_bytes = CARRIER_DIM * 4
    basis_count = CARRIER_DIM * 3 * CARRIER_H * CARRIER_W
    coefficient_count = N * CARRIER_DIM
    if carrier_blob[:4] == COMPACT_CARRIER_MAGIC:
        (
            basis_scales_array,
            basis_codes_array,
            coefficient_scales_array,
            encoded_coefficients,
        ) = decode_compact_carrier(
            carrier_blob,
            basis_count=basis_count,
            frames=N,
            dimensions=CARRIER_DIM,
        )
        basis_scales = torch.from_numpy(basis_scales_array)
        basis_codes = torch.from_numpy(basis_codes_array.astype(np.int8, copy=False))
        coefficient_scales = torch.from_numpy(coefficient_scales_array)
        coefficient_codes = torch.from_numpy(
            encoded_coefficients.astype(np.int32, copy=False)
        )
    else:
        basis_scales = torch.from_numpy(
            np.frombuffer(carrier_blob[:scale_bytes], dtype="<f4").copy()
        )
        carrier_blob = carrier_blob[scale_bytes:]
        coefficient_bytes = ((coefficient_count + 1) // 2) * 3
        basis_bytes = carrier_bytes - 2 * scale_bytes - coefficient_bytes
        if basis_bytes <= 0 or (basis_bytes * 8) % basis_count:
            raise ValueError("carrier payload has an invalid packed basis length")
        basis_bits = basis_bytes * 8 // basis_count
        if not 4 <= basis_bits <= 8:
            raise ValueError(f"unsupported carrier basis precision: {basis_bits}")
        basis_codes, carrier_blob = unpack_signed_bits(
            carrier_blob, basis_count, basis_bits
        )
        coefficient_scales = torch.from_numpy(
            np.frombuffer(carrier_blob[:scale_bytes], dtype="<f4").copy()
        )
        carrier_blob = carrier_blob[scale_bytes:]
        if CARRIER_COEFF_BITS != 12:
            raise ValueError("submission payload expects signed int12 coefficients")
        coefficient_codes, carrier_blob = unpack_signed_int12(
            carrier_blob, coefficient_count
        )
        if carrier_blob:
            raise ValueError("carrier payload has trailing bytes")
        coefficient_codes = coefficient_codes.reshape(N, CARRIER_DIM).to(torch.int32)
        coefficient_codes &= 0xFFF
    if CARRIER_COEFF_DELTA_ZIGZAG:
        delta = (coefficient_codes >> 1) ^ -(coefficient_codes & 1)
        coefficient_codes = torch.cumsum(delta, dim=0) & 0xFFF
        coefficient_codes = torch.where(
            coefficient_codes >= 0x800,
            coefficient_codes - 0x1000,
            coefficient_codes,
        )
    basis = basis_codes.reshape(CARRIER_DIM, 3, CARRIER_H, CARRIER_W).float()
    basis = basis * basis_scales[:, None, None, None]
    coefficients = (
        coefficient_codes.reshape(N, CARRIER_DIM).float() * coefficient_scales[None]
    )
    return semantic, basis, coefficients


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
    columns = torch.arange(HPAC_PATCH, device=device).view(1, HPAC_PATCH)
    grid = columns + HPAC_DELTA * rows
    patch_rows, patch_columns = EVAL_H // HPAC_PATCH, EVAL_W // HPAC_PATCH
    masks = []
    for group in range((1 + HPAC_DELTA) * HPAC_PATCH - HPAC_DELTA):
        local = grid == group
        full = local[None, None].expand(
            patch_rows, patch_columns, HPAC_PATCH, HPAC_PATCH
        )
        masks.append(full.permute(0, 2, 1, 3).reshape(EVAL_H, EVAL_W))
    return masks


def normalized_basis(raw_basis: torch.Tensor):
    basis = functional.interpolate(
        raw_basis, size=(EVAL_H, EVAL_W), mode="bicubic", align_corners=False
    )
    basis = basis - basis.mean(dim=(1, 2, 3), keepdim=True)
    rms = basis.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-5)
    return basis / rms


@torch.no_grad()
def render_video(semantic, basis, coefficients, tokens, destination: Path, device):
    semantic = semantic.eval().to(device)
    basis = normalized_basis(basis.to(device))
    coefficients = coefficients.to(device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(
        destination,
        mode="w+",
        dtype=np.uint8,
        shape=(N * 2, CAMERA_H, CAMERA_W, 3),
    )
    started = time.time()
    semantic_batch = 8 if device.type == "cuda" else 1
    for start in range(0, N, semantic_batch):
        end = min(start + semantic_batch, N)
        indices = torch.arange(start, end, device=device)
        master = (
            functional.interpolate(
                semantic(tokens[start:end].long().to(device), indices),
                size=(CAMERA_H, CAMERA_W),
                mode="bilinear",
                align_corners=False,
            )
            .clamp(0.0, 255.0)
            .round()
        )
        master_np = master.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for offset in range(end - start):
            output[2 * (start + offset) + 1] = master_np[offset]
        if end % 50 == 0 or end == N:
            print(
                f"rendered masters {end}/{N} in {time.time() - started:.1f}s",
                flush=True,
            )

    pose_batch = 64 if device.type == "cuda" else 1
    for start in range(0, N, pose_batch):
        end = min(start + pose_batch, N)
        carrier = torch.einsum("bk,kchw->bchw", coefficients[start:end], basis)
        carrier = carrier / math.sqrt(CARRIER_DIM)
        slave = (
            functional.interpolate(
                (127.5 + CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0).round(),
                size=(CAMERA_H, CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            .clamp(0.0, 255.0)
            .round()
        )
        slave_np = slave.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for offset in range(end - start):
            output[2 * (start + offset)] = slave_np[offset]
        if end % 64 == 0 or end == N:
            print(
                f"rendered carriers {end}/{N} in {time.time() - started:.1f}s",
                flush=True,
            )
    output.flush()
