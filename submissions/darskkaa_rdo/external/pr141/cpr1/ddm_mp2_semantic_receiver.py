#!/usr/bin/env python3
"""Counted semantic-format receiver used by the DDM-MP2 candidate trees.

The receiver is deliberately additive.  Untagged semantic payloads return
``None`` so the unchanged HV1 legacy q4 parser remains authoritative.  The
only tagged formats accepted here are the retained SD1M mixed-precision
packet and the SM3R row-prune packets used by MZ2/MP2 candidates, in both the
uniform-q4 (mode 5) and per-tensor mixed-depth (mode 6) variants.
"""

from __future__ import annotations

import struct
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import torch

SD1M_MAGIC = b"SD1M"
SD1M_VERSION = 1
SM3R_MAGIC = b"SM3R"
SM3R_VERSION = 1
SM3R_ROW_PRUNE_MODE = 5
SM3R_ROW_PRUNE_MIXED_MODE = 6
DEFAULT_SEMANTIC_BITS = 4

SD1M_V1_NAMES = (
    "token_embed.weight",
    "frame_embed.weight",
    "coord_mix.weight",
    "blocks.0.dw.weight",
    "blocks.0.pw.weight",
    "blocks.0.film.weight",
    "blocks.1.dw.weight",
    "blocks.1.pw.weight",
    "blocks.1.film.weight",
    "blocks.2.dw.weight",
    "blocks.2.pw.weight",
    "blocks.2.film.weight",
    "blocks.3.dw.weight",
    "blocks.3.pw.weight",
    "blocks.3.film.weight",
    "head.weight",
)
ROW_PRUNE_NAMES = frozenset(
    {
        "blocks.1.film.weight",
        "blocks.2.film.weight",
        "blocks.3.film.weight",
    }
)


class MP2SemanticFormatError(ValueError):
    """A tagged semantic packet is malformed or unsupported."""


def _take(blob: memoryview, count: int, label: str) -> tuple[memoryview, memoryview]:
    if count < 0 or len(blob) < count:
        raise MP2SemanticFormatError(f"truncated {label}")
    return blob[:count], blob[count:]


def _quantized_names(template: Mapping[str, torch.Tensor]) -> list[str]:
    return [name for name, value in template.items() if value.ndim >= 2]


def _scale_count(name: str, value: torch.Tensor) -> int:
    return int(value.shape[-1] if name.endswith("embed.weight") else value.shape[0])


def _selection_mask(names: list[str], selected: frozenset[str]) -> int:
    if not selected.issubset(names):
        raise MP2SemanticFormatError("row-prune tensor set differs from receiver template")
    return sum(1 << index for index, name in enumerate(names) if name in selected)


def _unpack_signed_bits(
    blob: memoryview,
    count: int,
    bits: int,
) -> tuple[torch.Tensor, memoryview]:
    if count < 0 or not 2 <= bits <= 8:
        raise MP2SemanticFormatError("invalid signed bit-stream geometry")
    byte_count = (count * bits + 7) // 8
    packed_view, remaining = _take(blob, byte_count, "signed code stream")
    packed = np.frombuffer(packed_view, dtype=np.uint8)
    bitstream = np.unpackbits(packed, bitorder="little")[: count * bits]
    if bitstream.size != count * bits:
        raise MP2SemanticFormatError("signed code stream ended early")
    bitstream = bitstream.reshape(count, bits).astype(np.int16, copy=False)
    shifts = (1 << np.arange(bits, dtype=np.int16))[None]
    unsigned = (bitstream * shifts).sum(axis=1, dtype=np.int16)
    sign = 1 << (bits - 1)
    values = np.where(unsigned >= sign, unsigned - (1 << bits), unsigned)
    return torch.from_numpy(values.astype(np.int8, copy=False)), remaining


def _decode_quantized(
    name: str,
    template: torch.Tensor,
    blob: memoryview,
    bits: int,
) -> tuple[torch.Tensor, memoryview]:
    scale_count = _scale_count(name, template)
    scale_view, remaining = _take(blob, scale_count * 2, f"fp16 scales for {name}")
    scales = np.frombuffer(scale_view, dtype="<f2").copy()
    codes, remaining = _unpack_signed_bits(remaining, template.numel(), bits)
    scale_shape = [1] * template.ndim
    scale_shape[-1 if name.endswith("embed.weight") else 0] = scale_count
    restored = codes.reshape(template.shape).float()
    restored *= torch.from_numpy(scales).float().reshape(scale_shape)
    return restored, remaining


def _decode_fp16(
    name: str,
    template: torch.Tensor,
    blob: memoryview,
) -> tuple[torch.Tensor, memoryview]:
    payload, remaining = _take(blob, template.numel() * 2, f"fp16 tensor {name}")
    array = np.frombuffer(payload, dtype="<f2").copy()
    return torch.from_numpy(array.reshape(template.shape)).float(), remaining


def _decode_depth_nibbles(
    blob: memoryview,
    count: int,
    label: str,
) -> tuple[list[int], memoryview]:
    """Read a packed per-tensor bit-depth table (two four-bit depths per byte)."""

    depth_bytes = (count + 1) // 2
    depth_view, remaining = _take(blob, depth_bytes, f"{label} depth allocation")
    packed = np.frombuffer(depth_view, dtype=np.uint8)
    depths_array = np.empty(depth_bytes * 2, dtype=np.uint8)
    depths_array[0::2] = packed & 0xF
    depths_array[1::2] = packed >> 4
    depths = depths_array[:count].astype(int).tolist()
    if any(depth < 2 or depth > 8 for depth in depths):
        raise MP2SemanticFormatError(f"invalid {label} bit depth")
    return depths, remaining


def _decode_sd1m(
    blob: bytes,
    template: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    names = _quantized_names(template)
    if tuple(names) != SD1M_V1_NAMES:
        raise MP2SemanticFormatError("SD1M v1 semantic template order differs")
    if len(blob) < 6 or not blob.startswith(SD1M_MAGIC):
        raise MP2SemanticFormatError("truncated SD1M header")
    version = blob[4]
    count = blob[5]
    if version != SD1M_VERSION or count != len(names):
        raise MP2SemanticFormatError("unsupported SD1M header")
    depths, remaining = _decode_depth_nibbles(memoryview(blob)[6:], count, "SD1M")
    allocation = dict(zip(names, depths, strict=True))
    restored: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in template.items():
        if value.ndim < 2:
            restored[name], remaining = _decode_fp16(name, value, remaining)
        else:
            restored[name], remaining = _decode_quantized(
                name,
                value,
                remaining,
                allocation[name],
            )
    if remaining:
        raise MP2SemanticFormatError(f"SD1M payload has {len(remaining)} trailing bytes")
    return restored


def _decode_row_prune(
    blob: bytes,
    template: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    if len(blob) < 10 or not blob.startswith(SM3R_MAGIC):
        raise MP2SemanticFormatError("truncated SM3R row-prune header")
    version, mode, keep_percent, reserved = blob[4:8]
    if version != SM3R_VERSION or mode != SM3R_ROW_PRUNE_MODE or reserved != 0:
        raise MP2SemanticFormatError("unsupported SM3R row-prune header")
    if not 1 <= keep_percent < 100:
        raise MP2SemanticFormatError("invalid SM3R row-prune keep percentage")
    remaining = memoryview(blob)[8:]
    mask_view, remaining = _take(remaining, 2, "SM3R row-prune selection mask")
    mask = struct.unpack_from("<H", mask_view)[0]
    names = _quantized_names(template)
    if mask != _selection_mask(names, ROW_PRUNE_NAMES):
        raise MP2SemanticFormatError("SM3R row-prune selection mask differs")

    restored: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in template.items():
        if value.ndim < 2:
            restored[name], remaining = _decode_fp16(name, value, remaining)
        elif name not in ROW_PRUNE_NAMES:
            restored[name], remaining = _decode_quantized(name, value, remaining, 4)
        else:
            rows = int(value.shape[0])
            mask_view, remaining = _take(
                remaining,
                (rows + 7) // 8,
                f"SM3R selected rows for {name}",
            )
            selected = np.unpackbits(
                np.frombuffer(mask_view, dtype=np.uint8),
                bitorder="little",
            )[:rows].astype(bool)
            expected_keep = max(1, round(rows * keep_percent / 100.0))
            if int(selected.sum()) != expected_keep:
                raise MP2SemanticFormatError(f"SM3R row count differs for {name}")
            columns = value.numel() // rows
            compact_template = torch.empty((expected_keep, columns), dtype=torch.float32)
            compact, remaining = _decode_quantized(
                "pruned.rows",
                compact_template,
                remaining,
                4,
            )
            dense = torch.zeros((rows, columns), dtype=torch.float32)
            dense[torch.from_numpy(selected)] = compact
            restored[name] = dense.reshape(value.shape)
    if remaining:
        raise MP2SemanticFormatError(
            f"SM3R row-prune payload has {len(remaining)} trailing bytes"
        )
    return restored


def _decode_row_prune_mixed(
    blob: bytes,
    template: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """SM3R row-prune carrying an SD1M-style per-tensor bit-depth table.

    Identical geometry to :func:`_decode_row_prune`; the depth table follows the
    selection mask and supplies the code width for every rank>=2 tensor,
    including the surviving rows of a pruned tensor.
    """

    if len(blob) < 10 or not blob.startswith(SM3R_MAGIC):
        raise MP2SemanticFormatError("truncated SM3R mixed row-prune header")
    version, mode, keep_percent, reserved = blob[4:8]
    if version != SM3R_VERSION or mode != SM3R_ROW_PRUNE_MIXED_MODE or reserved != 0:
        raise MP2SemanticFormatError("unsupported SM3R mixed row-prune header")
    if not 1 <= keep_percent < 100:
        raise MP2SemanticFormatError("invalid SM3R row-prune keep percentage")
    remaining = memoryview(blob)[8:]
    mask_view, remaining = _take(remaining, 2, "SM3R row-prune selection mask")
    mask = struct.unpack_from("<H", mask_view)[0]
    names = _quantized_names(template)
    if mask != _selection_mask(names, ROW_PRUNE_NAMES):
        raise MP2SemanticFormatError("SM3R row-prune selection mask differs")
    depths, remaining = _decode_depth_nibbles(remaining, len(names), "SM3R")
    allocation = dict(zip(names, depths, strict=True))

    restored: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in template.items():
        if value.ndim < 2:
            restored[name], remaining = _decode_fp16(name, value, remaining)
        elif name not in ROW_PRUNE_NAMES:
            restored[name], remaining = _decode_quantized(
                name,
                value,
                remaining,
                allocation[name],
            )
        else:
            rows = int(value.shape[0])
            mask_view, remaining = _take(
                remaining,
                (rows + 7) // 8,
                f"SM3R selected rows for {name}",
            )
            selected = np.unpackbits(
                np.frombuffer(mask_view, dtype=np.uint8),
                bitorder="little",
            )[:rows].astype(bool)
            expected_keep = max(1, round(rows * keep_percent / 100.0))
            if int(selected.sum()) != expected_keep:
                raise MP2SemanticFormatError(f"SM3R row count differs for {name}")
            columns = value.numel() // rows
            compact_template = torch.empty((expected_keep, columns), dtype=torch.float32)
            compact, remaining = _decode_quantized(
                "pruned.rows",
                compact_template,
                remaining,
                allocation[name],
            )
            dense = torch.zeros((rows, columns), dtype=torch.float32)
            dense[torch.from_numpy(selected)] = compact
            restored[name] = dense.reshape(value.shape)
    if remaining:
        raise MP2SemanticFormatError(
            f"SM3R mixed row-prune payload has {len(remaining)} trailing bytes"
        )
    return restored


def unpack_variant_semantic_or_none(
    blob: bytes,
    template: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor] | None:
    """Decode an MP2 tagged semantic packet or return ``None`` if untagged."""

    if blob.startswith(SD1M_MAGIC):
        return _decode_sd1m(blob, template)
    if blob.startswith(SM3R_MAGIC):
        if len(blob) < 6:
            raise MP2SemanticFormatError("truncated SM3R header")
        if blob[5] == SM3R_ROW_PRUNE_MODE:
            return _decode_row_prune(blob, template)
        if blob[5] == SM3R_ROW_PRUNE_MIXED_MODE:
            return _decode_row_prune_mixed(blob, template)
        raise MP2SemanticFormatError(f"unsupported SM3R mode {blob[5]}")
    return None


__all__ = [
    "ROW_PRUNE_NAMES",
    "SD1M_MAGIC",
    "SM3R_MAGIC",
    "SM3R_ROW_PRUNE_MIXED_MODE",
    "SM3R_ROW_PRUNE_MODE",
    "MP2SemanticFormatError",
    "unpack_variant_semantic_or_none",
]
