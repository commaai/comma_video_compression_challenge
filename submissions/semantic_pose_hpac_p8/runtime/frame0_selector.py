"""Decode the fixed sparse frame-0 selector used by F26."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

SPARSE_MAGIC = b"F0E1"
SPARSE_VERSION = 1
_HEADER = struct.Struct("<4sBH")

IDENTITY = 0
LUMA = 3
CHANNEL = 4
ROLL = 5
TILE = 6


class Frame0SelectorError(ValueError):
    """The fixed F26 selector is malformed or non-canonical."""


@dataclass(frozen=True, order=True)
class SelectorMode:
    """One integer-only frame-0 pixel operation."""

    kind: int
    a: int = 0
    b: int = 0
    c: int = 0


SPARSE_PIXEL_MODES = (
    SelectorMode(IDENTITY),
    SelectorMode(LUMA, 1),
    SelectorMode(LUMA, -1),
    SelectorMode(CHANNEL, 1, 0, -1),
    SelectorMode(ROLL, 1, 0),
    SelectorMode(ROLL, 0, 1),
    SelectorMode(TILE, 0, 1),
    SelectorMode(TILE, 3, 1),
)


def _combination_unrank(rank: int, count: int, frames: int) -> np.ndarray:
    if not 0 <= count <= frames or not 0 <= rank < math.comb(frames, count):
        raise Frame0SelectorError("selector combination rank is out of range")
    positions = np.empty(count, dtype=np.int64)
    upper = frames - 1
    remaining = rank
    for width in range(count, 0, -1):
        low, high = width - 1, upper
        while low < high:
            middle = (low + high + 1) // 2
            if math.comb(middle, width) <= remaining:
                low = middle
            else:
                high = middle - 1
        positions[width - 1] = low
        remaining -= math.comb(low, width)
        upper = low - 1
    if remaining:
        raise Frame0SelectorError("non-canonical selector combination rank")
    return positions


def _unpack_labels(payload: bytes, count: int) -> np.ndarray:
    bit_count = count * 3
    if len(payload) != (bit_count + 7) // 8:
        raise Frame0SelectorError("invalid selector label length")
    if bit_count % 8 and payload[-1] & ((1 << (8 - bit_count % 8)) - 1):
        raise Frame0SelectorError("non-zero selector label padding")
    labels = np.empty(count, dtype=np.uint8)
    cursor = 0
    for index in range(count):
        label = 0
        for _ in range(3):
            byte = payload[cursor // 8]
            shift = 7 - cursor % 8
            label = (label << 1) | ((byte >> shift) & 1)
            cursor += 1
        if label >= len(SPARSE_PIXEL_MODES) - 1:
            raise Frame0SelectorError("selector label is out of range")
        labels[index] = label + 1
    return labels


def decode_selector(payload: bytes) -> tuple[tuple[SelectorMode, ...], np.ndarray]:
    """Return the fixed mode catalog and one choice for each of 600 frames."""
    if len(payload) < _HEADER.size:
        raise Frame0SelectorError("truncated sparse selector header")
    magic, version, count = _HEADER.unpack_from(payload)
    if magic != SPARSE_MAGIC or version != SPARSE_VERSION or not 1 <= count <= 600:
        raise Frame0SelectorError("invalid sparse selector header")
    limit = math.comb(600, count)
    rank_bytes = ((limit - 1).bit_length() + 7) // 8
    label_bytes = (count * 3 + 7) // 8
    expected = _HEADER.size + rank_bytes + label_bytes
    if len(payload) != expected:
        raise Frame0SelectorError("truncated or trailing selector payload")
    offset = _HEADER.size
    rank = int.from_bytes(payload[offset : offset + rank_bytes], "big")
    positions = _combination_unrank(rank, count, 600)
    labels = _unpack_labels(payload[offset + rank_bytes :], count)
    choices = np.zeros(600, dtype=np.uint8)
    choices[positions] = labels
    return SPARSE_PIXEL_MODES, choices


def apply_pixel_mode(frames: np.ndarray, mode: SelectorMode) -> np.ndarray:
    """Apply one selector operation using deterministic integer arithmetic."""
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise Frame0SelectorError("selector requires BxHxWx3 uint8 frames")
    if mode.kind == IDENTITY:
        return values.copy()
    if mode.kind == ROLL:
        return np.roll(values, shift=(mode.b, mode.a), axis=(1, 2))
    if mode.kind == TILE:
        height, width = values.shape[1:3]
        yy, xx = np.indices((height, width), dtype=np.int32)
        if mode.a == 0:
            signs = ((yy + xx) & 1) * 2 - 1
        elif mode.a == 1:
            signs = (yy & 1) * 2 - 1
        elif mode.a == 2:
            signs = (xx & 1) * 2 - 1
        elif mode.a == 3:
            signs = (((yy >> 2) + (xx >> 2)) & 1) * 2 - 1
        else:
            raise Frame0SelectorError("invalid tile selector mode")
        delta = signs[None, :, :, None] * mode.b
    elif mode.kind == LUMA:
        delta = mode.a
    elif mode.kind == CHANNEL:
        delta = np.asarray((mode.a, mode.b, mode.c), dtype=np.int16).reshape(
            1, 1, 1, 3
        )
    else:
        raise Frame0SelectorError("unsupported F26 selector mode")
    return np.clip(values.astype(np.int16) + delta, 0, 255).astype(np.uint8)
