"""Lossless signed-row bitplane storage for the attributed PR #135 F24S model.

The original HPAC architecture, trained values, renderer, and entropy-coded
tokens are unchanged. Grouping integer rows by their stored signed width and
zigzagging their values exposes magnitude bitplanes to the outer LZMA codec.
This module restores the entire original F24S byte string before inference.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

MAGIC = b"HRP1"
_HEADER = struct.Struct("<4sHII")  # magic, tail bytes, bitplane bytes, original CRC32
_IHS_PREFIX = b"IHS2\x03\x31"
_IHS_BYTES = 16_599
_DEPTH_COUNT = 517
_DEPTH_BYTES = (_DEPTH_COUNT + 1) // 2
_FRONT_BYTES = len(_IHS_PREFIX) + 2 * _DEPTH_BYTES
_MAX_MODELS_BYTES = 1 << 20
# Values describe the published architecture only, never trained coefficients.
_ROW_COUNTS = tuple([8] * 64 + [8] * 64 + [161] * 64 + [14] * 64 +
                    [5] * 64 + [45] * 64 + [9] * 64 + [64] * 64 + [64] * 5)


class RowpackError(ValueError):
    """A row-packed model does not have the supported canonical representation."""


def _depths(front: bytes) -> np.ndarray:
    if len(front) != _FRONT_BYTES or front[:6] != _IHS_PREFIX:
        raise RowpackError("unsupported HPAC header or depth-table length")
    tables = []
    for start in (6, 6 + _DEPTH_BYTES):
        raw = np.frombuffer(front[start:start + _DEPTH_BYTES], dtype=np.uint8)
        if raw[-1] >> 4:
            raise RowpackError("nonzero depth-table padding")
        values = np.empty(raw.size * 2, dtype=np.uint8)
        values[0::2], values[1::2] = raw & 15, raw >> 4
        values = values[:_DEPTH_COUNT]
        if np.any(values > 8):
            raise RowpackError("unsupported weight depth above eight bits")
        tables.append(values)
    original, stored = tables
    if np.any(stored > original):
        raise RowpackError("stored depth exceeds original depth")
    return stored


def _weight_bytes(depths: np.ndarray) -> int:
    return (sum(int(depth) * count for depth, count in zip(depths, _ROW_COUNTS)) + 7) // 8


def _unpack_rows(packed: bytes, depths: np.ndarray) -> tuple[np.ndarray, ...]:
    if len(packed) != _weight_bytes(depths):
        raise RowpackError("incorrect packed-weight length")
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
    rows = []
    offset = 0
    for depth_value, count in zip(depths, _ROW_COUNTS):
        depth = int(depth_value)
        if not depth:
            rows.append(np.zeros(count, dtype=np.int16))
            continue
        end = offset + count * depth
        unsigned = (bits[offset:end].reshape(count, depth).astype(np.int16) *
                    (1 << np.arange(depth, dtype=np.int16))).sum(axis=1)
        signed = np.where(unsigned >= 1 << (depth - 1), unsigned - (1 << depth), unsigned)
        rows.append(signed.astype(np.int16))
        offset = end
    if np.any(bits[offset:]):
        raise RowpackError("nonzero packed-weight padding")
    return tuple(rows)


def _pack_rows(rows: tuple[np.ndarray, ...], depths: np.ndarray) -> bytes:
    if len(rows) != _DEPTH_COUNT:
        raise RowpackError("wrong number of weight rows")
    chunks = []
    for row, depth_value, count in zip(rows, depths, _ROW_COUNTS):
        depth = int(depth_value)
        row = np.asarray(row)
        if row.shape != (count,) or row.dtype.kind not in "iu":
            raise RowpackError("invalid weight-row shape or dtype")
        if not depth:
            if np.any(row):
                raise RowpackError("nonzero value in an implicit zero row")
            continue
        if np.any(row < -(1 << (depth - 1))) or np.any(row >= 1 << (depth - 1)):
            raise RowpackError("weight value exceeds stored signed depth")
        codes = row.astype(np.int16) & ((1 << depth) - 1)
        chunks.append(((codes[:, None] >> np.arange(depth)) & 1).astype(np.uint8).ravel())
    bits = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint8)
    return np.packbits(bits, bitorder="little").tobytes()


def _plane_size(depths: np.ndarray) -> int:
    return sum(depth * ((sum(count for actual, count in zip(depths, _ROW_COUNTS)
                             if actual == depth) + 7) // 8) for depth in range(1, 9))


def _encode_planes(rows: tuple[np.ndarray, ...], depths: np.ndarray) -> bytes:
    pieces = []
    for depth in range(1, 9):
        indices = [i for i in range(_DEPTH_COUNT) if depths[i] == depth]
        if not indices:
            continue
        signed = np.concatenate([rows[i] for i in indices]).astype(np.int16)
        unsigned = np.where(signed < 0, -2 * signed - 1, 2 * signed)
        for bit in range(depth):
            pieces.append(np.packbits(((unsigned >> bit) & 1).astype(np.uint8), bitorder="little").tobytes())
    return b"".join(pieces)


def _decode_planes(data: bytes, depths: np.ndarray) -> tuple[np.ndarray, ...]:
    if len(data) != _plane_size(depths):
        raise RowpackError("incorrect bitplane length")
    rows = [np.zeros(count, dtype=np.int16) for count in _ROW_COUNTS]
    cursor = 0
    for depth in range(1, 9):
        indices = [i for i in range(_DEPTH_COUNT) if depths[i] == depth]
        count = sum(_ROW_COUNTS[i] for i in indices)
        if not count:
            continue
        values = np.zeros(count, dtype=np.int16)
        length = (count + 7) // 8
        for bit in range(depth):
            raw = np.frombuffer(data[cursor:cursor + length], dtype=np.uint8)
            bits = np.unpackbits(raw, bitorder="little")
            if np.any(bits[count:]):
                raise RowpackError("nonzero bitplane padding")
            values |= bits[:count].astype(np.int16) << bit
            cursor += length
        signed = np.where(values & 1, -(values // 2) - 1, values // 2)
        offset = 0
        for index in indices:
            count = _ROW_COUNTS[index]
            rows[index] = signed[offset:offset + count].copy()
            offset += count
    if cursor != len(data):
        raise RowpackError("trailing bitplane bytes")
    return tuple(rows)


def pack_models(models: bytes) -> bytes:
    """Encode an F24S model section, retaining every trained value exactly."""
    if not isinstance(models, bytes) or not 4 + _IHS_BYTES < len(models) <= _MAX_MODELS_BYTES:
        raise RowpackError("invalid F24S model-section length")
    if not models.startswith(b"F24S"):
        raise RowpackError("expected an original F24S model section")
    hpac_end = 4 + _IHS_BYTES - len(_IHS_PREFIX)
    hpac = _IHS_PREFIX + models[4:hpac_end]
    front = hpac[:_FRONT_BYTES]
    depths = _depths(front)
    end = _FRONT_BYTES + _weight_bytes(depths)
    if end > len(hpac):
        raise RowpackError("weight rows overrun the HPAC payload")
    rows = _unpack_rows(hpac[_FRONT_BYTES:end], depths)
    planes = _encode_planes(rows, depths)
    tail = hpac[end:]
    encoded = (_HEADER.pack(MAGIC, len(tail), len(planes), zlib.crc32(models)) +
               front + tail + planes + models[hpac_end:])
    if unpack_models(encoded) != models:
        raise RowpackError("row-packed model failed its exact roundtrip")
    return encoded


def unpack_models(encoded: bytes) -> bytes:
    """Recover original F24S bytes with strict bounds, padding, and CRC checks."""
    minimum = _HEADER.size + _FRONT_BYTES
    if not isinstance(encoded, bytes) or not minimum < len(encoded) <= _MAX_MODELS_BYTES:
        raise RowpackError("invalid row-packed model-section length")
    magic, tail_size, plane_size, crc = _HEADER.unpack_from(encoded)
    if magic != MAGIC:
        raise RowpackError("invalid row-packed model magic")
    front = encoded[_HEADER.size:minimum]
    depths = _depths(front)
    if tail_size != _IHS_BYTES - _FRONT_BYTES - _weight_bytes(depths):
        raise RowpackError("incorrect retained HPAC-tail length")
    if plane_size != _plane_size(depths):
        raise RowpackError("incorrect bitplane size in header")
    tail_end = minimum + tail_size
    plane_end = tail_end + plane_size
    if plane_end >= len(encoded):
        raise RowpackError("truncated row-packed model section")
    rows = _decode_planes(encoded[tail_end:plane_end], depths)
    hpac = front + _pack_rows(rows, depths) + encoded[minimum:tail_end]
    restored = b"F24S" + hpac[len(_IHS_PREFIX):] + encoded[plane_end:]
    if zlib.crc32(restored) != crc:
        raise RowpackError("row-packed model CRC32 mismatch")
    return restored
