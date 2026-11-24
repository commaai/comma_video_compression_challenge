"""Strict joint sparse overlay for QS2 frame-0 compensation codes.

The counted payload carries sorted pair indices, one 12-bit support mask per
pair, and three-bit signed code deltas in the measured QS1 domain [-3, 4].
The decoder applies those deltas to the real 600x12 signed-int12 CP135 code
lattice.  It is copied byte-for-byte into the adapted contest runtime.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

import numpy as np

MAGIC = b"Q2C1"
VERSION = 1
PAIR_COUNT = 600
DIMENSIONS = 12
MIN_DELTA = -3
MAX_DELTA = 4
_SELECTOR_HEADER = struct.Struct("<4sBH")


class CompensationOverlayError(ValueError):
    """The counted QS2 overlay or its receiving code lattice is invalid."""


def _append_bits(bits: list[int], value: int, width: int) -> None:
    if width <= 0 or not 0 <= value < 1 << width:
        raise CompensationOverlayError("bit field value is out of range")
    bits.extend((value >> shift) & 1 for shift in range(width - 1, -1, -1))


def _take_bits(bits: np.ndarray, cursor: int, width: int) -> tuple[int, int]:
    if width <= 0 or cursor + width > bits.size:
        raise CompensationOverlayError("truncated compensation bitstream")
    value = 0
    for bit in bits[cursor : cursor + width]:
        value = (value << 1) | int(bit)
    return value, cursor + width


def encode_compensation_overlay(
    pair_indices: Sequence[int], deltas: np.ndarray
) -> bytes:
    """Encode a sorted sparse set of measured-domain int12 code deltas."""
    pairs = tuple(int(value) for value in pair_indices)
    values = np.asarray(deltas, dtype=np.int32)
    if not 1 <= len(pairs) <= 15 or values.shape != (len(pairs), DIMENSIONS):
        raise CompensationOverlayError("overlay geometry differs")
    if tuple(sorted(pairs)) != pairs or len(set(pairs)) != len(pairs):
        raise CompensationOverlayError("pair indices must be sorted and unique")
    if pairs[0] < 0 or pairs[-1] >= PAIR_COUNT:
        raise CompensationOverlayError("pair index exceeds the n600 domain")
    if np.any(values < MIN_DELTA) or np.any(values > MAX_DELTA):
        raise CompensationOverlayError("delta exceeds the measured three-bit domain")
    masks = np.zeros(len(pairs), dtype=np.uint16)
    for row in range(len(pairs)):
        for dimension in range(DIMENSIONS):
            if int(values[row, dimension]):
                masks[row] |= np.uint16(1 << dimension)
        if not int(masks[row]):
            raise CompensationOverlayError("zero-support pair is non-canonical")

    bits: list[int] = []
    for pair in pairs:
        _append_bits(bits, pair, 10)
    for mask in masks:
        _append_bits(bits, int(mask), DIMENSIONS)
    for row, mask in enumerate(masks):
        for dimension in range(DIMENSIONS):
            if int(mask) & (1 << dimension):
                _append_bits(bits, int(values[row, dimension]) - MIN_DELTA, 3)
    packed = np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big").tobytes()
    header = MAGIC + bytes(((VERSION << 4) | len(pairs),))
    return header + packed


def decode_compensation_overlay(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode and reject truncation, aliases, nonzero padding, and zero rows."""
    if len(payload) < len(MAGIC) + 2 or not payload.startswith(MAGIC):
        raise CompensationOverlayError("invalid compensation overlay magic or length")
    version_count = payload[len(MAGIC)]
    version, count = version_count >> 4, version_count & 0x0F
    if version != VERSION or not 1 <= count <= 15:
        raise CompensationOverlayError("unsupported compensation overlay header")
    bits = np.unpackbits(
        np.frombuffer(payload[len(MAGIC) + 1 :], dtype=np.uint8), bitorder="big"
    )
    cursor = 0
    pairs = np.empty(count, dtype=np.int16)
    for index in range(count):
        pairs[index], cursor = _take_bits(bits, cursor, 10)
    if (
        np.any(pairs < 0)
        or np.any(pairs >= PAIR_COUNT)
        or np.any(pairs[1:] <= pairs[:-1])
    ):
        raise CompensationOverlayError("pair index order or domain differs")
    masks = np.empty(count, dtype=np.uint16)
    for index in range(count):
        masks[index], cursor = _take_bits(bits, cursor, DIMENSIONS)
    if np.any(masks == 0):
        raise CompensationOverlayError("zero-support pair is non-canonical")
    values = np.zeros((count, DIMENSIONS), dtype=np.int32)
    for row, mask in enumerate(masks):
        for dimension in range(DIMENSIONS):
            if int(mask) & (1 << dimension):
                encoded, cursor = _take_bits(bits, cursor, 3)
                delta = encoded + MIN_DELTA
                if not MIN_DELTA <= delta <= MAX_DELTA or delta == 0:
                    raise CompensationOverlayError(
                        "non-canonical encoded compensation delta"
                    )
                values[row, dimension] = delta
    expected_bytes = (cursor + 7) // 8
    if len(payload) != len(MAGIC) + 1 + expected_bytes:
        raise CompensationOverlayError("compensation overlay has trailing bytes")
    if cursor < bits.size and np.any(bits[cursor:]):
        raise CompensationOverlayError("compensation overlay has nonzero padding")
    return pairs, values


def apply_compensation_overlay(base_codes: np.ndarray, payload: bytes) -> np.ndarray:
    """Apply the counted overlay to the actual signed-int12 receiver lattice."""
    codes = np.asarray(base_codes, dtype=np.int32)
    if codes.shape != (PAIR_COUNT, DIMENSIONS):
        raise CompensationOverlayError("base carrier code geometry differs")
    pairs, deltas = decode_compensation_overlay(payload)
    result = codes.copy()
    result[pairs.astype(np.int64)] += deltas
    if np.any(result < -2048) or np.any(result > 2047):
        raise CompensationOverlayError("overlay pushes carrier outside signed-int12")
    return result


def selector_payload_bytes(payload: bytes) -> int:
    """Return the exact F0E1 selector prefix length before an optional overlay."""
    if len(payload) < _SELECTOR_HEADER.size:
        raise CompensationOverlayError("truncated frame-0 selector")
    magic, version, count = _SELECTOR_HEADER.unpack_from(payload)
    if magic != b"F0E1" or version != 1 or not 1 <= count <= PAIR_COUNT:
        raise CompensationOverlayError("invalid frame-0 selector header")
    limit = math.comb(PAIR_COUNT, count)
    rank_bytes = ((limit - 1).bit_length() + 7) // 8
    label_bytes = (count * 3 + 7) // 8
    expected = _SELECTOR_HEADER.size + rank_bytes + label_bytes
    if len(payload) < expected:
        raise CompensationOverlayError("truncated frame-0 selector payload")
    return expected


def split_selector_compensation(payload: bytes) -> tuple[bytes, bytes | None]:
    """Split one strict F0E1 selector from an optional strict QS2 overlay."""
    selector_bytes = selector_payload_bytes(payload)
    selector = payload[:selector_bytes]
    overlay = payload[selector_bytes:]
    if not overlay:
        return selector, None
    decode_compensation_overlay(overlay)
    return selector, overlay

