"""Canonical little-endian signed fixed-width unpacking."""

from __future__ import annotations


class BitPackingError(ValueError):
    """A packed-code stream is structurally invalid."""


def _bounds(bits: int) -> tuple[int, int]:
    if not 2 <= bits <= 8:
        raise ValueError("signed packing supports 2 through 8 bits")
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def packed_length(count: int, bits: int) -> int:
    if count < 0:
        raise ValueError("count must be non-negative")
    _bounds(bits)
    return (count * bits + 7) // 8


def unpack_signed(blob: bytes, count: int, bits: int) -> tuple[int, ...]:
    """Unpack an exact stream and reject surplus or non-zero padding bits."""
    expected = packed_length(count, bits)
    if len(blob) != expected:
        raise BitPackingError(f"expected {expected} bytes, received {len(blob)}")
    if not count:
        return ()
    total_bits = count * bits
    if total_bits % 8:
        padding_mask = ~((1 << (total_bits % 8)) - 1) & 0xFF
        if blob[-1] & padding_mask:
            raise BitPackingError("non-zero padding bits")
    unsigned_mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    result: list[int] = []
    acc = 0
    available = 0
    index = 0
    for _ in range(count):
        while available < bits:
            acc |= blob[index] << available
            index += 1
            available += 8
        value = acc & unsigned_mask
        acc >>= bits
        available -= bits
        result.append(value - (1 << bits) if value & sign else value)
    return tuple(result)
