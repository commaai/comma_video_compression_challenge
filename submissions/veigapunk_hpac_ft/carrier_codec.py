#!/usr/bin/env python3
"""Deterministic lossless coding for the semantic-pose carrier."""

from __future__ import annotations

import heapq
import struct
from collections.abc import Iterable

import numpy as np


MAGIC = b"CPR1"
HEADER = struct.Struct("<4sII")
BASIS_BITS = 5
COEFF_BITS = 12
ALPHABET_SIZE = 1 << BASIS_BITS
MAX_HUFFMAN_LENGTH = 31


def _bit_payload_size(bit_count: int) -> int:
    if bit_count <= 0:
        raise ValueError("bit count must be positive")
    return (bit_count + 7) // 8


def _validate_bit_payload(
    payload: bytes | memoryview,
    bit_count: int,
    label: str,
) -> None:
    expected = _bit_payload_size(bit_count)
    if len(payload) != expected:
        raise ValueError(
            f"{label} payload length mismatch: {len(payload)} != {expected}"
        )
    padding = expected * 8 - bit_count
    if padding and (int(payload[-1]) & ((1 << padding) - 1)):
        raise ValueError(f"{label} payload has nonzero padding bits")


def _pack_bits(bits: Iterable[int]) -> tuple[bytes, int]:
    array = np.fromiter(bits, dtype=np.uint8)
    if array.size == 0:
        raise ValueError("cannot pack an empty bitstream")
    if np.any(array > 1):
        raise ValueError("bitstream contains a value other than zero or one")
    return (
        np.packbits(array, bitorder="big").tobytes(),
        int(array.size),
    )


def _huffman_code_lengths(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    if flat.size == 0:
        raise ValueError("cannot Huffman-code an empty symbol stream")
    if flat.min() < 0 or flat.max() >= ALPHABET_SIZE:
        raise ValueError("Huffman symbol is outside the 5-bit alphabet")
    counts = np.bincount(flat, minlength=ALPHABET_SIZE)
    heap: list[tuple[int, int, int | tuple]] = []
    serial = 0
    for symbol, count in enumerate(counts):
        if count:
            heap.append((int(count), serial, symbol))
            serial += 1
    heapq.heapify(heap)
    if len(heap) == 1:
        used_symbol = int(heap[0][2])
        unused_symbol = 0 if used_symbol != 0 else 1
        lengths = np.zeros(ALPHABET_SIZE, dtype=np.uint8)
        lengths[used_symbol] = 1
        lengths[unused_symbol] = 1
        return lengths
    while len(heap) > 1:
        weight_a, _, node_a = heapq.heappop(heap)
        weight_b, _, node_b = heapq.heappop(heap)
        heapq.heappush(
            heap,
            (weight_a + weight_b, serial, (node_a, node_b)),
        )
        serial += 1
    lengths = np.zeros(ALPHABET_SIZE, dtype=np.uint8)

    def visit(node: int | tuple, depth: int) -> None:
        if isinstance(node, int):
            if depth <= 0 or depth > MAX_HUFFMAN_LENGTH:
                raise ValueError("unsupported Huffman code length")
            lengths[node] = depth
            return
        visit(node[0], depth + 1)
        visit(node[1], depth + 1)

    visit(heap[0][2], 0)
    return lengths


def _canonical_codes(
    lengths: np.ndarray,
) -> dict[int, tuple[int, int]]:
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if lengths.size != ALPHABET_SIZE:
        raise ValueError("Huffman table must contain 32 code lengths")
    if np.any(lengths < 0) or np.any(lengths > MAX_HUFFMAN_LENGTH):
        raise ValueError("Huffman code length is outside the supported range")
    entries = sorted(
        (int(length), symbol)
        for symbol, length in enumerate(lengths)
        if length
    )
    if len(entries) < 2:
        raise ValueError("Huffman table must contain at least two symbols")
    code = 0
    previous_length = 0
    result: dict[int, tuple[int, int]] = {}
    for length, symbol in entries:
        code <<= length - previous_length
        if code >= (1 << length):
            raise ValueError("oversubscribed Huffman code lengths")
        result[symbol] = (code, length)
        code += 1
        previous_length = length
    if code != (1 << previous_length):
        raise ValueError("incomplete Huffman code lengths")
    return result


def _encode_huffman(values: np.ndarray) -> tuple[np.ndarray, bytes, int]:
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    lengths = _huffman_code_lengths(flat)
    codes = _canonical_codes(lengths)

    def bits() -> Iterable[int]:
        for value in flat:
            code, length = codes[int(value)]
            for shift in range(length - 1, -1, -1):
                yield (code >> shift) & 1

    payload, bit_count = _pack_bits(bits())
    return lengths, payload, bit_count


def _decode_huffman(
    lengths: np.ndarray,
    payload: bytes | memoryview,
    bit_count: int,
    symbol_count: int,
) -> np.ndarray:
    if symbol_count <= 0:
        raise ValueError("Huffman symbol count must be positive")
    _validate_bit_payload(payload, bit_count, "Huffman")
    codes = _canonical_codes(lengths)
    lookup = {
        (length, code): symbol
        for symbol, (code, length) in codes.items()
    }
    bits = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8),
        bitorder="big",
    )[:bit_count]
    result = np.empty(symbol_count, dtype=np.int32)
    current = 0
    length = 0
    output_index = 0
    for bit_index, bit in enumerate(bits):
        current = (current << 1) | int(bit)
        length += 1
        if length > MAX_HUFFMAN_LENGTH:
            raise ValueError("Huffman payload contains an invalid prefix")
        symbol = lookup.get((length, current))
        if symbol is None:
            continue
        result[output_index] = symbol
        output_index += 1
        current = 0
        length = 0
        if output_index == symbol_count:
            if bit_index + 1 != bit_count:
                raise ValueError("Huffman payload has surplus declared bits")
            return result
    raise ValueError("truncated Huffman payload")


def _zigzag_signed(values: np.ndarray, bits: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    if values.size == 0 or values.min() < lower or values.max() > upper:
        raise ValueError(f"signed values do not fit in {bits} bits")
    return (
        ((values << 1) ^ (values >> 63)) & ((1 << bits) - 1)
    ).astype(np.int32)


def _unzigzag_unsigned(values: np.ndarray, bits: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0 or values.min() < 0 or values.max() >= (1 << bits):
        raise ValueError(f"unsigned values do not fit in {bits} bits")
    decoded = (values >> 1) ^ -(values & 1)
    return decoded.astype(np.int32)


def _rice_bit_count(values: np.ndarray, k: int) -> int:
    values = np.asarray(values, dtype=np.uint64).reshape(-1)
    return int((values >> k).sum() + values.size * (k + 1))


def _encode_rice(
    encoded_coefficients: np.ndarray,
) -> tuple[np.ndarray, bytes, int]:
    values = np.asarray(encoded_coefficients, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("coefficient codes must have shape [frames, dimensions]")
    if values.size == 0 or values.min() < 0 or values.max() >= (1 << COEFF_BITS):
        raise ValueError("coefficient code is outside the 12-bit range")
    ks: list[int] = []

    def bits() -> Iterable[int]:
        for dimension in range(values.shape[1]):
            column = values[:, dimension]
            _, k = min(
                (_rice_bit_count(column, candidate), candidate)
                for candidate in range(COEFF_BITS)
            )
            ks.append(k)
            for item_value in column:
                item = int(item_value)
                quotient = item >> k
                yield from (0 for _ in range(quotient))
                yield 1
                for shift in range(k - 1, -1, -1):
                    yield (item >> shift) & 1

    payload, bit_count = _pack_bits(bits())
    return np.asarray(ks, dtype=np.uint8), payload, bit_count


def _decode_rice(
    ks: np.ndarray,
    payload: bytes | memoryview,
    bit_count: int,
    frames: int,
    dimensions: int,
) -> np.ndarray:
    if frames <= 0 or dimensions <= 0:
        raise ValueError("Rice output shape must be positive")
    ks = np.asarray(ks, dtype=np.int64).reshape(-1)
    if ks.size != dimensions:
        raise ValueError("Rice table does not match coefficient dimensions")
    if np.any(ks < 0) or np.any(ks >= COEFF_BITS):
        raise ValueError("Rice parameter is outside the supported range")
    _validate_bit_payload(payload, bit_count, "Rice")
    bits = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8),
        bitorder="big",
    )[:bit_count]
    cursor = 0
    result = np.empty((frames, dimensions), dtype=np.int32)
    for dimension, k_value in enumerate(ks):
        k = int(k_value)
        maximum_quotient = ((1 << COEFF_BITS) - 1) >> k
        for frame in range(frames):
            quotient = 0
            while True:
                if cursor >= bit_count:
                    raise ValueError("truncated Rice unary code")
                bit = int(bits[cursor])
                cursor += 1
                if bit:
                    break
                quotient += 1
                if quotient > maximum_quotient:
                    raise ValueError("Rice coefficient exceeds 12-bit range")
            if cursor + k > bit_count:
                raise ValueError("truncated Rice remainder")
            remainder = 0
            for _ in range(k):
                remainder = (remainder << 1) | int(bits[cursor])
                cursor += 1
            value = (quotient << k) | remainder
            if value >= (1 << COEFF_BITS):
                raise ValueError("Rice coefficient exceeds 12-bit range")
            result[frame, dimension] = value
    if cursor != bit_count:
        raise ValueError("Rice payload has surplus declared bits")
    return result


def encode_compact_carrier(
    basis_scales: np.ndarray,
    basis_codes: np.ndarray,
    coefficient_scales: np.ndarray,
    encoded_coefficients: np.ndarray,
) -> bytes:
    """Encode exact 5-bit basis codes and 12-bit delta/zigzag coefficients."""
    basis_codes = np.asarray(basis_codes, dtype=np.int64).reshape(-1)
    encoded_coefficients = np.asarray(encoded_coefficients, dtype=np.int64)
    if encoded_coefficients.ndim != 2:
        raise ValueError("coefficient codes must have shape [frames, dimensions]")
    dimensions = encoded_coefficients.shape[1]
    basis_scales = np.asarray(basis_scales, dtype="<f4").reshape(-1)
    coefficient_scales = np.asarray(
        coefficient_scales, dtype="<f4"
    ).reshape(-1)
    if basis_scales.size != dimensions or coefficient_scales.size != dimensions:
        raise ValueError("carrier scale count does not match coefficient dimensions")

    basis_unsigned = _zigzag_signed(basis_codes, BASIS_BITS)
    lengths, basis_payload, basis_bit_count = _encode_huffman(basis_unsigned)
    ks, coefficient_payload, coefficient_bit_count = _encode_rice(
        encoded_coefficients
    )
    return b"".join([
        HEADER.pack(MAGIC, basis_bit_count, coefficient_bit_count),
        basis_scales.tobytes(),
        coefficient_scales.tobytes(),
        lengths.tobytes(),
        ks.tobytes(),
        basis_payload,
        coefficient_payload,
    ])


def decode_compact_carrier(
    blob: bytes | memoryview,
    basis_count: int,
    frames: int,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode CPR1 and return scales, signed basis, scales, encoded coefficients."""
    if basis_count <= 0 or frames <= 0 or dimensions <= 0:
        raise ValueError("carrier dimensions must be positive")
    blob = memoryview(blob)
    prefix_bytes = (
        HEADER.size
        + 2 * dimensions * np.dtype("<f4").itemsize
        + ALPHABET_SIZE
        + dimensions
    )
    if len(blob) < prefix_bytes:
        raise ValueError("truncated compact carrier header")
    magic, basis_bit_count, coefficient_bit_count = HEADER.unpack(
        blob[:HEADER.size]
    )
    if magic != MAGIC:
        raise ValueError("unsupported compact carrier magic")
    basis_payload_bytes = _bit_payload_size(basis_bit_count)
    coefficient_payload_bytes = _bit_payload_size(coefficient_bit_count)
    expected_bytes = prefix_bytes + basis_payload_bytes + coefficient_payload_bytes
    if len(blob) != expected_bytes:
        raise ValueError(
            f"compact carrier length mismatch: {len(blob)} != {expected_bytes}"
        )

    cursor = HEADER.size
    scale_bytes = dimensions * np.dtype("<f4").itemsize
    basis_scales = np.frombuffer(
        blob[cursor:cursor + scale_bytes],
        dtype="<f4",
    ).copy()
    cursor += scale_bytes
    coefficient_scales = np.frombuffer(
        blob[cursor:cursor + scale_bytes],
        dtype="<f4",
    ).copy()
    cursor += scale_bytes
    lengths = np.frombuffer(
        blob[cursor:cursor + ALPHABET_SIZE],
        dtype=np.uint8,
    ).copy()
    cursor += ALPHABET_SIZE
    ks = np.frombuffer(
        blob[cursor:cursor + dimensions],
        dtype=np.uint8,
    ).copy()
    cursor += dimensions
    basis_payload = blob[cursor:cursor + basis_payload_bytes]
    cursor += basis_payload_bytes
    coefficient_payload = blob[cursor:cursor + coefficient_payload_bytes]

    basis_unsigned = _decode_huffman(
        lengths,
        basis_payload,
        basis_bit_count,
        basis_count,
    )
    basis_codes = _unzigzag_unsigned(basis_unsigned, BASIS_BITS)
    encoded_coefficients = _decode_rice(
        ks,
        coefficient_payload,
        coefficient_bit_count,
        frames,
        dimensions,
    )
    return (
        basis_scales,
        basis_codes,
        coefficient_scales,
        encoded_coefficients,
    )
