"""Decode the canonical CPR1 carrier consumed by the renderer."""

from __future__ import annotations

import struct

import numpy as np

MAGIC = b"CPR1"
HEADER = struct.Struct("<4sII")
BASIS_BITS = 5
COEFFICIENT_BITS = 12
ALPHABET_SIZE = 1 << BASIS_BITS
MAX_HUFFMAN_LENGTH = 31


def _payload_size(bit_count: int) -> int:
    if bit_count <= 0:
        raise ValueError("bit count must be positive")
    return (bit_count + 7) // 8


def _validate_payload(
    payload: bytes | memoryview,
    bit_count: int,
    label: str,
) -> None:
    expected = _payload_size(bit_count)
    if len(payload) != expected:
        raise ValueError(f"{label} payload length mismatch")
    padding = expected * 8 - bit_count
    if padding and int(payload[-1]) & ((1 << padding) - 1):
        raise ValueError(f"{label} payload has non-zero padding")


def _canonical_codes(lengths: np.ndarray) -> dict[int, tuple[int, int]]:
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
    result = {}
    for length, symbol in entries:
        code <<= length - previous_length
        if code >= 1 << length:
            raise ValueError("oversubscribed Huffman code lengths")
        result[symbol] = (code, length)
        code += 1
        previous_length = length
    if code != 1 << previous_length:
        raise ValueError("incomplete Huffman code lengths")
    return result


def _decode_huffman(
    lengths: np.ndarray,
    payload: bytes | memoryview,
    bit_count: int,
    symbol_count: int,
) -> np.ndarray:
    if symbol_count <= 0:
        raise ValueError("Huffman symbol count must be positive")
    _validate_payload(payload, bit_count, "Huffman")
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


def _decode_rice(
    parameters: np.ndarray,
    payload: bytes | memoryview,
    bit_count: int,
    frames: int,
    dimensions: int,
) -> np.ndarray:
    if frames <= 0 or dimensions <= 0:
        raise ValueError("Rice output shape must be positive")
    parameters = np.asarray(parameters, dtype=np.int64).reshape(-1)
    if parameters.size != dimensions:
        raise ValueError("Rice table does not match coefficient dimensions")
    if np.any(parameters < 0) or np.any(parameters >= COEFFICIENT_BITS):
        raise ValueError("Rice parameter is outside the supported range")
    _validate_payload(payload, bit_count, "Rice")
    bits = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8),
        bitorder="big",
    )[:bit_count]
    cursor = 0
    result = np.empty((frames, dimensions), dtype=np.int32)
    for dimension, value in enumerate(parameters):
        parameter = int(value)
        maximum_quotient = ((1 << COEFFICIENT_BITS) - 1) >> parameter
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
            if cursor + parameter > bit_count:
                raise ValueError("truncated Rice remainder")
            remainder = 0
            for _ in range(parameter):
                remainder = (remainder << 1) | int(bits[cursor])
                cursor += 1
            result[frame, dimension] = (quotient << parameter) | remainder
    if cursor != bit_count:
        raise ValueError("Rice payload has surplus declared bits")
    return result


def decode_compact_carrier(
    blob: bytes | memoryview,
    basis_count: int,
    frames: int,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return basis scales/codes and coefficient scales/codes from CPR1."""
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
    magic, basis_bits, coefficient_bits = HEADER.unpack(blob[: HEADER.size])
    if magic != MAGIC:
        raise ValueError("unsupported compact carrier magic")
    basis_bytes = _payload_size(basis_bits)
    coefficient_bytes = _payload_size(coefficient_bits)
    if len(blob) != prefix_bytes + basis_bytes + coefficient_bytes:
        raise ValueError("compact carrier length mismatch")

    cursor = HEADER.size
    scale_bytes = dimensions * np.dtype("<f4").itemsize
    basis_scales = np.frombuffer(
        blob[cursor : cursor + scale_bytes], dtype="<f4"
    ).copy()
    cursor += scale_bytes
    coefficient_scales = np.frombuffer(
        blob[cursor : cursor + scale_bytes], dtype="<f4"
    ).copy()
    cursor += scale_bytes
    lengths = np.frombuffer(
        blob[cursor : cursor + ALPHABET_SIZE], dtype=np.uint8
    ).copy()
    cursor += ALPHABET_SIZE
    parameters = np.frombuffer(
        blob[cursor : cursor + dimensions], dtype=np.uint8
    ).copy()
    cursor += dimensions
    basis_payload = blob[cursor : cursor + basis_bytes]
    cursor += basis_bytes
    coefficient_payload = blob[cursor : cursor + coefficient_bytes]

    basis_unsigned = _decode_huffman(
        lengths, basis_payload, basis_bits, basis_count
    )
    basis_codes = (
        (basis_unsigned.astype(np.int64) >> 1)
        ^ -(basis_unsigned.astype(np.int64) & 1)
    ).astype(np.int32)
    coefficients = _decode_rice(
        parameters,
        coefficient_payload,
        coefficient_bits,
        frames,
        dimensions,
    )
    return basis_scales, basis_codes, coefficient_scales, coefficients
