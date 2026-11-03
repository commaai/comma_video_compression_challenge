"""DX2 lossless CABAC-prefix recode for the fixed CAP1 coefficient stream.

The live FX5 carrier has two independent entropy riders:

* RR5 (reserved bit ``0x08``) recodes the 27,648 five-bit basis symbols.
* DX2 (reserved bit ``0x10``) recodes the 7,200 Rice coefficient symbols.

DX2 models only each Rice symbol's unary quotient prefix.  The ``k``-bit
remainder is sent through equiprobable bypass bins.  All probabilities are
integer, start from the same fixed state, and update from already-decoded bins;
there is no transmitted model and no device-dependent arithmetic.

This file is both the encoder reference and the receiver implementation.  The
builder copies these exact bytes into the candidate runtime, then proves that
``restore_carrier_body(apply_cabac_to_carrier_body(body)) == body``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rr5_arith_basis import (
    RiderError,
    assemble_carrier_body,
    split_carrier_body,
    unpack_unsigned,
)

N_FRAMES = 600
CARRIER_DIM = 12
SYMBOL_COUNT = N_FRAMES * CARRIER_DIM
CABAC_CONTEXT_CAP = 8
DX2_RESERVED_CABAC_COEFFICIENTS = 0x10


class CabacCoefficientError(RiderError):
    """The CABAC coefficient rider is malformed or fails an identity control."""


class _RangeEncoder:
    """Carryless 32-bit range encoder used by the measured DX1 winner."""

    __slots__ = ("cache", "cache_size", "low", "out", "range")

    def __init__(self) -> None:
        self.low = 0
        self.range = 0xFFFFFFFF
        self.out = bytearray()
        self.cache = 0xFF
        self.cache_size = 0

    def _shift_low(self) -> None:
        if self.low < 0xFF000000 or self.low > 0xFFFFFFFF:
            carry = self.low >> 32
            if self.cache_size:
                self.out.append((self.cache + carry) & 0xFF)
            for _ in range(self.cache_size - 1):
                self.out.append((0xFF + carry) & 0xFF)
            self.cache = (self.low >> 24) & 0xFF
            self.cache_size = 0
        self.cache_size += 1
        self.low = (self.low << 8) & 0xFFFFFFFF

    def encode(self, cumulative_low: int, frequency: int, total: int) -> None:
        unit = self.range // total
        self.low += unit * cumulative_low
        self.range = unit * frequency
        while self.range < (1 << 24):
            self.range <<= 8
            self._shift_low()

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class _RangeDecoder:
    """Integer inverse of :class:`_RangeEncoder`; no float or device path exists."""

    __slots__ = ("buffer", "code", "position", "range")

    def __init__(self, payload: bytes) -> None:
        if len(payload) < 4:
            raise CabacCoefficientError("CABAC payload is shorter than its range prefix")
        self.buffer = payload
        self.position = 0
        self.range = 0xFFFFFFFF
        self.code = 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF

    def _byte(self) -> int:
        if self.position < len(self.buffer):
            value = self.buffer[self.position]
            self.position += 1
            return value
        # The measured Subbotin stream uses virtual zero padding after finish.
        self.position += 1
        return 0

    def decode_frequency(self, total: int) -> int:
        unit = self.range // total
        value = self.code // unit
        return total - 1 if value >= total else value

    def update(self, cumulative_low: int, frequency: int, total: int) -> None:
        unit = self.range // total
        self.code -= unit * cumulative_low
        self.range = unit * frequency
        while self.range < (1 << 24):
            self.range <<= 8
            self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF


@dataclass
class _BinaryModel:
    probability_zero: int = 2048


def _validate_ks(ks: np.ndarray) -> np.ndarray:
    values = np.asarray(ks, dtype=np.int64).reshape(-1)
    if values.shape != (CARRIER_DIM,) or np.any(values < 0) or np.any(values >= 12):
        raise CabacCoefficientError("CAP1 Rice parameters are outside the fixed domain")
    return values


def packed_ks(metadata: bytes) -> np.ndarray:
    """Read the 12 fixed Rice ``k`` values from the 40-byte packed metadata."""

    if len(metadata) != 40:
        raise CabacCoefficientError("packed CAP1 metadata must be exactly 40 bytes")
    base = metadata[37]
    deltas = unpack_unsigned(metadata[38:40], CARRIER_DIM, 1).astype(np.int64)
    return _validate_ks(base + deltas)


def _dimension_sequence() -> np.ndarray:
    # DX1 encoded the retained array in C order: frame outer, dimension inner.
    return np.tile(np.arange(CARRIER_DIM, dtype=np.int64), N_FRAMES)


def cabac_encode(symbols: np.ndarray, ks: np.ndarray) -> bytes:
    """Encode exactly 600x12 unsigned-int12 symbols with the DX1 cap=8 model."""

    values = np.asarray(symbols, dtype=np.int64)
    if values.shape != (N_FRAMES, CARRIER_DIM):
        raise CabacCoefficientError("CABAC coefficient array must have shape (600, 12)")
    if np.any(values < 0) or np.any(values >= 4096):
        raise CabacCoefficientError("CABAC coefficient exceeds unsigned int12")
    parameters = _validate_ks(ks)
    contexts = [
        [_BinaryModel() for _ in range(CABAC_CONTEXT_CAP + 1)]
        for _ in range(CARRIER_DIM)
    ]
    encoder = _RangeEncoder()
    for value, dimension in zip(
        values.reshape(-1).tolist(), _dimension_sequence().tolist(), strict=True
    ):
        k = int(parameters[dimension])
        quotient = value >> k
        models = contexts[dimension]
        for index in range(quotient):
            model = models[min(index, CABAC_CONTEXT_CAP)]
            encoder.encode(0, model.probability_zero, 4096)
            model.probability_zero += (4096 - model.probability_zero) >> 4
        model = models[min(quotient, CABAC_CONTEXT_CAP)]
        encoder.encode(model.probability_zero, 4096 - model.probability_zero, 4096)
        model.probability_zero -= model.probability_zero >> 4
        remainder = value & ((1 << k) - 1)
        for shift in range(k - 1, -1, -1):
            bit = (remainder >> shift) & 1
            encoder.encode(2048 * bit, 2048, 4096)
    return encoder.finish()


def cabac_decode(payload: bytes, ks: np.ndarray) -> np.ndarray:
    """Decode the fixed DX2 symbol count using integer-only adaptive contexts."""

    parameters = _validate_ks(ks)
    contexts = [
        [_BinaryModel() for _ in range(CABAC_CONTEXT_CAP + 1)]
        for _ in range(CARRIER_DIM)
    ]
    decoder = _RangeDecoder(bytes(payload))
    output = np.empty(SYMBOL_COUNT, dtype=np.int32)
    for index, dimension in enumerate(_dimension_sequence().tolist()):
        k = int(parameters[dimension])
        models = contexts[dimension]
        quotient = 0
        while True:
            model = models[min(quotient, CABAC_CONTEXT_CAP)]
            target = decoder.decode_frequency(4096)
            if target < model.probability_zero:
                decoder.update(0, model.probability_zero, 4096)
                model.probability_zero += (4096 - model.probability_zero) >> 4
                quotient += 1
                if quotient > 4096:
                    raise CabacCoefficientError("CABAC unary quotient overrun")
            else:
                decoder.update(
                    model.probability_zero,
                    4096 - model.probability_zero,
                    4096,
                )
                model.probability_zero -= model.probability_zero >> 4
                break
        value = quotient << k
        for shift in range(k - 1, -1, -1):
            target = decoder.decode_frequency(4096)
            bit = int(target >= 2048)
            decoder.update(2048 * bit, 2048, 4096)
            value |= bit << shift
        if value >= 4096:
            raise CabacCoefficientError("decoded CABAC coefficient exceeds unsigned int12")
        output[index] = value
    return output.reshape(N_FRAMES, CARRIER_DIM)


def decode_cabac_checked(payload: bytes, ks: np.ndarray) -> np.ndarray:
    """Decode and require canonical re-encoding to consume the exact payload bytes."""

    decoded = cabac_decode(payload, ks)
    if cabac_encode(decoded, ks) != bytes(payload):
        raise CabacCoefficientError("CABAC payload is non-canonical or corrupted")
    return decoded


def rice_decode(payload: bytes, bit_count: int, ks: np.ndarray) -> np.ndarray:
    """Decode CAP1's original dimension-major fixed-``k`` Rice stream."""

    parameters = _validate_ks(ks)
    if bit_count <= 0 or len(payload) != (bit_count + 7) // 8:
        raise CabacCoefficientError("Rice payload length disagrees with its bit count")
    if bit_count % 8 and payload[-1] & ((1 << (8 - bit_count % 8)) - 1):
        raise CabacCoefficientError("Rice payload has nonzero padding")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")[
        :bit_count
    ]
    output = np.empty((N_FRAMES, CARRIER_DIM), dtype=np.int32)
    cursor = 0
    for dimension in range(CARRIER_DIM):
        k = int(parameters[dimension])
        for frame in range(N_FRAMES):
            quotient = 0
            while True:
                if cursor >= bit_count:
                    raise CabacCoefficientError("truncated Rice unary code")
                bit = int(bits[cursor])
                cursor += 1
                if bit:
                    break
                quotient += 1
            if cursor + k > bit_count:
                raise CabacCoefficientError("truncated Rice remainder")
            remainder = 0
            for _ in range(k):
                remainder = (remainder << 1) | int(bits[cursor])
                cursor += 1
            value = (quotient << k) | remainder
            if value >= 4096:
                raise CabacCoefficientError("Rice coefficient exceeds unsigned int12")
            output[frame, dimension] = value
    if cursor != bit_count:
        raise CabacCoefficientError("Rice payload has surplus declared bits")
    return output


def rice_encode(symbols: np.ndarray, ks: np.ndarray) -> tuple[bytes, int]:
    """Rebuild CAP1's canonical dimension-major Rice payload with fixed ``k``."""

    values = np.asarray(symbols, dtype=np.int64)
    if values.shape != (N_FRAMES, CARRIER_DIM):
        raise CabacCoefficientError("Rice coefficient array must have shape (600, 12)")
    if np.any(values < 0) or np.any(values >= 4096):
        raise CabacCoefficientError("Rice coefficient exceeds unsigned int12")
    parameters = _validate_ks(ks)
    bits: list[int] = []
    for dimension in range(CARRIER_DIM):
        k = int(parameters[dimension])
        for value in values[:, dimension].tolist():
            bits.extend((0,) * (value >> k))
            bits.append(1)
            bits.extend((value >> shift) & 1 for shift in range(k - 1, -1, -1))
    payload = np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big").tobytes()
    return payload, len(bits)


def apply_cabac_to_carrier_body(body: bytes) -> dict[str, object]:
    """Replace only the packed CAP1 Rice bytes and prove exact inverse closure."""

    fields = split_carrier_body(body)
    parameters = packed_ks(bytes(fields["metadata"]))
    symbols = rice_decode(
        bytes(fields["rice"]), int(fields["residual_bits"]), parameters
    )
    cabac_payload = cabac_encode(symbols, parameters)
    if not np.array_equal(decode_cabac_checked(cabac_payload, parameters), symbols):
        raise CabacCoefficientError("CABAC round-trip differs from the Rice symbols")
    candidate_fields = dict(fields)
    candidate_fields["rice"] = cabac_payload
    candidate_fields["residual_bits"] = len(cabac_payload) * 8
    candidate_body = assemble_carrier_body(candidate_fields)
    if restore_carrier_body(candidate_body) != body:
        raise CabacCoefficientError("CABAC carrier inverse failed byte identity")
    return {
        "body": candidate_body,
        "symbols": symbols,
        "ks": parameters,
        "rice_payload": bytes(fields["rice"]),
        "rice_bits": int(fields["residual_bits"]),
        "cabac_payload": cabac_payload,
        "cabac_bits": len(cabac_payload) * 8,
    }


def restore_carrier_body(body: bytes) -> bytes:
    """Receiver side: restore the original Rice-coded carrier byte-for-byte."""

    fields = split_carrier_body(body)
    if int(fields["residual_bits"]) % 8:
        raise CabacCoefficientError("DX2 CABAC payload must declare whole bytes")
    parameters = packed_ks(bytes(fields["metadata"]))
    symbols = decode_cabac_checked(bytes(fields["rice"]), parameters)
    rice_payload, rice_bits = rice_encode(symbols, parameters)
    restored_fields = dict(fields)
    restored_fields["rice"] = rice_payload
    restored_fields["residual_bits"] = rice_bits
    return assemble_carrier_body(restored_fields)
