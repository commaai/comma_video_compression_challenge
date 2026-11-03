"""Decode the CAP1 carrier and optional F26 frame-0 selector."""

from __future__ import annotations

import struct

import numpy as np

from .frame0_selector import decode_selector

CPR1_MAGIC = b"CPR1"
CAP1_MAGIC = b"CAP1"
FRAME0_SELECTOR_CARRIER_MAGIC = b"F0C1"


class CarrierRepackError(ValueError):
    """The F26 carrier representation is malformed or non-canonical."""


def _read_u24(raw: bytes) -> int:
    if len(raw) != 3:
        raise CarrierRepackError("truncated u24 carrier bit count")
    return int.from_bytes(raw, "little")


def pack_frame0_selector_carrier(carrier: bytes, selector: bytes) -> bytes:
    """Attach the charged sparse selector to its CAP1 carrier."""
    if not carrier or len(carrier) >= 1 << 16:
        raise CarrierRepackError("selector carrier base length must fit u16")
    try:
        decode_selector(selector)
    except ValueError as error:
        raise CarrierRepackError(f"invalid frame-0 selector: {error}") from error
    return (
        FRAME0_SELECTOR_CARRIER_MAGIC
        + struct.pack("<H", len(carrier))
        + carrier
        + selector
    )


def split_frame0_selector_carrier(blob: bytes) -> tuple[bytes, bytes | None]:
    """Return the carrier and its optional selector, rejecting trailing ambiguity."""
    if not blob.startswith(FRAME0_SELECTOR_CARRIER_MAGIC):
        return blob, None
    if len(blob) < 7:
        raise CarrierRepackError("truncated frame-0 selector carrier")
    carrier_bytes = struct.unpack_from("<H", blob, 4)[0]
    offset = 6
    if not carrier_bytes or offset + carrier_bytes >= len(blob):
        raise CarrierRepackError("invalid frame-0 selector carrier length")
    carrier = blob[offset : offset + carrier_bytes]
    selector = blob[offset + carrier_bytes :]
    try:
        decode_selector(selector)
    except ValueError as error:
        raise CarrierRepackError(f"invalid frame-0 selector: {error}") from error
    return carrier, selector


def materialize_cpr1(blob: bytes, runtime) -> bytes:
    """Restore the canonical CPR1 bytes accepted by the renderer."""
    carrier, _ = split_frame0_selector_carrier(blob)
    if carrier.startswith(CPR1_MAGIC):
        return carrier
    if carrier.startswith(CAP1_MAGIC):
        from .entropy.coefficient_ar1_codec import decode_cap1

        return decode_cap1(
            carrier,
            frames=runtime.N,
            dimensions=runtime.CARRIER_DIM,
        )
    raise CarrierRepackError("F26 requires a CPR1 or CAP1 carrier")


def _pack_bits(bits) -> tuple[bytes, int]:
    values = list(bits)
    if not values or any(bit not in (0, 1) for bit in values):
        raise CarrierRepackError("invalid carrier bitstream")
    payload = np.packbits(
        np.asarray(values, dtype=np.uint8), bitorder="big"
    ).tobytes()
    return payload, len(values)


def _zigzag(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if np.any(values < -2048) or np.any(values > 2047):
        raise CarrierRepackError("predictive residual exceeds signed int12")
    return ((values << 1) ^ (values >> 63)).astype(np.int32) & 0xFFF


def _unzigzag(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if np.any(values < 0) or np.any(values >= 4096):
        raise CarrierRepackError("Rice value exceeds unsigned int12")
    return ((values >> 1) ^ -(values & 1)).astype(np.int32)


def _rice_bits(values: np.ndarray, k: int) -> int:
    quotient_sum = int((np.asarray(values, dtype=np.uint64) >> k).sum())
    return quotient_sum + int(values.size) * (k + 1)


def _rice_encode(values: np.ndarray, segments: int) -> tuple[np.ndarray, bytes, int]:
    values = np.asarray(values, dtype=np.int64)
    if (
        values.ndim != 2
        or not values.size
        or np.any(values < 0)
        or np.any(values >= 4096)
        or segments not in (1, 2, 4, 8)
    ):
        raise CarrierRepackError("invalid predictive Rice input")
    frames, dimensions = values.shape
    if frames % segments:
        raise CarrierRepackError("Rice segment count must divide frame count")
    per_segment = frames // segments
    ks = np.empty((dimensions, segments), dtype=np.uint8)
    bits = []
    for dimension in range(dimensions):
        for segment in range(segments):
            column = values[
                segment * per_segment : (segment + 1) * per_segment,
                dimension,
            ]
            _, k = min(
                (_rice_bits(column, candidate), candidate)
                for candidate in range(12)
            )
            ks[dimension, segment] = k
            for item in column:
                item = int(item)
                bits.extend((0,) * (item >> k))
                bits.append(1)
                bits.extend(
                    (item >> shift) & 1 for shift in range(k - 1, -1, -1)
                )
    payload, bit_count = _pack_bits(bits)
    return ks, payload, bit_count


def _rice_decode(
    ks: np.ndarray,
    payload: bytes,
    bit_count: int,
    frames: int,
    dimensions: int,
) -> np.ndarray:
    ks = np.asarray(ks, dtype=np.int64)
    if (
        ks.ndim != 2
        or ks.shape[0] != dimensions
        or not ks.shape[1]
        or frames % ks.shape[1]
        or bit_count <= 0
        or len(payload) != (bit_count + 7) // 8
    ):
        raise CarrierRepackError("invalid predictive Rice payload")
    if (
        np.any(ks < 0)
        or np.any(ks >= 12)
        or (
            bit_count % 8
            and payload[-1] & ((1 << (8 - bit_count % 8)) - 1)
        )
    ):
        raise CarrierRepackError("invalid predictive Rice parameters or padding")
    bits = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8), bitorder="big"
    )[:bit_count]
    result = np.empty((frames, dimensions), dtype=np.int32)
    cursor = 0
    per_segment = frames // ks.shape[1]
    for dimension in range(dimensions):
        for segment in range(ks.shape[1]):
            k = int(ks[dimension, segment])
            first = segment * per_segment
            last = (segment + 1) * per_segment
            for frame in range(first, last):
                quotient = 0
                while True:
                    if cursor >= bit_count:
                        raise CarrierRepackError("truncated Rice unary code")
                    bit = int(bits[cursor])
                    cursor += 1
                    if bit:
                        break
                    quotient += 1
                    if quotient > 4095 >> k:
                        raise CarrierRepackError("Rice value exceeds int12")
                if cursor + k > bit_count:
                    raise CarrierRepackError("truncated Rice remainder")
                remainder = 0
                for _ in range(k):
                    remainder = (remainder << 1) | int(bits[cursor])
                    cursor += 1
                result[frame, dimension] = (quotient << k) | remainder
    if cursor != bit_count:
        raise CarrierRepackError("predictive Rice trailing bits")
    return result

def _signed_mod(values: np.ndarray) -> np.ndarray:
    return (
        ((np.asarray(values, dtype=np.int64) + 2048) & 0xFFF).astype(np.int32)
        - 2048
    )


def _predict_residuals(
    coefficients: np.ndarray, ids: np.ndarray, threshold: int
) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.int32)
    frames, dimensions = values.shape
    result = np.empty_like(values)
    for frame in range(frames):
        if frame == 0:
            prediction = np.zeros(dimensions, dtype=np.int32)
        elif frame == 1:
            prediction = values[frame - 1]
        else:
            previous_delta = _signed_mod(values[frame - 1] - values[frame - 2])
            second = _signed_mod(values[frame - 1] + previous_delta)
            if threshold:
                prediction = np.where(
                    np.abs(previous_delta) <= threshold,
                    second,
                    values[frame - 1],
                )
            else:
                prediction = np.where(ids.astype(bool), second, values[frame - 1])
        result[frame] = _signed_mod(values[frame] - prediction)
    return result
