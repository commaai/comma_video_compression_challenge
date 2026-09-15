"""Decode portable byte-aligned rANS with fixed integer models."""

from __future__ import annotations

import numpy as np

PRECISION = 12
TOTAL = 1 << PRECISION
RANS_L = 1 << 23
STATE_BYTES = 4
ALPHABET = 16
_RESCALE_AT = TOTAL * 2


class ANSError(ValueError):
    """An ANS stream or model configuration is malformed."""


def _prior(index: int, alphabet: int) -> np.ndarray:
    """Return one schema-defined symmetric, zero-centred geometric prior."""
    if not 0 <= index <= 3 or alphabet < 2 or alphabet > TOTAL:
        raise ANSError("unsupported ANS prior or alphabet")
    centre = alphabet // 2
    distance = np.abs(np.arange(alphabet, dtype=np.int64) - centre)
    # The four catalog entries become progressively more concentrated at zero.
    weights = 1 + (4 + 2 * index) * np.maximum(0, centre - distance)
    return weights.astype(np.uint32)


def _normalise(counts: np.ndarray) -> np.ndarray:
    """Map positive integer counts to exactly ``TOTAL`` frequencies."""
    source = np.asarray(counts, dtype=np.uint64)
    if (
        source.ndim != 1
        or source.size < 2
        or source.size > TOTAL
        or np.any(source == 0)
    ):
        raise ANSError("invalid ANS count vector")
    total = int(source.sum())
    product = source * TOTAL
    frequencies = np.maximum(1, product // total).astype(np.int64)
    remainder = int(TOTAL - frequencies.sum())
    if remainder > 0:
        # Larger integer remainders win; symbol number breaks ties canonically.
        fractions = product % total
        for index in np.argsort(-fractions, kind="stable")[:remainder]:
            frequencies[index] += 1
    elif remainder < 0:
        # Removing only from frequencies above one keeps every symbol decodable.
        fractions = product % total
        candidates = [
            int(index)
            for index in np.argsort(fractions, kind="stable")
            if frequencies[index] > 1
        ]
        if len(candidates) < -remainder:
            raise ANSError("ANS normalization underflow")
        for index in candidates[:-remainder]:
            frequencies[index] -= 1
    if int(frequencies.sum()) != TOTAL or np.any(frequencies <= 0):
        raise ANSError("ANS normalization failed")
    return frequencies.astype(np.uint16)


def _update(counts: np.ndarray, symbol: int) -> None:
    counts[symbol] += 1
    if int(counts.sum(dtype=np.uint64)) >= _RESCALE_AT:
        counts[:] = np.maximum(1, (counts + 1) // 2)


def _decode_events(
    payload: bytes, count: int, model: np.ndarray, *, adaptive: bool
) -> np.ndarray:
    if count < 0 or len(payload) < STATE_BYTES:
        raise ANSError("truncated ANS state")
    state = int.from_bytes(payload[:STATE_BYTES], "little")
    if state < RANS_L:
        raise ANSError("invalid ANS initial state")
    cursor = STATE_BYTES
    output = np.empty(count, dtype=np.uint8)
    counts = np.asarray(model, dtype=np.uint32).copy()
    for index in range(count):
        frequencies = _normalise(counts)
        slot = state & (TOTAL - 1)
        cumulative = np.cumsum(frequencies, dtype=np.uint32)
        symbol = int(np.searchsorted(cumulative, slot, side="right"))
        if symbol >= frequencies.size:
            raise ANSError("ANS state selects no symbol")
        lower = 0 if symbol == 0 else int(cumulative[symbol - 1])
        state = int(frequencies[symbol]) * (state >> PRECISION) + slot - lower
        while state < RANS_L:
            if cursor >= len(payload):
                raise ANSError("truncated ANS renormalization bytes")
            state = (state << 8) | payload[cursor]
            cursor += 1
        output[index] = symbol
        if adaptive:
            _update(counts, symbol)
    if cursor != len(payload):
        raise ANSError("ANS trailing bytes")
    return output


def decode_adaptive(
    payload: bytes, count: int, *, prior_index: int = 0, alphabet: int = ALPHABET
) -> np.ndarray:
    """Strictly decode one adaptive order-0 stream, rejecting trailing bytes."""
    if count == 0:
        if payload:
            raise ANSError("ANS empty stream has payload")
        return np.empty(0, dtype=np.uint8)
    return _decode_events(payload, count, _prior(prior_index, alphabet), adaptive=True)
