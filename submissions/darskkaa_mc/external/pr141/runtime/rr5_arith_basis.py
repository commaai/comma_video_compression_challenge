"""RR5 rider: the CPR1 basis inner-coder substitution (lossless, byte-only).

The shipped F26 carrier codes its 27,648 five-bit basis symbols with a STATIC
order-0 canonical Huffman code sharing ONE table across all 12 basis atoms.
``ddm_ra2`` MEASURED that an adaptive arithmetic coder contexted on the atom
index beats it, and that the coefficient stream must be left alone (Rice wins by
415 B).  This module is the production form of that measured result.

Everything here is a pure byte transform over an EXISTING carrier.  It changes
no decoded value, so ``d_seg`` and ``d_pose`` are unchanged *by construction*
rather than by measurement, and no scorer needs to run.

SINGLE SOURCE OF TRUTH.  ``tools/ddm_rr5_rider_apply.py`` encodes with this
module and copies this exact file (sha256 recorded in the receipt) into the
rider runtime tree, so the decoder runs byte-identical code.  The adaptive model
is driven identically on both sides, so no probability table is transmitted.

Provenance
----------
* arithmetic coder + ``adaptive_ctx_dim`` model: ported verbatim from
  ``experiments/ddm_ra2_cpr1_entropy_headroom.py`` (ra2, 2026-08-16), which
  MEASURED the win round-trip-exact on 27,648/27,648 symbols.
* packed-CAP1 byte map: DERIVED from the receiver's own reader,
  ``runtime/residual_archive.py::_restore_packed_cap1_metadata``.
"""

from __future__ import annotations

import heapq

import numpy as np

# --- carrier geometry (DERIVED from cpr1/inflate.py module constants) ------- #
CARRIER_DIM = 12
CARRIER_H, CARRIER_W = 24, 32
BASIS_PLANES = 3
BASIS_SYMBOLS = CARRIER_DIM * BASIS_PLANES * CARRIER_H * CARRIER_W  # 27,648
BASIS_ALPHABET = 32  # 5-bit zigzag; carrier_codec.ALPHABET_SIZE
MAX_HUFFMAN_LENGTH = 15  # the packed table stores 4 bits per symbol

# --- packed CAP1 byte map (DERIVED from residual_archive._restore_packed_*) - #
BIT_COUNT_BYTES = 6  # two little-endian u24 counts: basis_bits, residual_bits
SCALES_BYTES = 8 * CARRIER_DIM  # 96
PACKED_METADATA_OFFSET = BIT_COUNT_BYTES + SCALES_BYTES  # 102
PACKED_METADATA_BYTES = 40  # [102:142]; expands to 80 on restore
PACKED_LENGTHS_SPAN = (123, 139)  # 32 symbols x 4 bits = 16 B
BASIS_OFFSET = PACKED_METADATA_OFFSET + PACKED_METADATA_BYTES  # 142

# --- reserved-bit extension (follows the CK2/SZ1 precedent) ----------------- #
RR5_RESERVED_ARITH_BASIS = 0x08
RR5_RESERVED_KNOWN_BITS = 0x0F  # 0x07 (SZ1/CK2) | 0x08 (this rider)

# --- WNC arithmetic coder constants (verbatim from ra2) --------------------- #
_CODE_BITS = 32
_TOP = (1 << _CODE_BITS) - 1
_QTR = 1 << (_CODE_BITS - 2)
_HALF = 2 * _QTR
_3QTR = 3 * _QTR
_MAX_TOTAL = _QTR - 1


class RiderError(ValueError):
    """The carrier is malformed, or a fail-closed rider control did not hold."""


# --------------------------------------------------------------------------- #
# bit plumbing
# --------------------------------------------------------------------------- #
class _BitWriter:
    def __init__(self) -> None:
        self._bits: list[int] = []

    def put(self, bit: int) -> None:
        self._bits.append(bit & 1)

    def bytes(self) -> bytes:
        if not self._bits:
            return b""
        return np.packbits(
            np.asarray(self._bits, dtype=np.uint8), bitorder="big"
        ).tobytes()

    def bit_count(self) -> int:
        return len(self._bits)


class _BitReader:
    def __init__(self, payload: bytes, bit_count: int) -> None:
        self._bits = np.unpackbits(
            np.frombuffer(payload, dtype=np.uint8), bitorder="big"
        )[:bit_count]
        self._cursor = 0

    def get(self) -> int:
        if self._cursor >= self._bits.size:
            return 0  # virtual zero padding past the end, per WNC
        bit = int(self._bits[self._cursor])
        self._cursor += 1
        return bit


class _AdaptiveModel:
    """Per-context adaptive frequencies; encoder and decoder drive identical
    instances, so no table is transmitted."""

    def __init__(self, alphabet: int, n_contexts: int, increment: int = 32) -> None:
        self.alphabet = alphabet
        self.increment = increment
        self._freq = np.ones((n_contexts, alphabet), dtype=np.int64)
        self._total = np.full(n_contexts, alphabet, dtype=np.int64)

    def total(self, context: int) -> int:
        return int(self._total[context])

    def cum(self, context: int, symbol: int) -> tuple[int, int, int]:
        row = self._freq[context]
        low = int(row[:symbol].sum())
        return low, low + int(row[symbol]), int(self._total[context])

    def find(self, context: int, target: int) -> tuple[int, int, int]:
        row = self._freq[context]
        cumulative = np.cumsum(row)
        symbol = int(np.searchsorted(cumulative, target, side="right"))
        low = int(cumulative[symbol - 1]) if symbol else 0
        return symbol, low, int(cumulative[symbol])

    def update(self, context: int, symbol: int) -> None:
        self._freq[context, symbol] += self.increment
        self._total[context] += self.increment
        if self._total[context] >= _MAX_TOTAL:
            row = self._freq[context]
            np.maximum(row >> 1, 1, out=row)
            self._total[context] = int(row.sum())


def basis_contexts() -> np.ndarray:
    """Context = which of the 12 basis atoms (ra2's measured best model)."""
    return np.repeat(np.arange(CARRIER_DIM), BASIS_PLANES * CARRIER_H * CARRIER_W)


def arith_encode(
    symbols: np.ndarray, contexts: np.ndarray, model: _AdaptiveModel
) -> tuple[bytes, int]:
    writer = _BitWriter()
    low, high, pending = 0, _TOP, 0

    def emit(bit: int) -> None:
        nonlocal pending
        writer.put(bit)
        for _ in range(pending):
            writer.put(1 - bit)
        pending = 0

    for symbol, context in zip(symbols.tolist(), contexts.tolist(), strict=True):
        cum_low, cum_high, total = model.cum(context, symbol)
        span = high - low + 1
        high = low + (span * cum_high) // total - 1
        low = low + (span * cum_low) // total
        while True:
            if high < _HALF:
                emit(0)
            elif low >= _HALF:
                emit(1)
                low -= _HALF
                high -= _HALF
            elif low >= _QTR and high < _3QTR:
                pending += 1
                low -= _QTR
                high -= _QTR
            else:
                break
            low = (low << 1) & _TOP
            high = ((high << 1) | 1) & _TOP
        model.update(context, symbol)

    pending += 1
    emit(0 if low < _QTR else 1)
    return writer.bytes(), writer.bit_count()


def arith_decode(
    payload: bytes, bit_count: int, contexts: np.ndarray, model: _AdaptiveModel
) -> np.ndarray:
    reader = _BitReader(payload, bit_count)
    low, high = 0, _TOP
    value = 0
    for _ in range(_CODE_BITS):
        value = (value << 1) | reader.get()
    out = np.empty(contexts.size, dtype=np.int64)

    for index, context in enumerate(contexts.tolist()):
        total = model.total(context)
        span = high - low + 1
        target = ((value - low + 1) * total - 1) // span
        symbol, cum_low, cum_high = model.find(context, target)
        high = low + (span * cum_high) // total - 1
        low = low + (span * cum_low) // total
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                value -= _HALF
            elif low >= _QTR and high < _3QTR:
                low -= _QTR
                high -= _QTR
                value -= _QTR
            else:
                break
            low = (low << 1) & _TOP
            high = ((high << 1) | 1) & _TOP
            value = ((value << 1) | reader.get()) & _TOP
        out[index] = symbol
        model.update(context, symbol)
    return out


def encode_basis_arith(symbols: np.ndarray) -> tuple[bytes, int]:
    """Arithmetic-code the basis symbols; returns (payload, bit_count)."""
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    if symbols.size != BASIS_SYMBOLS:
        raise RiderError("basis symbol count is not 27,648")
    if symbols.min() < 0 or symbols.max() >= BASIS_ALPHABET:
        raise RiderError("basis symbol outside the 5-bit alphabet")
    model = _AdaptiveModel(BASIS_ALPHABET, CARRIER_DIM)
    return arith_encode(symbols, basis_contexts(), model)


def decode_basis_arith(payload: bytes, bit_count: int) -> np.ndarray:
    """Inverse of :func:`encode_basis_arith`."""
    model = _AdaptiveModel(BASIS_ALPHABET, CARRIER_DIM)
    return arith_decode(payload, bit_count, basis_contexts(), model)


# --------------------------------------------------------------------------- #
# canonical Huffman (the incumbent code the rider replaces)
# --------------------------------------------------------------------------- #
def huffman_lengths_from_histogram(histogram: np.ndarray) -> np.ndarray:
    """Rebuild the code-length vector the shipped table carries.

    The rider drops the transmitted table, so the decoder regenerates it from the
    decoded symbols.  The encoder VERIFIES this reproduces the shipped table
    exactly and refuses the drop otherwise, so the reconstruction is never
    trusted -- it is checked.
    """
    histogram = np.asarray(histogram, dtype=np.int64).reshape(-1)
    if histogram.size != BASIS_ALPHABET:
        raise RiderError("histogram must span the 32-symbol alphabet")
    items = [(int(f), i) for i, f in enumerate(histogram) if f > 0]
    if len(items) < 2:
        raise RiderError("Huffman alphabet must contain at least two symbols")
    heap = [(f, s, [s]) for f, s in items]
    heapq.heapify(heap)
    lengths = np.zeros(BASIS_ALPHABET, dtype=np.uint8)
    while len(heap) > 1:
        f1, _, s1 = heapq.heappop(heap)
        f2, _, s2 = heapq.heappop(heap)
        merged = s1 + s2
        for symbol in merged:
            lengths[symbol] += 1
        heapq.heappush(heap, (f1 + f2, min(merged), merged))
    if int(lengths.max()) > MAX_HUFFMAN_LENGTH:
        raise RiderError("rebuilt Huffman code exceeds the packed 4-bit field")
    return lengths


def canonical_codes(lengths: np.ndarray) -> dict[int, tuple[int, int]]:
    """Canonical (code, length) per symbol -- mirrors carrier_codec._canonical_codes."""
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if lengths.size != BASIS_ALPHABET:
        raise RiderError("Huffman table must contain 32 code lengths")
    entries = sorted(
        (int(length), symbol) for symbol, length in enumerate(lengths) if length
    )
    if len(entries) < 2:
        raise RiderError("Huffman table must contain at least two symbols")
    code = 0
    previous_length = 0
    result: dict[int, tuple[int, int]] = {}
    for length, symbol in entries:
        code <<= length - previous_length
        if code >= 1 << length:
            raise RiderError("oversubscribed Huffman code lengths")
        result[symbol] = (code, length)
        code += 1
        previous_length = length
    if code != 1 << previous_length:
        raise RiderError("incomplete Huffman code lengths")
    return result


def huffman_encode(symbols: np.ndarray, lengths: np.ndarray) -> tuple[bytes, int]:
    """Encode under the canonical code implied by ``lengths``."""
    codes = canonical_codes(lengths)
    bits: list[int] = []
    for symbol in np.asarray(symbols, dtype=np.int64).reshape(-1).tolist():
        entry = codes.get(int(symbol))
        if entry is None:
            raise RiderError(f"symbol {symbol} has no code in the table")
        code, length = entry
        bits.extend((code >> shift) & 1 for shift in range(length - 1, -1, -1))
    if not bits:
        raise RiderError("empty Huffman bitstream")
    payload = np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big").tobytes()
    return payload, len(bits)


def huffman_decode(
    lengths: np.ndarray, payload: bytes, bit_count: int, symbol_count: int
) -> np.ndarray:
    """Decode the shipped static order-0 stream (mirrors carrier_codec)."""
    codes = canonical_codes(lengths)
    lookup = {(length, code): symbol for symbol, (code, length) in codes.items()}
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")[
        :bit_count
    ]
    result = np.empty(symbol_count, dtype=np.int64)
    current = 0
    length = 0
    index = 0
    for position, bit in enumerate(bits.tolist()):
        current = (current << 1) | bit
        length += 1
        if length > 31:
            raise RiderError("Huffman payload contains an invalid prefix")
        symbol = lookup.get((length, current))
        if symbol is None:
            continue
        result[index] = symbol
        index += 1
        current = 0
        length = 0
        if index == symbol_count:
            if position + 1 != bit_count:
                raise RiderError("Huffman payload has surplus declared bits")
            return result
    raise RiderError("truncated Huffman payload")


# --------------------------------------------------------------------------- #
# packed-CAP1 bitfield plumbing (inverse of residual_archive._unpack_unsigned)
# --------------------------------------------------------------------------- #
def unpack_unsigned(raw: bytes, count: int, bits: int) -> np.ndarray:
    if len(raw) != (count * bits + 7) // 8:
        raise RiderError("packed CAP1 field has the wrong length")
    output = np.empty(count, dtype=np.int64)
    for index in range(count):
        offset = index * bits
        byte, shift = divmod(offset, 8)
        word = raw[byte]
        if byte + 1 < len(raw):
            word |= raw[byte + 1] << 8
        output[index] = (word >> shift) & ((1 << bits) - 1)
    return output


def pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if np.any(values < 0) or np.any(values >= (1 << bits)):
        raise RiderError(f"value does not fit the packed {bits}-bit field")
    total = values.size * bits
    buffer = bytearray((total + 7) // 8)
    for index, value in enumerate(values.tolist()):
        offset = index * bits
        byte, shift = divmod(offset, 8)
        word = int(value) << shift
        buffer[byte] |= word & 0xFF
        if byte + 1 < len(buffer):
            buffer[byte + 1] |= (word >> 8) & 0xFF
    return bytes(buffer)


def split_carrier_body(body: bytes) -> dict[str, object]:
    """Split a decompressed (CK2-restored) carrier body into its exact fields."""
    if len(body) < BASIS_OFFSET:
        raise RiderError("carrier body is shorter than the packed CAP1 prefix")
    basis_bits = int.from_bytes(body[0:3], "little")
    residual_bits = int.from_bytes(body[3:6], "little")
    if not basis_bits or not residual_bits:
        raise RiderError("carrier bit counts must be nonzero")
    basis_bytes = (basis_bits + 7) // 8
    rice_bytes = (residual_bits + 7) // 8
    packed_portion = BASIS_OFFSET + basis_bytes + rice_bytes
    if len(body) < packed_portion:
        raise RiderError("carrier body is truncated against its own bit counts")
    return {
        "basis_bits": basis_bits,
        "residual_bits": residual_bits,
        "scales": body[BIT_COUNT_BYTES:PACKED_METADATA_OFFSET],
        "metadata": bytearray(body[PACKED_METADATA_OFFSET:BASIS_OFFSET]),
        "basis": body[BASIS_OFFSET : BASIS_OFFSET + basis_bytes],
        "rice": body[BASIS_OFFSET + basis_bytes : packed_portion],
        "body_tail": body[packed_portion:],
    }


def packed_lengths(metadata: bytes) -> np.ndarray:
    lo, hi = PACKED_LENGTHS_SPAN
    start, end = lo - PACKED_METADATA_OFFSET, hi - PACKED_METADATA_OFFSET
    return unpack_unsigned(bytes(metadata[start:end]), BASIS_ALPHABET, 4).astype(
        np.uint8
    )


def with_packed_lengths(metadata: bytes, lengths: np.ndarray) -> bytearray:
    lo, hi = PACKED_LENGTHS_SPAN
    start, end = lo - PACKED_METADATA_OFFSET, hi - PACKED_METADATA_OFFSET
    out = bytearray(metadata)
    out[start:end] = pack_unsigned(np.asarray(lengths, dtype=np.int64), 4)
    return out


def assemble_carrier_body(fields: dict[str, object]) -> bytes:
    """Inverse of :func:`split_carrier_body`."""
    basis = bytes(fields["basis"])  # type: ignore[index]
    rice = bytes(fields["rice"])  # type: ignore[index]
    basis_bits = int(fields["basis_bits"])  # type: ignore[index]
    residual_bits = int(fields["residual_bits"])  # type: ignore[index]
    if len(basis) != (basis_bits + 7) // 8:
        raise RiderError("basis payload length disagrees with its bit count")
    if len(rice) != (residual_bits + 7) // 8:
        raise RiderError("rice payload length disagrees with its bit count")
    metadata = bytes(fields["metadata"])  # type: ignore[index]
    if len(metadata) != PACKED_METADATA_BYTES:
        raise RiderError("packed metadata must stay 40 bytes")
    scales = bytes(fields["scales"])  # type: ignore[index]
    if len(scales) != SCALES_BYTES:
        raise RiderError("scales block must stay 96 bytes")
    if basis_bits >= 1 << 24 or residual_bits >= 1 << 24:
        raise RiderError("bit count does not fit the container's u24 field")
    return b"".join(
        (
            basis_bits.to_bytes(3, "little"),
            residual_bits.to_bytes(3, "little"),
            scales,
            metadata,
            basis,
            rice,
            bytes(fields["body_tail"]),  # type: ignore[index]
        )
    )


def apply_rider_to_carrier_body(body: bytes) -> dict[str, object]:
    """Swap the basis inner coder Huffman -> adaptive arithmetic, losslessly.

    Every step carries a fail-closed control: the arithmetic stream must decode
    back to the exact input symbols, and the dropped Huffman table must rebuild
    to the exact shipped table AND re-encode to the exact shipped basis bytes.
    If the table control fails the table is KEPT (the rider still applies, it
    just claims fewer bytes) rather than silently shipping a lossy body.
    """
    fields = split_carrier_body(body)
    lengths = packed_lengths(bytes(fields["metadata"]))  # type: ignore[arg-type]
    symbols = huffman_decode(
        lengths,
        bytes(fields["basis"]),  # type: ignore[arg-type]
        int(fields["basis_bits"]),  # type: ignore[arg-type]
        BASIS_SYMBOLS,
    )

    payload, bit_count = encode_basis_arith(symbols)
    if not np.array_equal(decode_basis_arith(payload, bit_count), symbols):
        raise RiderError("arithmetic round-trip control FAILED; refusing to write")

    histogram = np.bincount(symbols, minlength=BASIS_ALPHABET)
    rebuilt = huffman_lengths_from_histogram(histogram)
    table_dropped = bool(np.array_equal(rebuilt, lengths))
    if table_dropped:
        replay, replay_bits = huffman_encode(symbols, rebuilt)
        if replay != bytes(fields["basis"]) or replay_bits != int(  # type: ignore[arg-type]
            fields["basis_bits"]
        ):
            table_dropped = False

    metadata = fields["metadata"]
    if table_dropped:
        metadata = with_packed_lengths(
            bytes(metadata), np.zeros(BASIS_ALPHABET, dtype=np.int64)  # type: ignore[arg-type]
        )

    rider = dict(fields)
    rider["metadata"] = metadata
    rider["basis"] = payload
    rider["basis_bits"] = bit_count
    return {
        "body": assemble_carrier_body(rider),
        "symbols": symbols,
        "shipped_basis_bytes": len(bytes(fields["basis"])),  # type: ignore[arg-type]
        "rider_basis_bytes": len(payload),
        "shipped_basis_bits": int(fields["basis_bits"]),  # type: ignore[arg-type]
        "rider_basis_bits": bit_count,
        "table_dropped": table_dropped,
        "shipped_lengths": lengths.tolist(),
    }


def restore_carrier_body(body: bytes) -> bytes:
    """Decoder side: turn a rider carrier body back into the shipped one.

    Byte-identical to the input of :func:`apply_rider_to_carrier_body`, so every
    stage downstream of it is bit-identical by construction.
    """
    fields = split_carrier_body(body)
    symbols = decode_basis_arith(
        bytes(fields["basis"]),  # type: ignore[arg-type]
        int(fields["basis_bits"]),  # type: ignore[arg-type]
    )
    lengths = packed_lengths(bytes(fields["metadata"]))  # type: ignore[arg-type]
    if not int(np.asarray(lengths, dtype=np.int64).sum()):
        lengths = huffman_lengths_from_histogram(
            np.bincount(symbols, minlength=BASIS_ALPHABET)
        )
    payload, bit_count = huffman_encode(symbols, lengths)
    restored = dict(fields)
    restored["metadata"] = with_packed_lengths(
        bytes(fields["metadata"]), np.asarray(lengths, dtype=np.int64)  # type: ignore[arg-type]
    )
    restored["basis"] = payload
    restored["basis_bits"] = bit_count
    return assemble_carrier_body(restored)
