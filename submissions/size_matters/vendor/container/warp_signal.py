"""Real byte-level serialization of the
per-frame warp-choice sequence, as a canonical Huffman code.

Format (all per-video data lives in the serialized bytes -- nothing is
hardcoded in source):

    header: 5 x 4-bit code lengths, one per WARP_ID_ORDER symbol (20 bits,
            padded to 3 bytes) -- enough to reconstruct the canonical code,
            no tree/choice data embedded in source.
    payload: bit-packed canonical-Huffman-coded warp IDs, MSB-first,
             one codeword per frame, in frame order.
    padding: payload is padded to a byte boundary with zero bits.

The caller decodes exactly `count` symbols from the payload (count == the
renderer's fixed frame count, N=600 for this challenge video -- not stored
in the signal itself, since it's already an existing constant of the
archive's fixed format, same as today).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

WARP_ID_ORDER: tuple[str, ...] = ("identity", "dx-1", "dx+1", "dy-1", "dy+1")
HEADER_LENGTH_BITS = 4  # per symbol; max representable length 15, far above any real Huffman length for 5 symbols
HEADER_BITS = len(WARP_ID_ORDER) * HEADER_LENGTH_BITS
HEADER_BYTES = (HEADER_BITS + 7) // 8


def _huffman_code_lengths(frequencies: dict[str, int]) -> dict[str, int]:
    """Standard Huffman-tree code-length derivation (exact, deterministic)."""
    import heapq

    if len(frequencies) == 1:
        # Degenerate single-symbol case: Huffman normally needs >=2 leaves;
        # assign length 1 so the format still works (never hit for this
        # project's observed 5-symbol-with-nonzero-identity distribution,
        # but handled for robustness).
        only = next(iter(frequencies))
        return {only: 1}

    heap = [[freq, [sym, 0]] for sym, freq in frequencies.items() if freq > 0]
    heapq.heapify(heap)
    counter = 0
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] += 1
        for pair in hi[1:]:
            pair[1] += 1
        counter += 1
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    lengths = {sym: length for sym, length in heap[0][1:]}
    for sym in frequencies:
        lengths.setdefault(sym, 1)
    return lengths


def _canonical_codes(lengths: dict[str, int]) -> dict[str, str]:
    """Canonical Huffman code assignment from code lengths alone (RFC1951-style)."""
    symbols_by_length = sorted(lengths.items(), key=lambda kv: (kv[1], WARP_ID_ORDER.index(kv[0])))
    codes: dict[str, str] = {}
    code = 0
    prev_length = 0
    for sym, length in symbols_by_length:
        code <<= (length - prev_length)
        codes[sym] = format(code, f"0{length}b")
        code += 1
        prev_length = length
    return codes


class _BitWriter:
    def __init__(self) -> None:
        self.bytes_out = bytearray()
        self.cur = 0
        self.nbits = 0

    def write_bits(self, bitstring: str) -> None:
        for ch in bitstring:
            self.cur = (self.cur << 1) | (1 if ch == "1" else 0)
            self.nbits += 1
            if self.nbits == 8:
                self.bytes_out.append(self.cur)
                self.cur = 0
                self.nbits = 0

    def finish(self) -> tuple[bytes, int]:
        """Returns (bytes, padding_bits_added)."""
        padding = 0
        if self.nbits:
            padding = 8 - self.nbits
            self.cur <<= padding
            self.bytes_out.append(self.cur)
            self.cur = 0
            self.nbits = 0
        return bytes(self.bytes_out), padding


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0  # bit position

    def read_bit(self) -> int:
        byte_index = self.pos >> 3
        bit_index = self.pos & 7
        bit = (self.data[byte_index] >> (7 - bit_index)) & 1
        self.pos += 1
        return bit

    def read_bits(self, n: int) -> int:
        value = 0
        for _ in range(n):
            value = (value << 1) | self.read_bit()
        return value


@dataclass
class WarpSignalStats:
    meaningful_bits: int
    header_bits: int
    payload_bits: int
    payload_padding_bits: int
    stored_bytes: int
    sha256: str


def encode_warp_signal(warp_ids: list[str]) -> tuple[bytes, WarpSignalStats]:
    """Serialize `warp_ids` (one per frame, values from WARP_ID_ORDER) to real bytes."""
    if any(w not in WARP_ID_ORDER for w in warp_ids):
        bad = sorted(set(warp_ids) - set(WARP_ID_ORDER))
        raise ValueError(f"unknown warp id(s): {bad}")

    frequencies = {w: 0 for w in WARP_ID_ORDER}
    for w in warp_ids:
        frequencies[w] += 1
    nonzero_frequencies = {w: f for w, f in frequencies.items() if f > 0}
    lengths_nonzero = _huffman_code_lengths(nonzero_frequencies)
    # Symbols with zero observed frequency still need a header entry (length
    # 0 = "unused"), so the decoder's fixed-order header stays fixed-size.
    lengths = {w: lengths_nonzero.get(w, 0) for w in WARP_ID_ORDER}
    codes = _canonical_codes(lengths_nonzero)

    header_writer = _BitWriter()
    for w in WARP_ID_ORDER:
        header_writer.write_bits(format(lengths[w], f"0{HEADER_LENGTH_BITS}b"))
    header_bytes, _header_padding = header_writer.finish()
    assert len(header_bytes) == HEADER_BYTES
    header_bits = len(WARP_ID_ORDER) * HEADER_LENGTH_BITS

    payload_writer = _BitWriter()
    payload_bits = 0
    for w in warp_ids:
        payload_writer.write_bits(codes[w])
        payload_bits += len(codes[w])
    payload_bytes, payload_padding = payload_writer.finish()

    data = header_bytes + payload_bytes
    stats = WarpSignalStats(
        meaningful_bits=header_bits + payload_bits,
        header_bits=header_bits,
        payload_bits=payload_bits,
        payload_padding_bits=payload_padding,
        stored_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    return data, stats


def decode_warp_signal(data: bytes, count: int) -> list[str]:
    """Reconstruct `count` warp IDs from serialized bytes -- header-driven, no hardcoded choices."""
    reader = _BitReader(data)
    lengths = {}
    for w in WARP_ID_ORDER:
        length = reader.read_bits(HEADER_LENGTH_BITS)
        if length > 0:
            lengths[w] = length
    codes = _canonical_codes(lengths)
    decode_table = {code: sym for sym, code in codes.items()}
    max_len = max(len(c) for c in decode_table) if decode_table else 0

    # Byte-align to the payload start (header occupies HEADER_BYTES bytes).
    reader.pos = HEADER_BYTES * 8

    result = []
    for _ in range(count):
        code = ""
        while True:
            code += str(reader.read_bit())
            if code in decode_table:
                result.append(decode_table[code])
                break
            if len(code) > max_len:
                raise ValueError(f"no matching Huffman code for prefix {code!r}")
    return result
