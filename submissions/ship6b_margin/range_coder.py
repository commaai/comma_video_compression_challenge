"""Pure-Python range coder for seg_targets compression.

Stdlib-only (no constriction, no numpy needed for the coder core -- numpy
used only for input-prep helpers outside the hot loop). Shipped in the
scoring archive.

Bitstream format (standard carry-less 32-bit range coder):
- 32-bit state (low, high), byte-level renormalization
- frequency tables are fixed-point uint16 (total_freq = 2^precision, typically 1<<16)
- symbols are uint8 in range [0, n_classes)

This is the Witten/Neal/Cleary arithmetic-coding algorithm rewritten with
plain 32-bit ints and E1/E2/E3 renormalization. Implementation is
byte-identical lossless by construction (encode/decode share arithmetic).
"""

from typing import List, Sequence


TOP = 0xFFFFFFFF
HALF = 0x80000000
QUARTER = 0x40000000
THREE_QUARTER = 0xC0000000


class RangeEncoder:
    """Encode symbols under per-symbol CDFs into a byte stream."""

    def __init__(self) -> None:
        self.low: int = 0
        self.high: int = TOP
        self.pending: int = 0
        self.out: bytearray = bytearray()
        self._byte_buf: int = 0
        self._byte_bits: int = 0

    def _emit_bit(self, bit: int) -> None:
        self._byte_buf = (self._byte_buf << 1) | bit
        self._byte_bits += 1
        if self._byte_bits == 8:
            self.out.append(self._byte_buf & 0xFF)
            self._byte_buf = 0
            self._byte_bits = 0

    def _emit_bit_and_pending(self, bit: int) -> None:
        self._emit_bit(bit)
        neg = 1 - bit
        for _ in range(self.pending):
            self._emit_bit(neg)
        self.pending = 0

    def encode_symbol(self, cum_low: int, cum_high: int, total: int) -> None:
        """Encode one symbol defined by its cumulative range [cum_low, cum_high)."""
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum_high) // total - 1
        self.low = self.low + (rng * cum_low) // total

        while True:
            if self.high < HALF:
                self._emit_bit_and_pending(0)
            elif self.low >= HALF:
                self._emit_bit_and_pending(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & TOP
            self.high = ((self.high << 1) | 1) & TOP

    def finish(self) -> bytes:
        """Flush remaining state; returns the encoded byte string."""
        self.pending += 1
        if self.low < QUARTER:
            self._emit_bit_and_pending(0)
        else:
            self._emit_bit_and_pending(1)
        # pad final byte
        if self._byte_bits > 0:
            self.out.append((self._byte_buf << (8 - self._byte_bits)) & 0xFF)
            self._byte_buf = 0
            self._byte_bits = 0
        return bytes(self.out)


class RangeDecoder:
    """Decode symbols from a byte stream using per-symbol CDFs.

    Matches RangeEncoder exactly; constructor consumes the first 32 bits.
    """

    def __init__(self, data: bytes) -> None:
        self.data: bytes = data
        self.byte_pos: int = 0
        self.bit_pos: int = 0  # bits consumed within self.data[byte_pos]
        self.low: int = 0
        self.high: int = TOP
        self.code: int = 0
        for _ in range(32):
            self.code = ((self.code << 1) | self._read_bit()) & TOP

    def _read_bit(self) -> int:
        if self.byte_pos >= len(self.data):
            return 0  # read past EOF -> zeros (matches encoder terminator)
        byte = self.data[self.byte_pos]
        bit = (byte >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        return bit

    def decode_target(self, total: int) -> int:
        """Return the 'offset' in [0, total) that the current state decodes to.

        Caller must then look up which symbol owns this offset under the same
        CDF as the encoder used, and pass cum_low/cum_high to advance().
        """
        rng = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // rng

    def advance(self, cum_low: int, cum_high: int, total: int) -> None:
        """Narrow the decoder state to match the consumed symbol."""
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum_high) // total - 1
        self.low = self.low + (rng * cum_low) // total

        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break
            self.low = (self.low << 1) & TOP
            self.high = ((self.high << 1) | 1) & TOP
            self.code = ((self.code << 1) | self._read_bit()) & TOP


def cdfs_from_freqs(freqs: Sequence[int]) -> List[int]:
    """Build cumulative frequency array [0, f0, f0+f1, ..., total]."""
    cdf = [0]
    acc = 0
    for f in freqs:
        acc += f
        cdf.append(acc)
    return cdf
