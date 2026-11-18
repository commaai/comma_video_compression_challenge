"""RID9 = a lossless re-encoding of #150's RID2 renderer body (SM3R mode 8) in which the large bit-packed integer code
arrays are arithmetic-coded with adaptive per-array models (folding.rid2's own deterministic integer coder), and every
other byte (headers, scales, small arrays) is brotli-compressed.

stored body:  b"SM3R" | ver u8 = 1 | mode u8 = 9 | n u8 | n x (gap_len varint, count varint, bits u8, inc u8)
              | tail_gap_len varint | u32 len(G) | G = brotli(concat(gaps)) | AC stream
decode:       rebuild the exact mode-8 blob = gap_0 + pack(a_0) + gap_1 + ... + pack(a_{n-1}) + tail_gap, which then goes
              through the unchanged folding.rid2.decode_state_dict.  Fail closed on any inconsistency.
"""
import struct
import brotli
import numpy as np
from folding.rid2 import Enc, Dec, Model, RID2Error

MAGIC = b"SM3R"; MODE = 9


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F; n >>= 7
        if n: out.append(b | 0x80)
        else: out.append(b); return bytes(out)


def _unvarint(buf, o):
    n = sh = 0
    while True:
        b = buf[o]; o += 1; n |= (b & 0x7F) << sh; sh += 7
        if not b & 0x80: return n, o


def pack_codes(codes, bits):
    """signed codes -> little-endian bit-packed bytes (identical to rid2._unq / sch._unpack_signed_bits layout)."""
    u = (np.asarray(codes, np.int64) & ((1 << bits) - 1)).astype(np.uint16)
    b = ((u[:, None] >> np.arange(bits, dtype=np.uint16)[None]) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(b, bitorder="little").tobytes()


def unpack_codes(data, count, bits):
    b = np.unpackbits(np.frombuffer(data, np.uint8), bitorder="little")[:count * bits].reshape(count, bits).astype(np.int64)
    u = (b * (1 << np.arange(bits, dtype=np.int64))[None]).sum(1)
    return np.where(u >= (1 << (bits - 1)), u - (1 << bits), u)


def _code_array(enc_or_dec, values_or_count, bits, inc, decode=False):
    m = Model(1 << bits, inc); off = 1 << (bits - 1)
    if not decode:
        for v in values_or_count.tolist():
            s = int(v) + off; cum = m.cum(); enc_or_dec.put(cum[s], cum[s + 1], cum[-1]); m.update(s)
        return None
    out = np.empty(values_or_count, np.int64)
    for i in range(values_or_count):
        cum = m.cum(); s = enc_or_dec.get(cum, cum[-1]); m.update(s); out[i] = s - off
    return out


def encode(blob8, arrays, inc=32):
    """blob8: exact mode-8 body; arrays: list of (offset, count, bits, codes) in increasing offset order (byte-aligned spans)."""
    gaps, hdr, pos = [], bytearray(), 0
    for off, cnt, bits, codes in arrays:
        packed = pack_codes(codes, bits)
        if blob8[off:off + len(packed)] != packed: raise RID2Error("array span does not match blob")
        gaps.append(blob8[pos:off]); hdr += _varint(off - pos) + _varint(cnt) + bytes([bits, inc]); pos = off + len(packed)
    gaps.append(blob8[pos:]); hdr += _varint(len(blob8) - pos)
    G = brotli.compress(b"".join(gaps), quality=11)
    enc = Enc()
    for off, cnt, bits, codes in arrays:
        _code_array(enc, np.asarray(codes, np.int64), bits, inc)
    body = MAGIC + bytes([1, MODE, len(arrays)]) + bytes(hdr) + struct.pack("<I", len(G)) + G + enc.finish()
    if decode_to_blob8(body) != blob8: raise RID2Error("RID9 round trip failed")
    return body


def decode_to_blob8(body):
    if body[:4] != MAGIC or body[4] != 1 or body[5] != MODE: raise RID2Error("bad RID9 header")
    n = body[6]; o = 7; specs = []
    for _ in range(n):
        g, o = _unvarint(body, o); c, o = _unvarint(body, o); bits, inc = body[o], body[o + 1]; o += 2; specs.append((g, c, bits, inc))
    tail, o = _unvarint(body, o)
    (gl,) = struct.unpack_from("<I", body, o); o += 4
    gaps = brotli.decompress(body[o:o + gl]); o += gl
    dec = Dec(body[o:]); out = bytearray(); gp = 0
    for g, c, bits, inc in specs:
        out += gaps[gp:gp + g]; gp += g
        out += pack_codes(_code_array(dec, c, bits, inc, decode=True), bits)
    out += gaps[gp:gp + tail]; gp += tail
    if gp != len(gaps): raise RID2Error("RID9 gap accounting mismatch")
    return bytes(out)


def decode_state_dict(body, template_sd):
    from folding.rid2 import decode_state_dict as rid2_decode
    return rid2_decode(decode_to_blob8(body), template_sd)
