"""Versioned extension container (decoder side + encoder side).
Archive trailer (unchanged):  warp' = warp_signal || ext_payload || u16(len(ext_payload)),  then the usual outer u16(len(warp')).
The decoder derives len(warp_signal) from the warp Huffman stream itself (symbols consumed), so an archive WITHOUT extension is recognised by remainder == 0.
ext_payload = records until the end:
  tag 1..5  EXT1 carrier-grid/DC/plane/DCT records (byte-identical):  [tag][bits][step*8][N] + N varint id deltas + N*m signed `bits`-bit LSB-first coefficients (byte padded)
  tag 0x20  EXT2 master-correction record: [0x20][varint body_len][body]   (body defined by the master-correction ops)
Unknown tags raise ExtError (fail closed)."""
import struct
FAM = {"grid4": (1, 16), "dc": (2, 1), "plane": (3, 3), "dct8": (4, 8), "grid2": (5, 4)}
FAMID = {v[0]: k for k, v in FAM.items()}
EXT2_TAG = 0x20


class ExtError(ValueError): pass


def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F; n >>= 7
        if n: out.append(b | 0x80)
        else: out.append(b); return bytes(out)


def unvarint(buf, o):
    n = 0; sh = 0
    while True:
        b = buf[o]; o += 1; n |= (b & 0x7F) << sh; sh += 7
        if not b & 0x80: return n, o


def pack_signed(vals, bits):
    acc = avail = 0; out = bytearray()
    for v in vals:
        v = int(v); assert -(1 << (bits - 1)) <= v < (1 << (bits - 1)), v
        acc |= (v & ((1 << bits) - 1)) << avail; avail += bits
        while avail >= 8: out.append(acc & 0xFF); acc >>= 8; avail -= 8
    if avail: out.append(acc & 0xFF)
    return bytes(out)


def unpack_signed(buf, o, n, bits):
    acc = avail = 0; out = []
    for _ in range(n):
        while avail < bits: acc |= buf[o] << avail; o += 1; avail += 8
        v = acc & ((1 << bits) - 1); acc >>= bits; avail -= bits
        out.append(v - (1 << bits) if v >= (1 << (bits - 1)) else v)
    return out, o


def ext1_record(q: dict, bits: int, step: float, fam="grid4"):
    ids = sorted(q); head = bytes([FAM[fam][0], bits, int(round(step * 8)), len(ids)]); prev = 0; idb = bytearray()
    for p in ids: idb += varint(p - prev); prev = p
    return head + bytes(idb) + pack_signed([x for p in ids for x in q[p]], bits)


def ext2_record(body: bytes): return bytes([EXT2_TAG]) + varint(len(body)) + body


def parse_payload(payload: bytes):
    """-> dict(ext1=dict(fam,bits,step,q{pair:ints}) | None, ext2=bytes | None)"""
    o = 0; out = dict(ext1=None, ext2=None)
    while o < len(payload):
        tag = payload[o]
        if tag in FAMID:
            if out["ext1"] is not None: raise ExtError("duplicate EXT1")
            fam = FAMID[tag]; mm = FAM[fam][1]; bits, st8, n = payload[o + 1], payload[o + 2], payload[o + 3]; o += 4; ids = []; prev = 0
            for _ in range(n): d, o = unvarint(payload, o); prev += d; ids.append(prev)
            vals, o = unpack_signed(payload, o, n * mm, bits)
            if o > len(payload): raise ExtError("truncated EXT1")
            out["ext1"] = dict(fam=fam, bits=bits, step=st8 / 8.0, q={p: vals[mm * k:mm * k + mm] for k, p in enumerate(ids)})
        elif tag == EXT2_TAG:
            n, o2 = unvarint(payload, o + 1)
            if o2 + n > len(payload): raise ExtError("truncated EXT2")
            if out["ext2"] is not None: raise ExtError("duplicate EXT2")
            out["ext2"] = bytes(payload[o2:o2 + n]); o = o2 + n
        else: raise ExtError(f"unknown extension tag {tag}")
    return out


def warp_consumed_len(w2: bytes, count: int):
    """number of bytes of `w2` used by the warp Huffman signal (header + count codes, byte padded)."""
    from container.warp_signal import _BitReader, WARP_ID_ORDER, HEADER_LENGTH_BITS, HEADER_BYTES, _canonical_codes
    r = _BitReader(w2); lengths = {}
    for w in WARP_ID_ORDER:
        L = r.read_bits(HEADER_LENGTH_BITS)
        if L > 0: lengths[w] = L
    codes = _canonical_codes(lengths); table = {c: s for s, c in codes.items()}; mx = max(len(c) for c in table); r.pos = HEADER_BYTES * 8
    for _ in range(count):
        code = ""
        while True:
            code += str(r.read_bit())
            if code in table: break
            if len(code) > mx: raise ExtError("bad warp stream")
    return (r.pos + 7) // 8


def split_trailer(w2: bytes, count=600):
    """-> (warp_signal, ext_payload | b'')"""
    n = warp_consumed_len(w2, count); rest = w2[n:]
    if not rest: return w2, b""
    ln = struct.unpack("<H", rest[-2:])[0]
    if ln != len(rest) - 2: raise ExtError("extension trailer length mismatch")
    return w2[:n], rest[:-2]


def join_trailer(warp: bytes, payload: bytes): return warp + ((payload + struct.pack("<H", len(payload))) if payload else b"")
