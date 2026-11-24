# PR151 HPAC rowpack layout; depth-grouped zigzag bitplanes follow the #148-style format.
import numpy as np

def mask_count(k, delta, kind):
    c = (k - 1) // 2; n = 0
    for r in range(k):
        for q in range(k):
            off = q - c + delta * (r - c)
            if off < 0 or (kind == "B" and off == 0): n += 1
    return n

MODULES = [  # name, rows, row_len
    ("frame_shift", 64, 8), ("frame_scale", 64, 8),
    ("conv_a", 64, 7 * mask_count(7, 2, "A")), ("conv_b1", 64, mask_count(5, 2, "B")),
    ("conv_b2", 64, mask_count(3, 2, "B")), ("conv_past", 64, 5 * 9),
    ("spm_dw", 64, 9), ("spm_pw", 64, 64), ("head", 5, 64)]
ROW_COUNTS = [n for _, r, n in MODULES for _ in range(r)]
ROW_MODULE = [i for i, (_, r, n) in enumerate(MODULES) for _ in range(r)]
TAIL = [("frame_embed.weight", 4800, 1)] + [(f"{m}.{f}", r, 2 if f == "bias" else 1) for m, r, _ in MODULES for f in ("bias", "exponent")]
TAIL_BYTES = sum(c * s for _, c, s in TAIL)
HEAD_BYTES = 4 + (len(ROW_COUNTS) + 1) // 2

def zz(v): v = np.asarray(v, np.int64); return np.where(v >= 0, 2 * v, -2 * v - 1)
def unzz(u): u = np.asarray(u, np.int64); return np.where(u % 2 == 0, u // 2, -(u + 1) // 2)

def depths_of(blob):
    raw = np.frombuffer(blob[4:HEAD_BYTES], np.uint8); v = np.empty(raw.size * 2, np.uint8)
    v[0::2] = raw & 15; v[1::2] = raw >> 4; return v[:len(ROW_COUNTS)]

def parse_rowpack(blob):
    """-> depths, rows(list of int arrays), tail bytes"""
    assert blob[:4] == b"IHS1"
    depths = depths_of(blob); off = HEAD_BYTES; rows = [None] * len(depths)
    for d in sorted(set(int(x) for x in depths if x > 0)):
        idx = [i for i, x in enumerate(depths) if x == d]; n = sum(ROW_COUNTS[i] for i in idx); nb = (n + 7) // 8
        vals = np.zeros(n, np.int64)
        for b in range(d):
            vals |= np.unpackbits(np.frombuffer(blob[off:off + nb], np.uint8))[:n].astype(np.int64) << b; off += nb
        sv = unzz(vals); pos = 0
        for i in idx: rows[i] = sv[pos:pos + ROW_COUNTS[i]]; pos += ROW_COUNTS[i]
    for i, x in enumerate(depths):
        if x == 0: rows[i] = np.zeros(ROW_COUNTS[i], np.int64)
    tail = blob[off:]
    assert len(tail) == TAIL_BYTES, (len(tail), TAIL_BYTES)
    return depths, rows, tail

def build_rowpack(depths, rows, tail):
    raw = np.zeros(((len(depths) + 1) // 2) * 2, np.uint8); raw[:len(depths)] = depths
    out = bytearray(b"IHS1" + (raw[0::2] | (raw[1::2] << 4)).tobytes())
    for d in sorted(set(int(x) for x in depths if x > 0)):
        idx = [i for i, x in enumerate(depths) if x == d]; vals = np.concatenate([zz(rows[i]) for i in idx]).astype(np.uint32)
        for b in range(d): out += np.packbits(((vals >> b) & 1).astype(np.uint8)).tobytes()
    return bytes(out) + tail
