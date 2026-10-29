# Custom lossless coder for the PR151 HPAC IHS1 (rowpack) blob.
import numpy as np
from hpac_layout import ROW_COUNTS, ROW_MODULE, TAIL, parse_rowpack, build_rowpack
from hpac_range import Model, Encoder, Decoder, code_int

def WCTX(i, j, d, acc, rows):
    return ('w', d, min(int(acc).bit_length(), 6))

def FCTX(j, vals):
    return ('fe', j % 8)

def _run(c, depths=None, rows=None, tail=None):
    enc = isinstance(c, Encoder)
    m = Model(); n = len(ROW_COUNTS)
    if not enc: depths = np.zeros(n, np.uint8); rows = [None] * n
    for i in range(n):  # depth nibbles, MSB first, context = module + bits so far
        d = int(depths[i]) if enc else 0; got = 0
        for b in range(3, -1, -1):
            got |= c.bit(m, ('d', ROW_MODULE[i], b, got >> (b + 1)), (d >> b) & 1) << b
        depths[i] = got
    for i in range(n):  # weights: context = depth + running magnitude in row
        d = int(depths[i]); out = np.zeros(ROW_COUNTS[i], np.int64)
        if d:
            acc = 2.0 ** (d - 2)
            for j in range(ROW_COUNTS[i]):
                v = code_int(c, m, WCTX(i, j, d, acc, rows), int(rows[i][j]) if enc else 0, d)
                out[j] = v; acc = acc * 0.8 + abs(v) * 0.2
        rows[i] = out
    parts = []; off = 0
    for name, count, size in TAIL:
        dt = "<i2" if size == 2 else "i1"
        vals = np.frombuffer(tail[off:off + count * size], dt).astype(np.int64) if enc else np.zeros(count, np.int64); off += count * size
        mod = name.split('.')[0]; kind = name.split('.')[-1]
        for j in range(count):
            ctx = FCTX(j, vals) if kind == 'weight' else (kind, mod)
            vals[j] = code_int(c, m, ctx, int(vals[j]))
        parts.append(vals.astype(dt).tobytes())
    return depths, rows, b"".join(parts)

def encode(blob):
    depths, rows, tail = parse_rowpack(blob)
    e = Encoder(); _run(e, depths, rows, tail); return e.finish()

def decode(data):
    depths, rows, tail = _run(Decoder(data)); return build_rowpack(depths, rows, tail)
