import numpy as np
from dz_bitcoder import Decoder, Models
MAGIC = b'IHS1'

def _mag(c, m, ctx, mag):
    n = (mag + 1).bit_length() - 1
    k = 0
    while c.bit(m[ctx, 'u', k], 1 if k < n else 0):
        k += 1
    rest, pre = (mag + 1 - (1 << k), 1)
    out = 0
    for i in range(k - 1, -1, -1):
        b = c.bit(m['m', ctx[0], k, pre if k - 1 - i < 2 else 0, i], rest >> i & 1)
        out = out << 1 | b
        pre = pre << 1 | b
    return (1 << k) + out - 1

def _int(c, m, ctx, sctx, v):
    if not c.bit(m[ctx, 'z'], 1 if v else 0):
        return 0
    neg = c.bit(m[sctx, 's'], 1 if v < 0 else 0)
    mag = _mag(c, m, ctx, abs(v) - 1) + 1
    return -mag if neg else mag

def _run(c, counts, fields, depths, rows, tail):
    enc = depths is not None
    m = Models()
    n = len(counts)
    if not enc:
        depths, rows = ([0] * n, [None] * n)
    for i in range(n):
        d, got = (int(depths[i]), 0)
        for b in range(3, -1, -1):
            got = got << 1 | c.bit(m['d', counts[i], b, got], d >> b & 1)
        depths[i] = got
    for i in range(n):
        d = int(depths[i])
        out = np.zeros(counts[i], np.int64)
        if d:
            acc = 2.0 ** (d - 2)
            for j in range(counts[i]):
                v = _int(c, m, (d, min(int(acc).bit_length(), 9)), (d,), int(rows[i][j]) if enc else 0)
                out[j] = v
                acc = acc * 0.75 + abs(v) * 0.25
        rows[i] = out
    parts, off = ([], 0)
    for name, nbytes, itemsize in fields:
        dt = '<i2' if itemsize == 2 else 'i1'
        cnt = nbytes // itemsize
        vals = np.frombuffer(tail[off:off + nbytes], dt).astype(np.int64) if enc else np.zeros(cnt, np.int64)
        off += nbytes
        mod, kind = (name.split('.')[0], name.split('.')[-1])
        for j in range(cnt):
            ctx = ('fe', j % 8) if kind == 'weight' else (kind, mod)
            vals[j] = _int(c, m, ctx, ctx, int(vals[j]) if enc else 0)
        parts.append(vals.astype(dt).tobytes())
    return (depths, rows, b''.join(parts))

def _layout(ihs2, layout):
    return (list(layout.row_counts), [(f.name, f.byte_count, f.itemsize) for f in layout.tail_fields])

def decode(data, ihs2, layout):
    counts, fields = _layout(ihs2, layout)
    depths, rows, tail = _run(Decoder(data), counts, fields, None, None, None)
    depths = np.asarray(depths, dtype=np.uint8)
    body = ihs2._write_signed_rows(tuple((r.astype(np.int16) for r in rows)), depths, layout)
    return MAGIC + ihs2._pack_nibbles(depths) + body + tail
