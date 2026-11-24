import struct
MAGIC = b'DZ1'
HEAD = struct.Struct('<3sBHHH')
HPAC_DZ, MOTION, TOKEN_MC, RENDER = (1, 2, 4, 8)

def unpack(p):
    magic, flags, nh, ns, nc = HEAD.unpack_from(p)
    if magic != MAGIC:
        raise ValueError('not a DZ1 payload')
    o = HEAD.size
    nm = 0
    if flags & TOKEN_MC:
        nm, = struct.unpack_from('<H', p, o)
        o += 2
    hpac = p[o:o + nh]
    o += nh
    semantic = p[o:o + ns]
    o += ns
    slaves = p[o:o + nc]
    o += nc
    mc = p[o:o + nm]
    o += nm
    return (flags, hpac, semantic, slaves, mc, p[o:])

def split_151(p):
    if p[:4] != b'BLK2':
        raise ValueError('expected a BLK2 payload')
    size, count = struct.unpack_from('<IB', p, 4)
    if count != 1 or p[9] != 0 or int.from_bytes(p[10:13], 'little') != size:
        raise ValueError('expected one stored model block')
    models = p[13:13 + size]
    if models[:6] != b'RX1M\x01\x02':
        raise ValueError('expected RX1M/brotli models')
    nh, ns, nc = struct.unpack_from('<HHH', models, 8)
    o = 14
    hpac = models[o:o + nh]
    o += nh
    semantic = models[o:o + ns]
    o += ns
    carrier = models[o:o + nc]
    return (models[6], models[7], hpac, semantic, carrier, p[13 + size:])
