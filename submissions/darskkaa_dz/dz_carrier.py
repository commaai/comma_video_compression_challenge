import struct
import numpy as np
from dz_bitcoder import Decoder, Models, put_int
RHO_NUM = (0, 1, 2, 3, 4)

def _basis_pass(coder, models, codes, weights, encode):
    K, C, H, W = codes.shape
    for k in range(K):
        a, b1, b2 = weights[k]
        for c in range(C):
            for y in range(H):
                for x in range(W):
                    r, g = (int(codes[k, 0, y, x]), int(codes[k, 1, y, x]))
                    pred = 0 if c == 0 else a * r + 8 >> 4 if c == 1 else b1 * r + b2 * g + 8 >> 4
                    ctx = ('b', c, min(3, abs(pred)))
                    sctx = ('b', c, (pred > 0) - (pred < 0))
                    res = _put(coder, models, ctx, sctx, int(codes[k, c, y, x]) - pred if encode else 0)
                    codes[k, c, y, x] = pred + res

def _put(coder, models, ctx, sctx, value):
    if not coder.bit(models[ctx, 'z'], 1 if value else 0):
        return 0
    neg = coder.bit(models[sctx, 's'], 1 if value < 0 else 0)
    mag = abs(value) - 1
    n = (mag + 1).bit_length() - 1
    k = 0
    while coder.bit(models[ctx, 'u', k], 1 if k < n else 0):
        k += 1
    rest, out = (mag + 1 - (1 << k), 0)
    for i in range(k - 1, -1, -1):
        out = out << 1 | coder.bit(models['m', ctx[0], ctx[1], k, i], rest >> i & 1)
    mag = (1 << k) + out
    return -mag if neg else mag

def _coef_pass(coder, models, q, rho, encode):
    N, K = q.shape
    for t in range(N):
        for k in range(K):
            pred = rho[k] * (int(q[t - 1, k]) if t else 0) >> 2
            if encode:
                put_int(coder, models, ('c', k), int(q[t, k]) - pred, True)
            else:
                q[t, k] = pred + put_int(coder, models, ('c', k), 0, False)

def decode(blob):
    K, N = struct.unpack_from('<BH', blob)
    rho = list(blob[3:3 + K])
    weights = np.frombuffer(blob[3 + K:3 + 4 * K], dtype=np.int8).reshape(K, 3).astype(int).tolist()
    dec = Decoder(blob[3 + 4 * K:])
    models = Models()
    codes = np.zeros((K, 3, 24, 32), dtype=np.int64)
    _basis_pass(dec, models, codes, weights, False)
    q = np.zeros((N, K), dtype=np.int64)
    _coef_pass(dec, models, q, rho, False)
    return (codes, q)
