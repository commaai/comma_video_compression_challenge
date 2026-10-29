# Binary range coder with one adaptive probability estimate per context.
class Model:
    """Adaptive bit probability per context; update step 1/(n+2) until n reaches the limit."""
    def __init__(self, limit=30):
        self.t = {}; self.limit = limit
    def p(self, ctx):
        a = self.t.get(ctx)
        if a is None: a = self.t[ctx] = [32768, 0]
        return a
    def update(self, a, bit):
        n = a[1]; a[0] += ((65536 if bit else 0) - a[0]) // (n + 2)
        if n < self.limit: a[1] = n + 1

def _p12(a):
    return min(max(a[0] >> 4, 1), 4095)

class Encoder:
    def __init__(self):
        self.low = 0; self.range = 0xFFFFFFFF; self.cache = 0; self.csize = 1; self.out = bytearray()
    def _shift(self):
        if self.low < 0xFF000000 or self.low >= 1 << 32:
            carry = self.low >> 32; c = self.cache
            while True:
                self.out.append((c + carry) & 0xFF); c = 0xFF; self.csize -= 1
                if not self.csize: break
            self.cache = (self.low >> 24) & 0xFF
        self.csize += 1; self.low = (self.low << 8) & 0xFFFFFFFF
    def bit(self, m, ctx, b):
        a = m.p(ctx); p1 = _p12(a); bound = (self.range >> 12) * p1
        if b: self.range = bound
        else: self.low += bound; self.range -= bound
        m.update(a, b)
        while self.range < 1 << 24: self.range <<= 8; self._shift()
        return b
    def finish(self):
        for _ in range(5): self._shift()
        return bytes(self.out[1:])  # first byte is always the initial zero cache

class Decoder:
    def __init__(self, data):
        self.d = data; self.i = 0; self.range = 0xFFFFFFFF; self.code = 0
        for _ in range(4): self.code = (self.code << 8) | self._byte()
    def _byte(self):
        b = self.d[self.i] if self.i < len(self.d) else 0; self.i += 1; return b
    def bit(self, m, ctx, _=None):
        a = m.p(ctx); p1 = _p12(a); bound = (self.range >> 12) * p1
        if self.code < bound: self.range = bound; b = 1
        else: self.code -= bound; self.range -= bound; b = 0
        m.update(a, b)
        while self.range < 1 << 24: self.range <<= 8; self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF
        return b

def code_int(c, m, ctx, v=0, bound_bits=None):
    """Signed integer: zero flag, unary bucket (truncated if bounded), mantissa bits, sign.
    Same routine for Encoder (v is coded) and Decoder (v ignored, value returned)."""
    if c.bit(m, (ctx, 'z'), int(v == 0)):
        return 0
    kmax = bound_bits - 1 if bound_bits else 30
    rest = abs(v) - 1; k = 0; base = 0
    while k < kmax and c.bit(m, (ctx, 'u', k), int(rest - base >= (1 << k))):
        base += 1 << k; k += 1
    r = 0
    for b in range(k - 1, -1, -1):
        r |= c.bit(m, (ctx, 'b', k, b if b >= k - 2 else -1), ((rest - base) >> b) & 1) << b
    mag = base + r + 1
    return -mag if c.bit(m, (ctx, 's'), int(v < 0)) else mag
