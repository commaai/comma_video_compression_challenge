PROB_BITS = 12
ONE = 1 << PROB_BITS

class Bit:
    __slots__ = ('n0', 'n1')
    LIMIT = 1023

    def __init__(self):
        self.n0 = 0
        self.n1 = 0

    def p0(self):
        p = (2 * self.n0 + 1 << PROB_BITS) // (2 * (self.n0 + self.n1) + 2)
        return min(max(p, 8), ONE - 8)

    def update(self, bit):
        if bit:
            self.n1 += 1
        else:
            self.n0 += 1
        if self.n0 + self.n1 > self.LIMIT:
            self.n0 = self.n0 + 1 >> 1
            self.n1 = self.n1 + 1 >> 1

class Decoder:

    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.range = 4294967295
        self.code = 0
        for _ in range(4):
            self.code = self.code << 8 | self._byte()

    def _byte(self):
        b = self.data[self.pos] if self.pos < len(self.data) else 0
        self.pos += 1
        return b

    def decode(self, model):
        bound = (self.range >> PROB_BITS) * model.p0()
        if self.code < bound:
            self.range = bound
            bit = 0
        else:
            self.code -= bound
            self.range -= bound
            bit = 1
        model.update(bit)
        while self.range < 1 << 24:
            self.range = self.range << 8 & 4294967295
            self.code = (self.code << 8 | self._byte()) & 4294967295
        return bit

    def bit(self, model, _value=0):
        return self.decode(model)

class Models(dict):

    def __missing__(self, key):
        m = self[key] = Bit()
        return m

def put_uint(coder, models, ctx, value, encode=True):
    if encode:
        n = (value + 1).bit_length() - 1
        for i in range(n):
            coder.encode(models[ctx, 'u', i], 1)
        coder.encode(models[ctx, 'u', n], 0)
        rest = value + 1 - (1 << n)
        for i in range(n - 1, -1, -1):
            coder.encode(models[ctx, 'm', n, i] if i >= n - 2 else models['raw',], rest >> i & 1)
        return value
    n = 0
    while coder.decode(models[ctx, 'u', n]):
        n += 1
    rest = 0
    for i in range(n - 1, -1, -1):
        rest = rest << 1 | coder.decode(models[ctx, 'm', n, i] if i >= n - 2 else models['raw',])
    return (1 << n) + rest - 1

def put_int(coder, models, ctx, value, encode=True):
    if encode:
        coder.encode(models[ctx, 'z'], 1 if value else 0)
        if value:
            coder.encode(models[ctx, 's'], 1 if value < 0 else 0)
            put_uint(coder, models, ctx, abs(value) - 1, True)
        return value
    if not coder.decode(models[ctx, 'z']):
        return 0
    neg = coder.decode(models[ctx, 's'])
    mag = put_uint(coder, models, ctx, 0, False) + 1
    return -mag if neg else mag
