"""Integer context-mixing coder for the small non-token sections of tcm_ctxmix: the renderer header "gaps", the
carrier prefix, the RR5 carrier basis and the carrier coefficients.

Everything here is integer arithmetic on Python ints: no float, no true division, no math module, no numpy.
Python ints are exact and `>>` / `//` round toward minus infinity on every platform, so the coder is bit-identical
on any CPU and any Python 3 (work/sub/test_ncm.py scans this file for float use and cross-checks under x86_64).

One code path serves the encoder and the decoder: a model calls `cm.code_bit(contexts, weight_set, bit)`; with an
ArithEncoder the given bit is coded and returned, with an ArithDecoder the argument is ignored and the decoded bit is
returned.  Every context is computed from already coded values only.

Pieces (the usual lpaq/paq building blocks, same family as tcm/tcm.c):
  * bit counters: probability P(bit=1) in 22 bits + a hit count; adaptive rate 1/(count+1.5) until `limit`
  * logistic mixer: inputs = stretch(p) = ln(p/(1-p)) in 1/256 units, weights in 1/65536 units, selected by `weight_set`
  * 32-bit carry-less binary arithmetic coder driven by a 16-bit probability
The stretch/squash tables are generated below with integer-only code (fixed-point Taylor series for exp).
"""

MASK32 = 0xFFFFFFFF
ST_MAX = 2047                  # stretch domain: +-2047/256 = +-8 nats
P_BITS = 22                    # counter precision
P_HALF = 1 << (P_BITS - 1)
P_MIN = 1 << (P_BITS - 11)     # counters stay within [1/2048, 1 - 1/2048]
BIAS_INPUT = 77                # constant mixer input (0.3 nats)
LR_SHIFT = 18


def _make_tables():
    """SQUASH[x + ST_MAX] = round(65536 / (1 + exp(-x/256))) for x in -ST_MAX..ST_MAX; STRETCH[p16] = its inverse."""
    one = 1 << 64
    step = term = one                                   # step -> exp(1/256) in 64-bit fixed point (Taylor series)
    for k in range(1, 24):
        term //= 256 * k
        step += term
    squash = [0] * (2 * ST_MAX + 1)
    power = one                                         # exp(x/256)
    for x in range(ST_MAX + 1):
        p = ((power << 16) + (power + one) // 2) // (power + one)
        squash[ST_MAX + x] = min(p, 65535)
        squash[ST_MAX - x] = max(65536 - p, 1)
        power = (power * step) >> 64
    stretch = [ST_MAX] * 65536
    filled = 0
    for x in range(-ST_MAX, ST_MAX + 1):
        upto = squash[x + ST_MAX]
        for p in range(filled, upto + 1):
            stretch[p] = x
        filled = max(filled, upto + 1)
    return squash, stretch


SQUASH, STRETCH = _make_tables()
RATE = [(2 << 16) // (2 * n + 3) for n in range(1024)]   # 65536 / (n + 1.5)


class ArithEncoder:
    def __init__(self):
        self.x1, self.x2, self.out = 0, MASK32, bytearray()

    def code(self, bit, p16):
        """Code `bit` with P(bit=1) = p16/65536 (1..65535)."""
        span = self.x2 - self.x1
        xmid = self.x1 + (span >> 16) * p16 + (((span & 0xFFFF) * p16) >> 16)
        if bit:
            self.x2 = xmid
        else:
            self.x1 = xmid + 1
        while not (self.x1 ^ self.x2) & 0xFF000000:
            self.out.append(self.x2 >> 24)
            self.x1 = (self.x1 << 8) & MASK32
            self.x2 = ((self.x2 << 8) & MASK32) | 255
        return bit

    def finish(self):
        self.out.append(self.x1 >> 24)       # the decoder pads with 0xFF: top(x1).FFFFFF lies in [x1, x2]
        return bytes(self.out)


class ArithDecoder:
    def __init__(self, data):
        self.data, self.pos, self.x1, self.x2, self.x = data, 0, 0, MASK32, 0
        for _ in range(4):
            self.x = (self.x << 8) | self._next()

    def _next(self):
        byte = self.data[self.pos] if self.pos < len(self.data) else 0xFF
        self.pos += 1
        return byte

    def code(self, _bit, p16):
        span = self.x2 - self.x1
        xmid = self.x1 + (span >> 16) * p16 + (((span & 0xFFFF) * p16) >> 16)
        bit = 1 if self.x <= xmid else 0
        if bit:
            self.x2 = xmid
        else:
            self.x1 = xmid + 1
        while not (self.x1 ^ self.x2) & 0xFF000000:
            self.x1 = (self.x1 << 8) & MASK32
            self.x2 = ((self.x2 << 8) & MASK32) | 255
            self.x = ((self.x << 8) & MASK32) | self._next()
        return bit

    def check_end(self, what):
        """The encoder writes one byte per decoder shift plus one final byte, and the decoder pre-loads four: after the
        last symbol of an intact stream it has read exactly len(data) + 3 bytes.  A different count proves damage, but
        the same count proves nothing: a stream that is cut or altered, above all near its end, can decode to other
        symbols with the same count.  Integrity comes from the sha256 check of the decoded components
        (tcx_format.check_component); this is only an early, cheap test."""
        if self.pos != len(self.data) + 3:
            raise ValueError(f"{what}: stream length mismatch (decoder position {self.pos}, stream {len(self.data)} B)")


class ContextMixer:
    def __init__(self, coder, lr, init_weight, limit=1023):
        self.coder, self.lr, self.init_weight, self.limit = coder, lr, init_weight, limit
        self.cells = {}        # (input index, context) -> [p (P_BITS), count]
        self.weights = {}      # weight_set -> list of weights (one per input + bias)

    def code_bit(self, contexts, weight_set, bit=0):
        cells, inputs = [], []
        table = self.cells
        for key in enumerate(contexts):
            cell = table.get(key)
            if cell is None:
                cell = table[key] = [P_HALF, 0]
            cells.append(cell)
            inputs.append(STRETCH[cell[0] >> (P_BITS - 16)])
        inputs.append(BIAS_INPUT)
        weights = self.weights.get(weight_set)
        if weights is None:
            weights = self.weights[weight_set] = [self.init_weight] * (len(inputs) - 1) + [0]
        dot = 0
        for w, s in zip(weights, inputs):
            dot += w * s
        x = dot >> 16
        x = ST_MAX if x > ST_MAX else -ST_MAX if x < -ST_MAX else x
        p16 = SQUASH[x + ST_MAX]
        p16 = 16 if p16 < 16 else 65520 if p16 > 65520 else p16
        bit = self.coder.code(bit, p16)
        err = ((bit << 16) - p16) * self.lr
        for j, s in enumerate(inputs):
            weights[j] += (s * err) >> LR_SHIFT
        target, limit = bit << P_BITS, self.limit
        for cell in cells:
            n = cell[1]
            p = cell[0] + (((target - cell[0]) * RATE[n]) >> 16)
            cell[0] = P_MIN if p < P_MIN else (1 << P_BITS) - P_MIN if p > (1 << P_BITS) - P_MIN else p
            if n < limit:
                cell[1] = n + 1
        return bit


# ------------------------------------------------------------------------------------------------ renderer gaps
# The "gaps" are the bytes of #153's SM3R mode-8 renderer body that lie between its arithmetic-coded integer arrays
# (vendor/folding/rid9.py): headers, fp16 scales, the pruned-FiLM row masks and codes.  GAP_WALK lists, per gap, its
# fields as (kind, count, group): kind "b" = bytes, "h" = fp16 words (little-endian), group = context label.
B_HDR, B_SUB, B_MASK, B_PCODE = 0, 1, 2, 3
H_T, H_WC, H_VEC, H_FE, H_DW, H_PW, H_FILM, H_PSCALE, H_HEAD, H_HBIAS = range(10)
_PRUNED_FILM = [("b", 24, B_MASK), ("h", 4, H_PSCALE), ("b", 16, B_PCODE)]
GAP_WALK = (
    [[("b", 15, B_HDR), ("h", 96, H_T)], [("h", 4, H_WC)]] + [[("h", 1, H_VEC)]] * 9
    + [[("b", 17, B_SUB), ("h", 8, H_FE)]]
    + [[("h", 96, H_DW)], [("h", 96, H_PW)], [("h", 192, H_FILM)], [("h", 96, H_DW)], [("h", 96, H_PW)]]
    + [_PRUNED_FILM + [("h", 96, H_DW)], [("h", 96, H_PW)], _PRUNED_FILM + [("h", 96, H_DW)], [("h", 96, H_PW)]]
    + [_PRUNED_FILM + [("h", 3, H_HEAD)], [("h", 3, H_HBIAS)]]
)
# The two header fields hold only settings that are fixed for #153's renderer format (bit depths, modes, fold flags).
# Layouts n4+ treat them as format constants (like the RDQ2/CGA2 magics): they are not coded, the decoder emits them.
GAP_CONSTANTS = {
    # RID2 header: version 1, mode 8, t/wc/v bits 12/11/11, fe_mode 0, fe_inc 1, film_r 0, head_r 0, fold 3, sp 0
    B_HDR: b"SM3R\x01\x08\x0c\x0b\x0b\x00\x01\x00\x00\x03\x00",
    # sub-blob header: version 1, mode 6, keep byte, 3 more bytes, 14 depth nibbles (4 bits, 3 for blocks.0.film)
    B_SUB: b"SM3R\x01\x06\x02\x00\x40\x12\x44\x34\x44\x44\x44\x44\x44",
}
GAP_LENGTHS = [sum(n * (2 if kind == "h" else 1) for kind, n, _ in gap) for gap in GAP_WALK]
GAPS_LR, GAPS_WEIGHT, GAPS_TOP = 24, 12000, 8


def code_gaps(coder, gaps=None, constant_headers=False):
    """Code (ArithEncoder + gaps) or decode (ArithDecoder) the concatenated renderer gaps; returns them as bytes.
    constant_headers: the GAP_CONSTANTS fields are not coded (layouts n4+)."""
    cm = ContextMixer(coder, GAPS_LR, GAPS_WEIGHT)
    out = bytearray()
    previous_byte, instance = 0, 0
    for kind, count, group in (field for gap in GAP_WALK for field in gap):
        if kind == "b" and constant_headers and group in GAP_CONSTANTS:
            constant = GAP_CONSTANTS[group]
            if gaps is not None and gaps[len(out):len(out) + count] != constant:
                raise ValueError("renderer header bytes differ from the format constants")
            out += constant
            previous_byte = constant[-1]
        elif kind == "b":
            for _ in range(count):
                value = gaps[len(out)] if gaps is not None else 0
                node = 1
                for j in range(7, -1, -1):
                    contexts = ((0, node), (1, group, node), (2, previous_byte, node), (3, group, previous_byte, node))
                    node = node * 2 + cm.code_bit(contexts, (0, j), (value >> j) & 1)
                previous_byte = node & 255
                out.append(previous_byte)
        else:
            instance += 1
            previous_top = -1
            for k in range(count):
                value = (gaps[len(out)] | (gaps[len(out) + 1] << 8)) if gaps is not None else 0
                node = 1
                for j in range(15, -1, -1):
                    depth = 15 - j
                    if depth < GAPS_TOP:      # sign, exponent, top mantissa bits: binary tree
                        contexts = ((4, node), (5, group, node), (6, group, previous_top, node),
                                    (7, group, k == 0, node), (8, count > 8, node), (9, instance, node))
                        weight_set = (1, depth)
                    else:                     # mantissa tail: by bit position and the top bits already coded
                        top = node >> (depth - GAPS_TOP)
                        contexts = ((10, depth), (11, group, depth), (12, top & 7, depth), (13, depth, node & 1))
                        weight_set = (2, depth)
                    node = node * 2 + cm.code_bit(contexts, weight_set, (value >> j) & 1)
                word = node & 0xFFFF
                previous_top = word >> (16 - GAPS_TOP)
                out.append(word & 255)
                out.append(word >> 8)
    return bytes(out)


def encode_gaps(gaps, constant_headers=False):
    if len(gaps) != sum(GAP_LENGTHS):
        raise ValueError("renderer gaps do not match the walk")
    encoder = ArithEncoder()
    code_gaps(encoder, gaps, constant_headers)
    return encoder.finish()


def decode_gaps(data, constant_headers=False):
    decoder = ArithDecoder(data)
    gaps = code_gaps(decoder, None, constant_headers)
    decoder.check_end("renderer gaps")
    return gaps


# ------------------------------------------------------------------------------------------------ RR5 carrier basis
# 12 atoms x 3 colour planes x 24x32 five-bit values in [-16, 15] (#153 carrier_joint.py).  Plane 1 is predicted from
# plane 0 and plane 2 from planes 0 and 1 with #153's stored per-atom integer weights (fixed point, `fraction` bits);
# the wrapped, zigzagged residual (0..31) is coded MSB-first as a 5-bit binary tree.
ATOMS, PLANES, ATOM_H, ATOM_W = 12, 3, 24, 32
BASIS_LR, BASIS_WEIGHT = 4, 6000


def _rounded(value, fraction):
    half = 1 << (fraction - 1)
    return (value + half) >> fraction if value >= 0 else -((-value + half) >> fraction)


def code_basis(coder, weights, fraction, atoms=None):
    """Code (ArithEncoder + atoms) or decode (ArithDecoder) the basis; returns atoms[12][3][768].
    weights: 3 per atom (plane 1 <- plane 0; plane 2 <- plane 0, plane 1), as stored by #153."""
    return _code_basis(ContextMixer(coder, BASIS_LR, BASIS_WEIGHT), weights, fraction, atoms)


def _code_basis(cm, weights, fraction, atoms=None):
    size = ATOM_H * ATOM_W
    out = []
    for a in range(ATOMS):
        used = [[], weights[3 * a:3 * a + 1], weights[3 * a + 1:3 * a + 3]]
        atom = []
        for plane in range(PLANES):
            residuals, values = [0] * size, [0] * size
            magnitude_sum = 0
            for i in range(size):
                prediction = _rounded(sum(w * atom[k][i] for k, w in enumerate(used[plane])), fraction) if plane else 0
                symbol = 0
                if atoms is not None:
                    r = ((atoms[a][plane][i] - prediction + 16) & 31) - 16
                    symbol = 2 * r if r >= 0 else -2 * r - 1
                left = abs(residuals[i - 1]) if i % ATOM_W else 0
                up = abs(residuals[i - ATOM_W]) if i >= ATOM_W else 0
                near = min((left + up + 1) // 2, 7)
                first = min(abs(atom[0][i]), 7) if plane else 8          # |plane-0 value| at this position
                predicted = min(abs(prediction), 7)
                mean = min(2 * magnitude_sum // i, 12) if i else 0        # running mean |residual| of this plane, x2
                node = 1
                for j in range(4, -1, -1):
                    contexts = ((plane, node), (a, plane, node), (plane, first, node), (plane, predicted, node),
                                (plane, near, node), (plane, mean, node), (plane, first, mean, node),
                                (a, plane, mean, node), (plane, i // ATOM_W, node))
                    node = node * 2 + cm.code_bit(contexts, node, (symbol >> j) & 1)
                symbol = node - 32
                r = symbol // 2 if symbol % 2 == 0 else -(symbol // 2) - 1
                residuals[i] = r
                magnitude_sum += abs(r)
                values[i] = ((r + prediction + 16) & 31) - 16
            atom.append(values)
        out.append(atom)
    return out


def encode_basis(atoms, weights, fraction):
    encoder = ArithEncoder()
    code_basis(encoder, weights, fraction, atoms)
    return encoder.finish()


def decode_basis(data, weights, fraction):
    decoder = ArithDecoder(data)
    atoms = code_basis(decoder, weights, fraction)
    decoder.check_end("RR5 basis")
    return atoms


# ------------------------------------------------------------------------------------------------ carrier stream
# Layouts n5+ code the carrier fields with ONE mixer and one arithmetic stream, in this order: the 142-byte carrier
# prefix, then the basis, then (from n6 on) the coefficients.  The prefix is #153's: 6 bytes of bit counts, 24 fp32
# scales (12 basis, 12 coefficient), 40 bytes of packed coefficient metadata.
CARRIER_PREFIX = 142


def _code_prefix(cm, prefix=None):
    out = bytearray()
    previous = 0
    for position in range(CARRIER_PREFIX):
        value = prefix[position] if prefix is not None else 0
        if position < 6:
            field, half = 0, 0                                               # bit counts
        elif position < 102:
            field, half = 1 + (position - 6) % 4, (position - 6) // 48       # byte of an fp32; which group of 12 scales
        else:
            field, half = 5, 0                                               # packed metadata
        node = 1
        for j in range(7, -1, -1):
            contexts = (("p", field, node), ("p", field, previous, node), ("p", field, half, node))
            node = node * 2 + cm.code_bit(contexts, ("p", j), (value >> j) & 1)
        previous = node & 255
        out.append(previous)
    return bytes(out)


# Layouts n6+ append the 600 x 12 carrier coefficient symbols to the carrier stream.  They are the unsigned
# zigzag codes of the DX2 payload (2v for a residual v >= 0, -2v-1 for v < 0, below 4096), frame by frame.  The
# binarisation is DX2's own: a Rice code with a fixed k per dimension, i.e. a unary quotient and k remainder bits.
# (DX2 = vendor/external/pr141/runtime/dx2_cabac_coefficients.py, first seen in PR #140, vendored unchanged since.)
FRAMES, DIMENSIONS = 600, 12
MEAN_RATE = 26                 # running mean |v| per dimension, in 1/256 units: mean += (|v| - mean) * 26/256


def _bucket(magnitude):
    """0 for 0, else the bit length of the magnitude, capped at 9."""
    return min(magnitude.bit_length(), 9)


def _sign(value):
    return (value > 0) - (value < 0)


def _code_coefficients(cm, ks, symbols=None):
    values = [[0] * DIMENSIONS for _ in range(FRAMES)]       # signed residuals coded so far
    means = [8 << 8] * DIMENSIONS
    previous_frame_mean = 0
    out = []
    for t in range(FRAMES):
        frame_sum = 0
        row = []
        for d in range(DIMENSIONS):
            symbol = symbols[t][d] if symbols is not None else 0
            above = values[t - 1][d] if t else 0                      # same dimension, previous frame
            left = _sign(values[t][d - 1]) if d else 2
            left2 = _sign(values[t][d - 2]) if d > 1 else 2
            mean = _bucket(means[d] >> 8)
            base = ((d,), (d, mean), (d, _bucket(abs(above))), (mean, _bucket(frame_sum // d) if d else 10),
                    (mean, _bucket(previous_frame_mean)), (d, mean, _bucket(abs(above))), (mean,),
                    (d, _sign(above), left), (d, left, left2))
            k = ks[d]
            quotient = 0
            more = 1 if (symbol >> k) > quotient else 0
            while cm.code_bit(tuple(("q", min(quotient, 8)) + c for c in base), "q", more):
                quotient += 1
                more = 1 if (symbol >> k) > quotient else 0
                if (quotient << k) >= 4096:
                    raise ValueError("carrier coefficient out of range")
            remainder = 0
            for j in range(k - 1, -1, -1):
                tag = ("r", quotient if j == k - 1 else -1, j)
                remainder = remainder * 2 + cm.code_bit(tuple(tag + c for c in base), "r", (symbol >> j) & 1)
            symbol = (quotient << k) | remainder
            row.append(symbol)
            value = symbol // 2 if symbol % 2 == 0 else -(symbol // 2) - 1
            values[t][d] = value
            means[d] += (((abs(value) << 8) - means[d]) * MEAN_RATE) >> 8
            frame_sum += abs(value)
        previous_frame_mean = frame_sum // DIMENSIONS
        out.append(row)
    return out


class CarrierCoder:
    """The carrier stream.  CarrierCoder() encodes: call prefix(prefix), basis(weights, fraction, atoms) and, from
    layout n6 on, coefficients(ks, symbols), then finish() for the bytes.  CarrierCoder(data) decodes: the same
    calls without the values return them; then check_end()."""

    def __init__(self, data=None):
        self.coder = ArithEncoder() if data is None else ArithDecoder(data)
        self.cm = ContextMixer(self.coder, BASIS_LR, BASIS_WEIGHT)

    def prefix(self, prefix=None):
        return _code_prefix(self.cm, prefix)

    def basis(self, weights, fraction, atoms=None):
        return _code_basis(self.cm, weights, fraction, atoms)

    def coefficients(self, ks, symbols=None):
        return _code_coefficients(self.cm, ks, symbols)

    def finish(self):
        return self.coder.finish()

    def check_end(self):
        self.coder.check_end("carrier stream")
