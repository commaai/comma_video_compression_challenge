# SPDX-License-Identifier: MIT
"""Context-modeled entropy coder for the #101/#110 payload sections.

Replaces the off-the-shelf coders of the current #1 submission payload with
constriction-based adaptive/parametric range coding:

  decoder weights : one range stream; per-tensor adaptive 256-ary models with
                    geometric-primed prior counts. Prior decay rho, prior
                    strength M, floor eps, and adaptation increment are chosen
                    per model by exact simulated code length and transmitted
                    in a 2-byte/model header. fp16 scale low bytes are stored
                    raw; the (highly redundant) high bytes are entropy-coded.
  latents         : per-dim causal prediction (AR(1) on own quantized deltas,
                    optional own-lag-2 and up to 4 already-decoded dims at the
                    same time step; integer-quantized LS coefficients) with
                    static discrete-Gaussian residual models (u16 parameter
                    per dim) or, where the encoder measures a win, an
                    adaptive Gaussian-primed model. fp16 min/scale high bytes
                    entropy-coded like the decoder scales.
  selector        : adaptive 16-ary model over the FEC6 mode indices
                    (recovers the exact 249-byte FEC6 wrapper on decode).

All decode-side model construction uses only IEEE-exact float64 operations
(multiply / add / divide), so encoder and decoder build bit-identical
probability tables on any IEEE-754 platform. Decode-side deps: numpy +
constriction (harness base deps) (+ brotli/lzma stdlib for passthrough).

Container (member produced by `pack_container`):
  u8 version | u8 coder-id bitmap (2 bits/section: decoder, latent, selector)
  | u24 len(decoder section) | u24 len(latent section) | selector section.
Coder id 0 = passthrough (current bytes verbatim), 1 = this coder.
"""
from __future__ import annotations

import math
import struct

import numpy as np

try:
    import constriction
except ImportError:  # schema constants remain importable without it
    constriction = None

VERSION = 1
CODER_PASSTHROUGH = 0
CODER_CTX = 1

# ---------------------------------------------------------------------------
# fixed schema of the #101 decoder payload (verified against
# submissions/hnerv_fec6_fixed_huffman_k16/src/codec.py and the archive)
# ---------------------------------------------------------------------------
# (tensor_idx, numel) per storage position; streams split per STREAM_ENDS.
TENSOR_SCHEMA = (
    (14, 972), (22, 1458), (7, 108), (6, 34992), (19, 18), (10, 12960),
    (25, 3), (4, 46656), (20, 1458), (9, 80), (12, 11664), (15, 27),
    (5, 144), (11, 72), (18, 360), (1, 1728), (21, 9), (3, 144), (27, 3),
    (13, 72), (2, 46656), (26, 486), (24, 486), (17, 20), (16, 540),
    (23, 18), (8, 19440), (0, 48384),
)
STREAM_ENDS = (1, 2, 22, 23, 26, 27, 28)
N_STREAMS = len(STREAM_ENDS)
N_TENSORS = len(TENSOR_SCHEMA)
# byte maps that need a magnitude-ordering remap for the geometric prior
TWOS_TENSORS = frozenset({20})
OFF_TENSORS = frozenset({27})
# tensors sharing one adaptive model (chosen by exact cost on the encoder;
# fixed here so the decoder needs no signalling)
SHARED_MODEL_TENSORS = frozenset({7, 5, 1, 3})

M_GRID = (4.0, 16.0, 64.0, 256.0, 1024.0)          # 3 bits
INC_GRID = (1.0, 1.5, 2.0, 3.0)                    # 2 bits
EPS_GRID = (0.01, 0.05, 0.15, 0.4)                 # 2 bits

N_PAIRS = 600
LATENT_DIM = 28
LATENT_HDR = 112  # fp16 mins + scales
LATENT_ADAPT_M = 480.0
SELECTOR_MAGIC = b"FEC6"
FEC6_CODE_BITS = (
    "00", "1100", "01", "111010", "11010", "111011", "111100", "100",
    "111101", "11011", "1111110", "111110", "11111110", "101", "11100",
    "11111111",
)
LAG2_SRC = 255  # spec marker: feature is own lag-2 instead of a cross dim

# discrete-Gaussian parameter grid: Q_TABLE[k] = round(65535*exp(-1/(2*s^2)))
# for s = 0.3*(80/0.3)^(k/255) -- precomputed so the decoder never calls exp().
Q_TABLE = (
    253, 321, 404, 502, 619, 756, 915, 1099, 1309, 1549,
    1818, 2120, 2456, 2827, 3235, 3681, 4164, 4686, 5247, 5847,
    6485, 7161, 7874, 8623, 9406, 10222, 11070, 11947, 12851, 13780,
    14733, 15706, 16698, 17706, 18728, 19761, 20803, 21852, 22905, 23961,
    25017, 26071, 27122, 28167, 29206, 30235, 31255, 32263, 33258, 34239,
    35205, 36155, 37089, 38005, 38903, 39783, 40643, 41484, 42305, 43106,
    43888, 44648, 45389, 46109, 46809, 47489, 48150, 48790, 49411, 50013,
    50596, 51160, 51706, 52234, 52744, 53238, 53714, 54174, 54618, 55046,
    55459, 55858, 56241, 56611, 56968, 57311, 57641, 57959, 58265, 58560,
    58843, 59115, 59377, 59629, 59871, 60103, 60326, 60541, 60747, 60945,
    61135, 61317, 61492, 61660, 61822, 61976, 62125, 62267, 62404, 62535,
    62661, 62781, 62897, 63008, 63114, 63216, 63314, 63407, 63497, 63583,
    63666, 63745, 63820, 63893, 63963, 64029, 64093, 64154, 64213, 64269,
    64323, 64374, 64424, 64471, 64516, 64559, 64601, 64641, 64679, 64715,
    64750, 64784, 64816, 64846, 64876, 64904, 64931, 64957, 64981, 65005,
    65028, 65049, 65070, 65090, 65109, 65127, 65144, 65161, 65177, 65192,
    65207, 65221, 65235, 65247, 65260, 65271, 65283, 65294, 65304, 65314,
    65323, 65332, 65341, 65349, 65357, 65365, 65372, 65379, 65386, 65392,
    65398, 65404, 65410, 65415, 65420, 65425, 65430, 65434, 65439, 65443,
    65447, 65451, 65454, 65458, 65461, 65464, 65467, 65470, 65473, 65475,
    65478, 65480, 65483, 65485, 65487, 65489, 65491, 65493, 65495, 65497,
    65498, 65500, 65501, 65503, 65504, 65505, 65507, 65508, 65509, 65510,
    65511, 65512, 65513, 65514, 65515, 65516, 65517, 65518, 65518, 65519,
    65520, 65520, 65521, 65522, 65522, 65523, 65523, 65524, 65524, 65525,
    65525, 65526, 65526, 65526, 65527, 65527, 65527, 65528, 65528, 65528,
    65529, 65529, 65529, 65529, 65530, 65530,
)


# ---------------------------------------------------------------------------
# deterministic model construction helpers
# ---------------------------------------------------------------------------
def _geometric_pmf(rho_q: int) -> np.ndarray:
    """pmf[k] = (1-r) * r^k built by repeated IEEE multiplication."""
    r = rho_q / 255.0
    out = np.empty(256, np.float64)
    v = 1.0 - r
    for k in range(256):
        out[k] = v
        v = v * r
    return out


def _remap_table(tensor_idx: int) -> np.ndarray:
    """Map stored byte -> magnitude-ordered index used by the prior."""
    v = np.arange(256, dtype=np.int64)
    if tensor_idx in TWOS_TENSORS:
        x = np.where(v >= 128, v - 256, v)
    elif tensor_idx in OFF_TENSORS:
        x = v - 128
    else:
        return v
    return np.where(x >= 0, 2 * x, -2 * x - 1)


def _prior_counts(rho_q: int, m_idx: int, eps_idx: int,
                  remap: np.ndarray) -> np.ndarray:
    geo = _geometric_pmf(rho_q)
    return EPS_GRID[eps_idx] + M_GRID[m_idx] * geo[remap]


def _gauss_pmf_from_q(q_u16: int) -> np.ndarray:
    """Discrete Gaussian pmf[k] proportional to q^(k^2), k in [-128, 127].

    Built with exact float64 multiplications only (binary exponentiation), so
    it is bit-identical across IEEE platforms. Unnormalized (constriction
    normalizes internally); underflow to 0.0 is fine (constriction floors)."""
    q = q_u16 / 65535.0
    out = np.empty(256, np.float64)
    for i in range(256):
        k = i - 128
        e = k * k
        acc = 1.0
        base = q
        while e:
            if e & 1:
                acc *= base
            base *= base
            e >>= 1
        out[i] = acc
    return out


def _pmf_bits(pmf: np.ndarray, symbols: np.ndarray) -> float:
    p = pmf / pmf.sum()
    p = np.maximum(p, 1e-300)
    return float(-np.log2(p[symbols]).sum())


def _cumcount(a: np.ndarray) -> np.ndarray:
    """out[t] = number of previous occurrences of a[t] in a[:t]."""
    order = np.argsort(a, kind="stable")
    sa = a[order]
    starts = np.flatnonzero(np.concatenate([[True], sa[1:] != sa[:-1]]))
    sizes = np.diff(np.concatenate([starts, [len(sa)]]))
    grp = np.repeat(np.arange(len(starts)), sizes)
    cc = np.arange(len(sa)) - starts[grp]
    out = np.empty(len(a), np.int64)
    out[order] = cc
    return out


def _sim_adaptive_bits(symbols: np.ndarray, cc: np.ndarray,
                       prior: np.ndarray, inc: float) -> float:
    """Exact code length of adaptive AC: P_t(s) = (a_s + inc*c) / (A + inc*t)."""
    A = float(prior.sum())
    n = len(symbols)
    num = prior[symbols] + inc * cc
    den = A + inc * np.arange(n, dtype=np.float64)
    return float(np.log2(den).sum() - np.log2(num).sum())


def _adaptive_dm_bits(symbols: np.ndarray, prior: np.ndarray) -> float:
    return _sim_adaptive_bits(symbols, _cumcount(symbols), prior, 1.0)


# ---------------------------------------------------------------------------
# fp16 high-byte coding (scales / mins are tightly clustered in exponent)
# ---------------------------------------------------------------------------
def _encode_hi_bytes(enc, hi: np.ndarray) -> bytes:
    """Encode high bytes into `enc`; returns 2-byte (base, width) header."""
    base = int(hi.min())
    width = int(hi.max()) - base + 1
    counts = np.full(width, 0.5, np.float64)
    Cat = constriction.stream.model.Categorical
    for v in (hi.astype(np.int64) - base).tolist():
        enc.encode(v, Cat(counts, lazy=True))
        counts[v] += 1.0
    return struct.pack("<BB", base, width)


def _decode_hi_bytes(dec, base: int, width: int, n: int) -> np.ndarray:
    counts = np.full(width, 0.5, np.float64)
    Cat = constriction.stream.model.Categorical
    out = np.empty(n, np.uint8)
    for i in range(n):
        v = dec.decode(Cat(counts, lazy=True))
        out[i] = base + v
        counts[v] += 1.0
    return out


# ---------------------------------------------------------------------------
# section 1: decoder weight streams
# ---------------------------------------------------------------------------
def _decoder_model_plan():
    """Per-storage-pos model key + model keys in first-use order."""
    model_of, keys = [], {}
    for tidx, _ in TENSOR_SCHEMA:
        key = "shared" if tidx in SHARED_MODEL_TENSORS else f"t{tidx}"
        model_of.append(key)
        if key not in keys:
            keys[key] = len(keys)
    return model_of, keys


def _split_streams_to_tensors(raws: list[bytes]):
    """raws (7 streams) -> (per-pos weight byte arrays, per-pos 2B scales)."""
    weights, scales = [], []
    start = 0
    for si, end in enumerate(STREAM_ENDS):
        data = np.frombuffer(raws[si], np.uint8)
        pos = 0
        for p in range(start, end):
            _tidx, numel = TENSOR_SCHEMA[p]
            weights.append(data[pos:pos + numel].copy())
            scales.append(bytes(data[pos + numel:pos + numel + 2]))
            pos += numel + 2
        if pos != len(data):
            raise ValueError(f"stream {si} length mismatch")
        start = end
    return weights, scales


def _tensors_to_streams(weights, scales) -> list[bytes]:
    out, start = [], 0
    for end in STREAM_ENDS:
        parts = []
        for p in range(start, end):
            parts.append(weights[p].astype(np.uint8).tobytes())
            parts.append(scales[p])
        out.append(b"".join(parts))
        start = end
    return out


def _search_model_params(m: np.ndarray) -> tuple[int, int, int, int]:
    """Pick (rho_q, m_idx, inc_idx, eps_idx) minimizing simulated bits."""
    cc = _cumcount(m)
    mu = max(float(m.mean()), 0.05)
    rho0 = int(np.clip(round(255.0 * mu / (1.0 + mu)), 1, 254))
    best = None
    geos = {}
    for stage, offsets in ((0, range(-12, 13, 4)), (1, range(-3, 4, 1))):
        center = rho0 if stage == 0 else best[1]
        for dr in offsets:
            rho_q = int(np.clip(center + dr, 1, 254))
            if rho_q not in geos:
                geos[rho_q] = _geometric_pmf(rho_q)
            geo = geos[rho_q]
            for m_idx, M in enumerate(M_GRID):
                for eps_idx, eps in enumerate(EPS_GRID):
                    prior = eps + M * geo
                    for inc_idx, inc in enumerate(INC_GRID):
                        bits = _sim_adaptive_bits(m, cc, prior, inc)
                        cand = (bits, rho_q, m_idx, inc_idx, eps_idx)
                        if best is None or cand[0] < best[0]:
                            best = cand
    return best[1], best[2], best[3], best[4]


def encode_decoder_section(raws: list[bytes]) -> bytes:
    weights, scales = _split_streams_to_tensors(raws)
    model_of, keys = _decoder_model_plan()

    model_syms = {k: [] for k in keys}
    for pos, w in enumerate(weights):
        tidx, _ = TENSOR_SCHEMA[pos]
        model_syms[model_of[pos]].append(_remap_table(tidx)[w])
    params = {k: _search_model_params(np.concatenate(model_syms[k]))
              for k in keys}

    scale_bytes = b"".join(scales)
    lo = np.frombuffer(scale_bytes, np.uint8)[0::2]
    hi = np.frombuffer(scale_bytes, np.uint8)[1::2]

    enc = constriction.stream.queue.RangeEncoder()
    hi_hdr = _encode_hi_bytes(enc, hi)

    Cat = constriction.stream.model.Categorical
    model_counts, model_inc = {}, {}
    for pos, w in enumerate(weights):
        tidx, _ = TENSOR_SCHEMA[pos]
        key = model_of[pos]
        if key not in model_counts:
            rho_q, m_idx, inc_idx, eps_idx = params[key]
            model_counts[key] = _prior_counts(rho_q, m_idx, eps_idx,
                                              _remap_table(tidx))
            model_inc[key] = INC_GRID[inc_idx]
        counts, inc = model_counts[key], model_inc[key]
        for v in w.tolist():
            enc.encode(v, Cat(counts, lazy=True))
            counts[v] += inc
    body = enc.get_compressed().astype("<u4").tobytes()

    hdr = [lo.tobytes(), hi_hdr]
    hdr.append(bytes(params[k][0] for k in keys))                     # rho
    hdr.append(bytes((params[k][1] | (params[k][2] << 3)
                      | (params[k][3] << 5)) for k in keys))          # M|inc|eps
    return b"".join(hdr) + body


def decode_decoder_section(blob: bytes) -> list[bytes]:
    model_of, keys = _decoder_model_plan()
    n_models = len(keys)
    p = 0
    lo = np.frombuffer(blob[p:p + N_TENSORS], np.uint8)
    p += N_TENSORS
    hi_base, hi_width = struct.unpack_from("<BB", blob, p)
    p += 2
    rho_bytes = blob[p:p + n_models]
    p += n_models
    packed = blob[p:p + n_models]
    p += n_models
    params = {}
    for i, k in enumerate(keys):
        b = packed[i]
        params[k] = (rho_bytes[i], b & 7, (b >> 3) & 3, (b >> 5) & 3)

    words = np.frombuffer(blob[p:], dtype="<u4").astype(np.uint32)
    dec = constriction.stream.queue.RangeDecoder(words)
    hi = _decode_hi_bytes(dec, hi_base, hi_width, N_TENSORS)
    scale_arr = np.empty(2 * N_TENSORS, np.uint8)
    scale_arr[0::2] = lo
    scale_arr[1::2] = hi
    scales = [scale_arr[2 * i:2 * i + 2].tobytes() for i in range(N_TENSORS)]

    Cat = constriction.stream.model.Categorical
    model_counts, model_inc = {}, {}
    weights = []
    for pos, (tidx, numel) in enumerate(TENSOR_SCHEMA):
        key = model_of[pos]
        if key not in model_counts:
            rho_q, m_idx, inc_idx, eps_idx = params[key]
            model_counts[key] = _prior_counts(rho_q, m_idx, eps_idx,
                                              _remap_table(tidx))
            model_inc[key] = INC_GRID[inc_idx]
        counts, inc = model_counts[key], model_inc[key]
        out = np.empty(numel, np.uint8)
        for i in range(numel):
            v = dec.decode(Cat(counts, lazy=True))
            out[i] = v
            counts[v] += inc
        weights.append(out)
    return _tensors_to_streams(weights, scales)


# ---------------------------------------------------------------------------
# section 2: latents
# ---------------------------------------------------------------------------
def _latent_signed_deltas(codes: np.ndarray) -> np.ndarray:
    d = (codes[:, 1:].astype(np.int64) - 128) & 255
    return np.where(d >= 128, d - 256, d)


def _fit_q_idx(res: np.ndarray) -> tuple[int, float]:
    """Q_TABLE index minimizing exact coded bits for the residuals."""
    sig = max(float(res.std()), 0.35)
    k0 = int(np.clip(round(255.0 * math.log(sig / 0.3)
                           / math.log(80.0 / 0.3)), 0, 255))
    syms = (res + 128).astype(np.int64)
    best = None
    for dk in range(-14, 15, 2):
        k = int(np.clip(k0 + dk, 0, 255))
        bits = _pmf_bits(_gauss_pmf_from_q(Q_TABLE[k]), syms)
        if best is None or bits < best[1]:
            best = (k, bits)
    k_center = best[0]
    for dk in (-1, 1):
        k = int(np.clip(k_center + dk, 0, 255))
        bits = _pmf_bits(_gauss_pmf_from_q(Q_TABLE[k]), syms)
        if bits < best[1]:
            best = (k, bits)
    return best


def encode_latent_section(latent_raw: bytes) -> bytes:
    lat = np.frombuffer(latent_raw, np.uint8)
    if len(lat) != LATENT_HDR + LATENT_DIM * N_PAIRS:
        raise ValueError("unexpected latent payload length")
    hdr16 = lat[:LATENT_HDR]
    lo = hdr16[0::2]
    hi = hdr16[1::2]
    codes = lat[LATENT_HDR:].reshape(LATENT_DIM, N_PAIRS)
    first_col = bytes(codes[:, 0])
    sd = _latent_signed_deltas(codes)  # (28, 599)
    T = N_PAIRS - 1

    C = np.corrcoef(sd.astype(np.float64))
    own_f = sd.astype(np.float64)
    specs, all_syms = [], []
    for i in range(LATENT_DIM):
        ar1 = np.concatenate([[0.0], own_f[i, :-1]])
        lag2 = np.concatenate([[0.0, 0.0], own_f[i, :-2]])
        lag2_corr = abs(np.corrcoef(sd[i, 2:], sd[i, :-2])[0, 1])
        cands = sorted(range(i), key=lambda k: -abs(C[i, k]))
        cands = [(k, abs(C[i, k])) for k in cands if abs(C[i, k]) > 0.10][:4]
        cands.append((LAG2_SRC, lag2_corr))
        cands.sort(key=lambda kc: -kc[1])
        best = None
        for ncand in range(len(cands) + 1):
            feats = [ar1]
            for k, _ in cands[:ncand]:
                feats.append(lag2 if k == LAG2_SRC else own_f[k])
            X = np.stack(feats, 1)
            coef, *_ = np.linalg.lstsq(X, own_f[i], rcond=None)
            cq = np.clip(np.round(coef * 64.0), -127, 127).astype(np.int64)
            num = (X.astype(np.int64) * cq).sum(axis=1)
            pred = (num + 32) >> 6
            r = ((sd[i] - pred + 128) & 255) - 128
            q_idx, bits = _fit_q_idx(r)
            cost = bits + 8.0 * (2 * ncand)
            if best is None or cost < best[0]:
                best = (cost, cq, cands[:ncand], q_idx, r)
        _, cq, used, q_idx, r = best
        syms = (r + 128).astype(np.int64)
        # static vs Gaussian-primed adaptive residual model
        pmf = _gauss_pmf_from_q(Q_TABLE[q_idx])
        static_bits = _pmf_bits(pmf, syms)
        prior = 1e-4 + LATENT_ADAPT_M * (pmf / pmf.sum())
        adapt_bits = _adaptive_dm_bits(syms, prior)
        adaptive = bool(adapt_bits + 2.0 < static_bits)
        specs.append((int(cq[0]),
                      [(k, int(c)) for (k, _), c in zip(used, cq[1:])],
                      q_idx, adaptive))
        all_syms.append(syms)

    out = [lo.tobytes(), first_col]
    for ar1_q, cross, q_idx, adaptive in specs:
        out.append(struct.pack("<Bb", q_idx, ar1_q))
    nib = bytearray(LATENT_DIM // 2)
    for i, (_, cross, _, adaptive) in enumerate(specs):
        v = len(cross) | (8 if adaptive else 0)
        nib[i // 2] |= v << (4 * (i % 2))
    out.append(bytes(nib))
    for _, cross, _, _ in specs:
        for k, c in cross:
            out.append(struct.pack("<Bb", k, c))

    enc = constriction.stream.queue.RangeEncoder()
    # mins-hi and scales-hi are two distinct exponent clusters: separate models
    hi_hdr = _encode_hi_bytes(enc, hi[:LATENT_DIM]) \
        + _encode_hi_bytes(enc, hi[LATENT_DIM:])
    out.insert(1, hi_hdr)
    Cat = constriction.stream.model.Categorical
    for i in range(LATENT_DIM):
        q_idx, adaptive = specs[i][2], specs[i][3]
        pmf = _gauss_pmf_from_q(Q_TABLE[q_idx])
        if adaptive:
            counts = 1e-4 + LATENT_ADAPT_M * (pmf / pmf.sum())
            for v in all_syms[i].tolist():
                enc.encode(v, Cat(counts, lazy=True))
                counts[v] += 1.0
        else:
            enc.encode(all_syms[i].astype(np.int32), Cat(pmf, perfect=False))
    out.append(enc.get_compressed().astype("<u4").tobytes())
    return b"".join(out)


def decode_latent_section(blob: bytes) -> bytes:
    p = 0
    lo = np.frombuffer(blob[p:p + LATENT_HDR // 2], np.uint8)
    p += LATENT_HDR // 2
    hb0, hw0, hb1, hw1 = struct.unpack_from("<BBBB", blob, p)
    p += 4
    first_col = np.frombuffer(blob[p:p + LATENT_DIM], np.uint8)
    p += LATENT_DIM
    qa = []
    for _ in range(LATENT_DIM):
        q_idx, ar1 = struct.unpack_from("<Bb", blob, p)
        p += 2
        qa.append((q_idx, ar1))
    nib = blob[p:p + LATENT_DIM // 2]
    p += LATENT_DIM // 2
    specs = []
    for i in range(LATENT_DIM):
        v = (nib[i // 2] >> (4 * (i % 2))) & 0xF
        specs.append([qa[i][1], v & 7, qa[i][0], bool(v & 8)])
    for i in range(LATENT_DIM):
        cross = []
        for _ in range(specs[i][1]):
            k, c = struct.unpack_from("<Bb", blob, p)
            p += 2
            cross.append((k, c))
        specs[i][1] = cross

    words = np.frombuffer(blob[p:], dtype="<u4").astype(np.uint32)
    dec = constriction.stream.queue.RangeDecoder(words)
    hi = np.concatenate([_decode_hi_bytes(dec, hb0, hw0, LATENT_DIM),
                         _decode_hi_bytes(dec, hb1, hw1, LATENT_DIM)])
    hdr16 = np.empty(LATENT_HDR, np.uint8)
    hdr16[0::2] = lo
    hdr16[1::2] = hi

    Cat = constriction.stream.model.Categorical
    T = N_PAIRS - 1
    sd = np.zeros((LATENT_DIM, T), np.int64)
    for i in range(LATENT_DIM):
        ar1, cross, q_idx, adaptive = specs[i]
        pmf = _gauss_pmf_from_q(Q_TABLE[q_idx])
        if adaptive:
            counts = 1e-4 + LATENT_ADAPT_M * (pmf / pmf.sum())
            syms = np.empty(T, np.int64)
            for t in range(T):
                v = dec.decode(Cat(counts, lazy=True))
                syms[t] = v
                counts[v] += 1.0
        else:
            syms = np.asarray(dec.decode(Cat(pmf, perfect=False), T),
                              dtype=np.int64)
        r = syms - 128
        cross_term = np.zeros(T, np.int64)
        lag2_coef = 0
        for k, c in cross:
            if k == LAG2_SRC:
                lag2_coef = c
            else:
                cross_term += c * sd[k]
        row = sd[i]
        prev = prev2 = 0
        for t in range(T):
            num = ar1 * prev + lag2_coef * prev2 + int(cross_term[t])
            pred = (num + 32) >> 6
            v = ((r[t] + pred + 128) & 255) - 128
            row[t] = v
            prev2 = prev
            prev = v
    codes = np.empty((LATENT_DIM, N_PAIRS), np.uint8)
    codes[:, 0] = first_col
    codes[:, 1:] = ((sd + 128) & 255).astype(np.uint8)
    return hdr16.tobytes() + codes.tobytes()


# ---------------------------------------------------------------------------
# section 3: FEC6 selector
# ---------------------------------------------------------------------------
def _selector_codes_from_payload(payload: bytes) -> np.ndarray:
    if payload[:4] != SELECTOR_MAGIC:
        raise ValueError("not a FEC6 selector payload")
    n_pairs = struct.unpack_from("<H", payload, 4)[0]
    decode = {b: i for i, b in enumerate(FEC6_CODE_BITS)}
    bits = np.unpackbits(np.frombuffer(payload[6:], np.uint8))
    codes, prefix = [], ""
    for bit in bits:
        prefix += "1" if bit else "0"
        if prefix in decode:
            codes.append(decode[prefix])
            prefix = ""
            if len(codes) == n_pairs:
                break
    if len(codes) != n_pairs:
        raise ValueError("truncated FEC6 bitstream")
    return np.array(codes, np.int64)


def _selector_payload_from_codes(codes: np.ndarray) -> bytes:
    bitstr = "".join(FEC6_CODE_BITS[c] for c in codes.tolist())
    bitstr += "0" * ((-len(bitstr)) % 8)
    body = int(bitstr, 2).to_bytes(len(bitstr) // 8, "big") if bitstr else b""
    return SELECTOR_MAGIC + struct.pack("<H", len(codes)) + body


def encode_selector_section(payload: bytes) -> bytes:
    codes = _selector_codes_from_payload(payload)
    enc = constriction.stream.queue.RangeEncoder()
    Cat = constriction.stream.model.Categorical
    counts = np.full(16, 0.6, np.float64)
    for c in codes.tolist():
        enc.encode(c, Cat(counts, lazy=True))
        counts[c] += 1.0
    return enc.get_compressed().astype("<u4").tobytes()


def decode_selector_section(blob: bytes) -> bytes:
    words = np.frombuffer(blob, dtype="<u4").astype(np.uint32)
    dec = constriction.stream.queue.RangeDecoder(words)
    Cat = constriction.stream.model.Categorical
    counts = np.full(16, 0.6, np.float64)
    codes = np.empty(N_PAIRS, np.int64)
    for i in range(N_PAIRS):
        c = dec.decode(Cat(counts, lazy=True))
        codes[i] = c
        counts[c] += 1.0
    return _selector_payload_from_codes(codes)


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------
def pack_container(decoder_sec: bytes, latent_sec: bytes, selector_sec: bytes,
                   coder_ids: tuple[int, int, int]) -> bytes:
    head = bytes([(VERSION << 4) | coder_ids[0] | (coder_ids[1] << 1)
                  | (coder_ids[2] << 2)])
    head += len(decoder_sec).to_bytes(3, "little")
    head += len(latent_sec).to_bytes(3, "little")
    return head + decoder_sec + latent_sec + selector_sec


def unpack_container(blob: bytes):
    """-> (decoder_streams: list[bytes], latent_raw: bytes, selector: bytes)."""
    if blob[0] >> 4 != VERSION:
        raise ValueError("bad container version")
    ids = (blob[0] & 1, (blob[0] >> 1) & 1, (blob[0] >> 2) & 1)
    ld = int.from_bytes(blob[1:4], "little")
    ll = int.from_bytes(blob[4:7], "little")
    p = 7
    dec_sec = blob[p:p + ld]
    p += ld
    lat_sec = blob[p:p + ll]
    p += ll
    sel_sec = blob[p:]

    if ids[0] == CODER_CTX:
        streams = decode_decoder_section(dec_sec)
    else:
        streams = _split_legacy_brotli(dec_sec)
    if ids[1] == CODER_CTX:
        latent_raw = decode_latent_section(lat_sec)
    else:
        import lzma
        latent_raw = lzma.decompress(
            lat_sec, format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 4096,
                      "lc": 3, "lp": 0, "pb": 0}])
    selector = decode_selector_section(sel_sec) if ids[2] == CODER_CTX else sel_sec
    return streams, latent_raw, selector


def _split_legacy_brotli(data: bytes) -> list[bytes]:
    import brotli
    out, pos = [], 0
    for _ in range(N_STREAMS):
        dec = brotli.Decompressor()
        chunks = []
        while pos < len(data) and not dec.is_finished():
            chunks.append(dec.process(data[pos:pos + 1]))
            pos += 1
        if not dec.is_finished():
            raise ValueError("truncated legacy decoder payload")
        out.append(b"".join(chunks))
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def summary_table(rows: list[tuple[str, int, int]]) -> str:
    """rows: (section, current_bytes, new_bytes) -> formatted table."""
    lines = [f"{'section':<12} {'current':>10} {'ctx-coder':>10} {'delta':>8}"]
    tc = tn = 0
    for name, cur, new in rows:
        tc += cur
        tn += new
        lines.append(f"{name:<12} {cur:>10,} {new:>10,} {new - cur:>+8,}")
    lines.append(f"{'TOTAL':<12} {tc:>10,} {tn:>10,} {tn - tc:>+8,}")
    return "\n".join(lines)
