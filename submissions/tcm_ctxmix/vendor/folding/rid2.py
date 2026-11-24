"""RID2 = RID1 + residual-state packing.  SM3R mode 8 (the frozen RX1 reader only accepts SD1M/SM3R tags).  After LAY1 unpacking:
    b"SM3R" | ver u8 (=1) | mode u8 (=8) | t_bits u8 | wc_bits u8 (0 = fp16) | v_bits u8 (0 = vectors stay fp16 in the sub-blob; 16 = per-vector bit depths follow the header, one byte each) | fe_mode u8 | fe_inc u8 | film_r u8 (0 = off) | head_r u8 (0 = off) | fold u8 (bit0: dw.bias->pw.bias & norm affine->FiLM; bit1: coord_mix.bias->T) | sp u8 (0 = fp16 scales; else log-scale steps per octave: every weight/T scale is stored as int16 base + uint8 code, s = up16(2^(k/sp)))
    T scales fp16[96] | T codes int<t_bits> [96 ch x 5 tok] | Wc [fp16 96x4 | wc_bits: fp16 scale[4] + int<wc_bits> codes 96x4 col-scaled]
    [vectors: fp16 scale + int<v_bits> codes per 1-D tensor with >=16 elements, template order]
    [fe_mode != 0: frame scales fp16[FD] | u16 stream length | arithmetic-coded frame codes (600 x FD, 4-bit signed alphabet)]   (fe_mode 0: frame_embed stays inside the sub-blob as packed nibbles)
    SM3R(mixed) sub-blob of the REDUCED template.
fe_mode: 1 = per-dim adaptive categorical on absolute codes, 2 = per-dim adaptive categorical on temporal deltas, 3 = per-dim adaptive 'same-as-previous' flag + signed delta.
film_r > 0: joint FiLM basis (frame_embed [600,r] + every film.weight [*,r]); head_r > 0: head stored as 1x1 (96->r) o 3x3 (r->3), expanded exactly at decode.  The decoder rebuilds the PARENT-shaped state dict so the unmodified renderer runs.
Fail closed: bad magic/version/mode/bits, truncated or trailing bytes, non-canonical arithmetic stream, invalid sub-blob."""
import struct, sys
import numpy as np, torch
sys.path.insert(0, '.')
from semantic import sem_codec as sc
from runtime.segment_layout import pack_segments, unpack_segments
MAGIC = b"SM3R"; MODE = 8; NTOK, WIDTH, NCOORD, NFRAME = 5, 96, 4, 600
DROP = ("token_embed.weight", "coord_mix.weight"); FE = "frame_embed.weight"


class RID2Error(ValueError): pass


# ------------------------------------------------------------------------------------------- integer arithmetic coder (deterministic, adaptive counts)
TOP, HALF, QUART = 0xFFFFFFFF, 0x80000000, 0x40000000


class Enc:
    def __init__(self): self.low, self.high, self.pend, self.bits = 0, TOP, 0, []
    def _emit(self, b): self.bits.append(b); self.bits.extend([1 - b] * self.pend); self.pend = 0
    def put(self, lo, hi, tot):
        r = self.high - self.low + 1; self.high = self.low + r * hi // tot - 1; self.low = self.low + r * lo // tot
        while True:
            if self.high < HALF: self._emit(0)
            elif self.low >= HALF: self._emit(1); self.low -= HALF; self.high -= HALF
            elif self.low >= QUART and self.high < HALF + QUART: self.pend += 1; self.low -= QUART; self.high -= QUART
            else: break
            self.low <<= 1; self.high = (self.high << 1) | 1
    def finish(self):
        self.pend += 1; self._emit(0 if self.low < QUART else 1); b = self.bits + [0] * (-len(self.bits) % 8)
        return np.packbits(np.array(b, np.uint8)).tobytes()


class Dec:
    def __init__(self, data):
        self.bits = np.unpackbits(np.frombuffer(data, np.uint8)); self.pos = 0; self.low, self.high = 0, TOP; self.code = 0
        for _ in range(32): self.code = (self.code << 1) | self._bit()
    def _bit(self):
        b = int(self.bits[self.pos]) if self.pos < len(self.bits) else 0; self.pos += 1; return b
    def get(self, cum, tot):
        r = self.high - self.low + 1; v = ((self.code - self.low + 1) * tot - 1) // r; s = 0
        while cum[s + 1] <= v: s += 1
        lo, hi = cum[s], cum[s + 1]; self.high = self.low + r * hi // tot - 1; self.low = self.low + r * lo // tot
        while True:
            if self.high < HALF: pass
            elif self.low >= HALF: self.low -= HALF; self.high -= HALF; self.code -= HALF
            elif self.low >= QUART and self.high < HALF + QUART: self.low -= QUART; self.high -= QUART; self.code -= QUART
            else: break
            self.low <<= 1; self.high = (self.high << 1) | 1; self.code = (self.code << 1) | self._bit()
        return s


class Model:
    def __init__(self, n, inc): self.c = [1] * n; self.inc = inc; self.n = n
    def cum(self):
        out = [0]
        for x in self.c: out.append(out[-1] + x)
        return out
    def update(self, s):
        self.c[s] += self.inc
        if sum(self.c) > 1 << 15: self.c = [(x + 1) // 2 for x in self.c]


def _symbols(codes, mode):
    """codes int [N,D] in [-8,7] -> list over (t,d) of (model_key, symbol) in coding order (frame-major, dim-minor)."""
    N, D = codes.shape; seq = []
    for t in range(N):
        for d in range(D):
            c = int(codes[t, d]); p = int(codes[t - 1, d]) if t else 0
            if mode == 1: seq.append((("a", d), c + 8))
            elif mode == 2: seq.append((("d", d), c - p + 15))
            else:
                dl = c - p; seq.append((("s", d), 1 if dl == 0 else 0))
                if dl != 0: seq.append((("e", d), dl + 15 if dl < 0 else dl + 14))
    return seq


_ALPHA = {"a": 16, "d": 31, "s": 2, "e": 30}


def encode_frames(codes, mode, inc):
    codes = np.asarray(codes, np.int64)
    if codes.min() < -8 or codes.max() > 7: raise RID2Error("frame code out of 4-bit range")
    enc = Enc(); models = {}
    for key, s in _symbols(codes, mode):
        m = models.setdefault(key, Model(_ALPHA[key[0]], inc)); cum = m.cum(); enc.put(cum[s], cum[s + 1], cum[-1]); m.update(s)
    return enc.finish()


def decode_frames(data, mode, inc, N, D):
    dec = Dec(data); models = {}; out = np.zeros((N, D), np.int64)
    def sym(key):
        m = models.setdefault(key, Model(_ALPHA[key[0]], inc)); cum = m.cum(); s = dec.get(cum, cum[-1]); m.update(s); return s
    for t in range(N):
        for d in range(D):
            p = int(out[t - 1, d]) if t else 0
            if mode == 1: c = sym(("a", d)) - 8
            elif mode == 2: c = p + sym(("d", d)) - 15
            else:
                if sym(("s", d)) == 1: c = p
                else: e = sym(("e", d)); c = p + (e - 15 if e < 15 else e - 14)
            if not -8 <= c <= 7: raise RID2Error("frame code out of range")
            out[t, d] = c
    if encode_frames(out, mode, inc) != bytes(data): raise RID2Error("non-canonical frame stream")
    return out


# ------------------------------------------------------------------------------------------------------------ section encode / decode
def _up16(x):
    """smallest fp16 >= x (no clipping of the largest element after the fp16 rounding of the scale)."""
    s = np.float16(x)
    if float(s) < x: s = np.nextafter(s, np.float16(np.inf))
    return s


def _sk(k, sp): return _up16(float(2.0 ** (int(k) / sp)))


def _kof(s, sp): return int(np.round(sp * np.log2(max(float(s), 1e-9))))


def _snap(scales, sp, up=True):
    """fp16 scales -> nearest (or next-above when up) point of the log grid."""
    return np.array([_sk(int(np.ceil(sp * np.log2(max(float(s), 1e-9)) - 1e-9)) if up else _kof(s, sp), sp) for s in scales], "<f2")


def _pack_ks(scales, sp):
    ks = np.array([_kof(s, sp) for s in scales]); k0 = int(ks.min())
    if ks.max() - k0 > 255 or not -32767 <= k0 < 32768: return struct.pack("<h", -32768) + np.asarray(scales).astype("<f2").tobytes()      # span too wide: this tensor keeps raw fp16 scales
    return struct.pack("<h", k0) + (ks - k0).astype(np.uint8).tobytes()


def _unpack_ks(mv, off, n, sp):
    if len(mv) < off + 2: raise RID2Error("truncated scales")
    (k0,) = struct.unpack("<h", bytes(mv[off:off + 2]))
    if k0 == -32768:
        if len(mv) < off + 2 + 2 * n: raise RID2Error("truncated scales")
        return np.frombuffer(mv[off + 2:off + 2 + 2 * n], "<f2").copy(), off + 2 + 2 * n
    if len(mv) < off + 2 + n: raise RID2Error("truncated scales")
    ks = np.frombuffer(mv[off + 2:off + 2 + n], np.uint8).astype(np.int64) + k0; return np.array([_sk(k, sp) for k in ks], "<f2"), off + 2 + n


def _walk_strip(blob, tm, ks):
    names = [n for n, v in tm.items() if v.ndim >= 2]; hl = 10 + (len(names) + 1) // 2; out = bytearray(blob[:hl]); dep = {}
    for i, n in enumerate(names): dep[n] = (blob[10 + i // 2] >> (4 * (i % 2))) & 15
    pos = hl
    for n, v in tm.items():
        if v.ndim < 2: m = 2 * v.numel(); out += blob[pos:pos + m]; pos += m; continue
        if n in sc.PRUNE:
            rows = v.shape[0]; mb = (rows + 7) // 8; mk = blob[pos:pos + mb]; out += mk; pos += mb; k = int(np.unpackbits(np.frombuffer(mk, np.uint8), bitorder="little")[:rows].sum()); cnt = k * (v.numel() // rows); ns = k
        else: cnt = v.numel(); ns = sc.sch._scale_count(n, v)
        ks.append((n, np.frombuffer(blob[pos:pos + 2 * ns], "<f2").copy())); pos += 2 * ns; cb = (cnt * dep[n] + 7) // 8; out += blob[pos:pos + cb]; pos += cb
    if pos != len(blob): raise RID2Error("sub-blob length mismatch")
    return bytes(out)


def unstrip_scales(stripped, tm, scale_lists):
    names = [n for n, v in tm.items() if v.ndim >= 2]; hl = 10 + (len(names) + 1) // 2
    if len(stripped) < hl: raise RID2Error("sub-blob too short")
    out = bytearray(stripped[:hl]); dep = {}; pos = hl; j = 0
    for i, n in enumerate(names): dep[n] = (stripped[10 + i // 2] >> (4 * (i % 2))) & 15
    for n, v in tm.items():
        if v.ndim < 2: m = 2 * v.numel(); out += stripped[pos:pos + m]; pos += m; continue
        if n in sc.PRUNE:
            rows = v.shape[0]; mb = (rows + 7) // 8; mk = stripped[pos:pos + mb]; out += mk; pos += mb; k = int(np.unpackbits(np.frombuffer(mk, np.uint8), bitorder="little")[:rows].sum()); cnt = k * (v.numel() // rows); ns = k
        else: cnt = v.numel(); ns = sc.sch._scale_count(n, v)
        out += scale_lists(n, ns); cb = (cnt * dep[n] + 7) // 8; out += stripped[pos:pos + cb]; pos += cb
    if pos != len(stripped): raise RID2Error("stripped sub-blob length mismatch")
    return bytes(out)


def _qvec(v, bits):
    qm = (1 << (bits - 1)) - 1; sv = _up16(max(float(np.abs(v).max()) / qm, 1e-6)); cv = np.clip(np.round(v / float(np.float32(sv))), -qm, qm).astype(np.int16); return sv, cv


def shaped_template(base, film_r, head_r):
    """parent template -> template with the reparameterised tensor shapes (order preserved)."""
    out = {}
    for k, v in base.items():
        if film_r and (k == FE or (k.startswith("blocks.") and k.endswith(".film.weight"))): out[k] = torch.zeros(v.shape[0], film_r)
        elif head_r and k == "head.weight": out["head.u.weight"] = torch.zeros(head_r, WIDTH, 1, 1); out["head.v.weight"] = torch.zeros(3, head_r, 3, 3)
        else: out[k] = v
    return out


def fold_drop(fold):
    """vectors omitted from the payload when the affine fold is on (decoder fills 0 / 1).  fold=1: dw.bias (all blocks) -> pw.bias, norm.weight (all) and norm.bias (blocks 1-3) -> FiLM."""
    out = []
    if fold & 1: out += [f"blocks.{l}.dw.bias" for l in range(4)] + [f"blocks.{l}.norm.weight" for l in range(4)] + [f"blocks.{l}.norm.bias" for l in range(1, 4)]
    if fold & 2: out += ["coord_mix.bias"]
    return out


def vec_names(tm): return [k for k, v in tm.items() if v.ndim == 1 and v.numel() >= 16]


def encode(qr, t_bits=12, wc_bits=0, v_bits=12, fe_mode=0, fe_inc=1, film_r=0, head_r=0, base_qr=None, fold=0, vlist=None, sp=0):
    """qr: RendererQ (parent-shaped).  extra: dict of replacement RendererQ-style pieces for film/head (see folding.joint).  -> (LAY1 section, info)"""
    if not 2 <= t_bits <= 15 or not (wc_bits == 0 or 2 <= wc_bits <= 15) or not (v_bits in (0, 16) or 2 <= v_bits <= 15) or fe_mode not in (0, 1, 2, 3) or not 1 <= fe_inc <= 16 or sp not in (0, 8, 16, 32, 64): raise RID2Error("unsupported format")
    sd = (base_qr or qr).state_dict(); E = sd["token_embed.weight"].double(); W = sd["coord_mix.weight"][:, :, 0, 0].double(); T = E @ W[:, :WIDTH].T
    if fold & 2: T = T + torch.from_numpy(qr.fp16["coord_mix.bias"].astype(np.float32)).double()[None]
    qm = (1 << (t_bits - 1)) - 1; scale = np.array([_up16(max(float(x), 1e-6)) for x in (T.abs().amax(0) / qm).numpy()], "<f2"); s32 = torch.from_numpy(scale.astype(np.float32)).double().clamp_min(1e-8)
    if sp: scale = _snap(scale, sp); s32 = torch.from_numpy(scale.astype(np.float32)).double().clamp_min(1e-8)
    tc = torch.clamp(torch.round(T / s32[None]), -qm, qm).numpy().astype(np.int16)
    head = MAGIC + bytes([1, MODE, t_bits, wc_bits, v_bits, fe_mode, fe_inc, film_r, head_r, fold, sp]) + (bytes(vlist) if v_bits == 16 else b"") + (scale.tobytes() if not sp else _pack_ks(scale, sp)) + sc._pack_bits(tc.T.reshape(-1), t_bits)
    Wc = W[:, WIDTH:].numpy()
    if wc_bits == 0: head += Wc.astype("<f2").tobytes()
    else:
        sw, cw = zip(*[_qvec(Wc[:, j], wc_bits) for j in range(NCOORD)]); head += np.array(sw, "<f2").tobytes() + sc._pack_bits(np.stack(cw, 1).reshape(-1), wc_bits)
    if fold and not v_bits: raise RID2Error('fold needs v_bits')
    base = shaped_template(sc.TEMPLATE, film_r, head_r); fdrop = fold_drop(fold); vn = [n for n in vec_names(base) if n not in fdrop] if v_bits else []
    if v_bits == 16 and (vlist is None or len(vlist) != len(vn) or not all(2 <= b <= 15 for b in vlist)): raise RID2Error('bad vlist')
    for j, n in enumerate(vn):
        vbj = vlist[j] if v_bits == 16 else v_bits; sv, cv = _qvec(qr.fp16[n].astype(np.float32).astype(np.float64), vbj); head += np.array([sv], "<f2").tobytes() + sc._pack_bits(cv, vbj)
    drop = set(DROP) | set(vn) | set(fdrop)
    if fe_mode:
        fc = np.asarray(qr.codes[FE]).reshape(NFRAME, -1); assert fc.shape[1] == (film_r or 8); head += np.asarray(qr.scales[FE]).astype("<f2").tobytes(); st = encode_frames(fc, fe_mode, fe_inc)
        if len(st) > 0xFFFF: raise RID2Error("frame stream too long")
        head += struct.pack("<H", len(st)) + st; drop.add(FE)
    tm0, qn0 = sc.TEMPLATE, sc.QNAMES; tm = {k: v for k, v in base.items() if k not in drop}; q2 = qr.copy()
    for n in drop: q2.depth.pop(n, None); q2.scales.pop(n, None); q2.codes.pop(n, None); q2.fp16.pop(n, None)
    if sp:
        for n in list(q2.scales):
            s_old = np.asarray(q2.scales[n]).astype(np.float32); ks = np.array([_kof(x, sp) for x in s_old])
            if not n.endswith("embed.weight"):
                nz = (np.asarray(q2.codes[n]).reshape(len(s_old), -1) != 0).any(1); ks = np.where(nz, ks, ks[nz].min() if nz.any() else 0)
            q2.scales[n] = np.array([_sk(k, sp) for k in ks], "<f2")
    try:
        sc.TEMPLATE = tm; sc.QNAMES = [n for n, v in tm.items() if v.ndim >= 2]; _, subblob, _, per = sc.encode(q2)
    finally: sc.TEMPLATE, sc.QNAMES = tm0, qn0
    if sp:
        kl = []; subblob = _walk_strip(subblob, tm, kl); pay = b"".join(_pack_ks(a, sp) for _, a in kl)
        if len(pay) > 0xFFFF: raise RID2Error("scale payload too long")
        head += struct.pack("<H", len(pay)) + pay
    blob = head + subblob; section = pack_segments([head, subblob]); assert unpack_segments(section) == blob
    return section, dict(prefix_B=len(head), sub_B=len(subblob), blob_B=len(blob), section_B=len(section))


def _unq(mv, off, cnt, bits, scale_cnt=None):
    nb = (cnt * bits + 7) // 8
    if len(mv) < off + nb: raise RID2Error("truncated payload")
    b_ = np.unpackbits(np.frombuffer(mv[off:off + nb], np.uint8), bitorder="little")[:cnt * bits].reshape(-1, bits).astype(np.int32)
    u = (b_ * (1 << np.arange(bits, dtype=np.int32))[None]).sum(1); u = np.where(u >= (1 << (bits - 1)), u - (1 << bits), u); return u, off + nb


def decode_state_dict(blob, template_sd):
    from ddm_mp2_semantic_receiver import unpack_variant_semantic_or_none
    mv = memoryview(blob)
    if len(mv) < 15 + 2 * WIDTH or bytes(mv[:4]) != MAGIC or mv[5] != MODE: raise RID2Error("bad RID2 header")
    ver, tb, wb, vb, fm, fi, fr, hr, fo, spv = mv[4], mv[6], mv[7], mv[8], mv[9], mv[10], mv[11], mv[12], mv[13], mv[14]
    if ver != 1 or not 2 <= tb <= 15 or not (wb == 0 or 2 <= wb <= 15) or not (vb in (0, 16) or 2 <= vb <= 15) or fm > 3 or not 1 <= fi <= 16 or fr > 8 or fr == 1 or hr > 96 or fo > 3 or (fo and not vb) or spv not in (0, 8, 16, 32, 64): raise RID2Error("unsupported RID2 version/format")
    tsd = shaped_template(template_sd, fr, hr); fdrop = fold_drop(fo); vn = [n for n in vec_names(tsd) if n not in fdrop] if vb else []; off = 15; vls = None
    if vb == 16:
        if len(mv) < off + len(vn): raise RID2Error('truncated vlist')
        vls = list(mv[off:off + len(vn)]); off += len(vn)
        if not all(2 <= b <= 15 for b in vls): raise RID2Error('bad vlist')
    if len(mv) < off + 2 * WIDTH: raise RID2Error('truncated')
    if spv: scale16, off = _unpack_ks(mv, off, WIDTH, spv); scale = scale16.astype(np.float32)
    else: scale = np.frombuffer(mv[off:off + 2 * WIDTH], "<f2").astype(np.float32).copy(); off += 2 * WIDTH
    u, off = _unq(mv, off, NTOK * WIDTH, tb); T = torch.from_numpy((u.astype(np.float32).reshape(WIDTH, NTOK) * scale[:, None]).T.copy())
    if wb == 0:
        if len(mv) < off + 2 * WIDTH * NCOORD: raise RID2Error("truncated Wc")
        Wc = torch.from_numpy(np.frombuffer(mv[off:off + 2 * WIDTH * NCOORD], "<f2").astype(np.float32).reshape(WIDTH, NCOORD).copy()); off += 2 * WIDTH * NCOORD
    else:
        if len(mv) < off + 2 * NCOORD: raise RID2Error("truncated Wc")
        sw = np.frombuffer(mv[off:off + 2 * NCOORD], "<f2").astype(np.float32); off += 2 * NCOORD; cw, off = _unq(mv, off, WIDTH * NCOORD, wb); Wc = torch.from_numpy(cw.astype(np.float32).reshape(WIDTH, NCOORD) * sw[None])
    vecs = {}
    for j, n in enumerate(vn):
        cnt = tsd[n].numel()
        if len(mv) < off + 2: raise RID2Error("truncated vectors")
        sv = np.frombuffer(mv[off:off + 2], "<f2").astype(np.float32)[0]; off += 2; uv, off = _unq(mv, off, cnt, vls[j] if vb == 16 else vb); vecs[n] = torch.from_numpy(uv.astype(np.float32) * sv).reshape(tsd[n].shape)
    drop = set(DROP) | set(vn) | set(fdrop); fe = None
    if fm:
        D = fr or 8
        if len(mv) < off + 2 * D + 2: raise RID2Error("truncated frame header")
        fs = np.frombuffer(mv[off:off + 2 * D], "<f2").astype(np.float32).copy(); off += 2 * D; (ln,) = struct.unpack("<H", bytes(mv[off:off + 2])); off += 2
        if len(mv) < off + ln: raise RID2Error("truncated frame stream")
        fc = decode_frames(bytes(mv[off:off + ln]), fm, fi, NFRAME, D); off += ln; fe = torch.from_numpy(fc.astype(np.float32) * fs[None]); drop.add(FE)
    red = {k: v for k, v in tsd.items() if k not in drop}; body = bytes(mv[off:])
    if spv:
        if len(mv) < off + 2: raise RID2Error("truncated scale payload")
        (pl,) = struct.unpack("<H", bytes(mv[off:off + 2])); off += 2
        if len(mv) < off + pl: raise RID2Error("truncated scale payload")
        pay = mv[off:off + pl]; off += pl; pos = [0]
        def lists(n, ns):
            a, pos[0] = _unpack_ks(pay, pos[0], ns, spv); return a.tobytes()
        body = unstrip_scales(bytes(mv[off:]), red, lists)
        if pos[0] != pl: raise RID2Error("scale payload length mismatch")
    sub = unpack_variant_semantic_or_none(body, red)
    if sub is None: raise RID2Error("invalid RID2 sub-blob")
    out = {k: v.clone() for k, v in tsd.items()}
    for k, v in sub.items(): out[k] = v
    for k, v in vecs.items(): out[k] = v
    if fe is not None: out[FE] = fe
    for n in fdrop: out[n] = torch.ones_like(tsd[n]) if n.endswith('norm.weight') else torch.zeros_like(tsd[n])      # coord_mix.bias -> 0 (folded into T)
    if fr:
        from folding import joint; out = joint.expand_film(out, fr)
    if hr:
        from folding import joint; out = joint.compose_head(out)
    e = torch.zeros_like(out["token_embed.weight"]); e[:, :NTOK] = torch.eye(NTOK); w = torch.zeros_like(out["coord_mix.weight"]); w[:, :NTOK, 0, 0] = T.T; w[:, WIDTH:, 0, 0] = Wc
    out["token_embed.weight"] = e; out["coord_mix.weight"] = w; return out
