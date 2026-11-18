"""RID1 = Renderer Interface Distillation v1 semantic section (fold family).
After LAY1 unpacking the blob is:
    b"SM3R" | ver u8 (=1) | mode u8 (=7, RID1 fold) | t_bits u8 | c_fmt u8 (0 = fp16) | v_bits u8 (0 = vectors stay fp16 inside the sub-blob) | T scales fp16[96] | T codes int<t_bits> packed [96 channels x 5 tokens, channel-major] | Wc fp16 [96,4] row-major | [if v_bits: for every 1-D tensor with >=16 elements in template order: fp16 scale + int<v_bits> codes, byte aligned] | SM3R(mixed) sub-blob of the REDUCED template
The reduced template is the parent template without token_embed.weight / coord_mix.weight (coord_mix.bias stays in the sub-blob).  The decoder rebuilds the PARENT state dict:
    token_embed.weight = [I5 | 0],  coord_mix.weight[:, :5] = T^T,  coord_mix.weight[:, 96:] = Wc  => the unmodified renderer computes the folded function.
Fail closed: bad magic/version/bits, truncated or trailing bytes, invalid sub-blob."""
import struct, sys
import numpy as np, torch
sys.path.insert(0, '.')
from semantic import sem_codec as sc
from runtime.segment_layout import pack_segments, unpack_segments
MAGIC = b"SM3R"; MODE = 7   # tagged SM3R so the frozen RX1 reader accepts the section; mode 7 = RID1 fold (SM3R modes 5/6 are the parent ones)
NTOK, WIDTH, NCOORD = 5, 96, 4; DROP = ("token_embed.weight", "coord_mix.weight")


class RID1Error(ValueError): pass


def _reduced():
    tm = {k: v for k, v in sc.TEMPLATE.items() if k not in DROP}; return tm


def vec_names(tm): return [k for k, v in tm.items() if v.ndim == 1 and v.numel() >= 16]


def encode(qr, t_bits=12, c_fmt=0, v_bits=0):
    """qr: RendererQ (parent).  -> (packed LAY1 section, info)"""
    if not 2 <= t_bits <= 15 or c_fmt != 0 or not (v_bits == 0 or 2 <= v_bits <= 15): raise RID1Error("unsupported fold format")
    sd = qr.state_dict(); E = sd["token_embed.weight"].double(); W = sd["coord_mix.weight"][:, :, 0, 0].double(); T = E @ W[:, :WIDTH].T       # [5,96]
    qm = (1 << (t_bits - 1)) - 1; scale = (T.abs().amax(0) / qm).clamp_min(1e-8).numpy().astype("<f2"); s32 = torch.from_numpy(scale.astype(np.float32)).double().clamp_min(1e-8)
    codes = torch.clamp(torch.round(T / s32[None]), -qm, qm).numpy().astype(np.int8 if t_bits <= 8 else np.int16)                      # [5,96]
    Wc = W[:, WIDTH:].numpy().astype("<f2")                                                                                              # [96,4]
    head = MAGIC + bytes([1, MODE, t_bits, c_fmt, v_bits]) + scale.tobytes(); tcodes = sc._pack_bits(codes.T.reshape(-1), t_bits); wc = Wc.tobytes()
    vn = vec_names(sc.TEMPLATE) if v_bits else []; vb = b""
    for n in vn:
        v = torch.from_numpy(qr.fp16[n].astype(np.float32)).double(); qm_ = (1 << (v_bits - 1)) - 1; sv = np.float16(max(float(v.abs().max()) / qm_, 1e-8)); cv = torch.clamp(torch.round(v / float(np.float32(sv))), -qm_, qm_).numpy().astype(np.int16)
        vb += np.array([sv], "<f2").tobytes() + sc._pack_bits(cv, v_bits)
    tm0, qn0 = sc.TEMPLATE, sc.QNAMES; tm = {k: v for k, v in _reduced().items() if k not in vn}; q2 = qr.copy()
    for n in DROP + tuple(vn): q2.depth.pop(n, None); q2.scales.pop(n, None); q2.codes.pop(n, None); q2.fp16.pop(n, None)
    try:
        sc.TEMPLATE = tm; sc.QNAMES = [n for n in qn0 if n not in DROP]; _, subblob, _, per = sc.encode(q2)
    finally: sc.TEMPLATE, sc.QNAMES = tm0, qn0
    prefix = head + tcodes + wc + vb; blob = prefix + subblob; section = pack_segments([prefix, subblob])
    assert unpack_segments(section) == blob
    return section, dict(prefix_B=len(prefix), sub_B=len(subblob), blob_B=len(blob), section_B=len(section), T=T.numpy(), codes=codes, scale=scale)


def decode_state_dict(blob, template_sd):
    """blob (LAY1-unpacked) -> parent-shaped state dict (float32 tensors)."""
    from ddm_mp2_semantic_receiver import unpack_variant_semantic_or_none
    mv = memoryview(blob)
    if len(mv) < 9 + 2 * WIDTH or bytes(mv[:4]) != MAGIC or mv[5] != MODE: raise RID1Error("bad RID1 header")
    ver, tb, cf, vbits = mv[4], mv[6], mv[7], mv[8]
    if ver != 1 or not 2 <= tb <= 15 or cf != 0 or not (vbits == 0 or 2 <= vbits <= 15): raise RID1Error("unsupported RID1 version/format")
    off = 9; scale = np.frombuffer(mv[off:off + 2 * WIDTH], "<f2").astype(np.float32).copy(); off += 2 * WIDTH
    nb = (NTOK * WIDTH * tb + 7) // 8
    if len(mv) < off + nb + 2 * WIDTH * NCOORD: raise RID1Error("truncated RID1 payload")
    bits = np.unpackbits(np.frombuffer(mv[off:off + nb], np.uint8), bitorder="little")[:NTOK * WIDTH * tb].reshape(-1, tb).astype(np.int32); off += nb
    u = (bits * (1 << np.arange(tb, dtype=np.int32))[None]).sum(1); u = np.where(u >= (1 << (tb - 1)), u - (1 << tb), u).astype(np.float32).reshape(WIDTH, NTOK)      # [96 ch, 5 tok]
    T = torch.from_numpy((u * scale[:, None]).T.copy())                                                                                                                        # [5,96]
    Wc = torch.from_numpy(np.frombuffer(mv[off:off + 2 * WIDTH * NCOORD], "<f2").astype(np.float32).reshape(WIDTH, NCOORD).copy()); off += 2 * WIDTH * NCOORD
    vn = vec_names(template_sd) if vbits else []; vecs = {}
    for n in vn:
        cnt = template_sd[n].numel(); nbv = (cnt * vbits + 7) // 8
        if len(mv) < off + 2 + nbv: raise RID1Error("truncated RID1 vectors")
        sv = np.frombuffer(mv[off:off + 2], "<f2").astype(np.float32)[0]; off += 2
        b_ = np.unpackbits(np.frombuffer(mv[off:off + nbv], np.uint8), bitorder="little")[:cnt * vbits].reshape(-1, vbits).astype(np.int32); off += nbv
        uv = (b_ * (1 << np.arange(vbits, dtype=np.int32))[None]).sum(1); uv = np.where(uv >= (1 << (vbits - 1)), uv - (1 << vbits), uv).astype(np.float32); vecs[n] = torch.from_numpy(uv * sv).reshape(template_sd[n].shape)
    red = {k: v for k, v in template_sd.items() if k not in DROP and k not in vn}; sub = unpack_variant_semantic_or_none(bytes(mv[off:]), red)
    if sub is None: raise RID1Error("invalid RID1 sub-blob")
    out = {k: v.clone() for k, v in template_sd.items()}
    for k, v in sub.items(): out[k] = v
    for k, v in vecs.items(): out[k] = v
    e = torch.zeros_like(out["token_embed.weight"]); e[:, :NTOK] = torch.eye(NTOK); w = torch.zeros_like(out["coord_mix.weight"]); w[:, :NTOK, 0, 0] = T.T; w[:, WIDTH:, 0, 0] = Wc
    out["token_embed.weight"] = e; out["coord_mix.weight"] = w; return out
