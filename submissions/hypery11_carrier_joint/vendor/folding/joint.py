"""Structural reparameterisations expressed on a RendererQ (parent-shaped quantised state):
  make_film(qr, r)  : joint frame_embed+FiLM basis.  Y = Z Wall^T (600 x active-rows) -> SVD rank r: frame coefficients Z' [600,r] (4-bit, per-dim fp16 scales) and shared basis W' [rows,r] re-quantised per row (same codec as the parent FiLM tensors).
  make_head(qr, r)  : output head 3x3 [3,96,3,3] = (1x1 96->r) o (3x3 r->3); decoder composes them exactly.
Both return a RendererQ whose tensors carry the NEW shapes + a template override dict (name -> shape) used by rid2 for the sub-blob."""
import sys
import numpy as np, torch
sys.path.insert(0, '.')
from semantic import sem_codec as sc
KEEP = 2   # SM3R keep_percent for the pruned FiLM tensors (the frozen reader requires kept rows == round(192*keep/100) for every pruned tensor): 2 -> 4 rows
FE = "frame_embed.weight"; FILMS = [f"blocks.{l}.film.weight" for l in range(4)]


def _qrows(W, bits):
    """per-row symmetric quantisation like the parent (fp16 scale per row)."""
    qm = sc.qmax(bits); s = np.maximum(np.abs(W).max(1) / qm, 1e-6).astype(np.float16); s32 = s.astype(np.float32); c = np.clip(np.round(W / s32[:, None]), -qm, qm).astype(np.int8); return c, s


def film_matrices(qr):
    sd = qr.state_dict(); Z = (np.asarray(qr.codes[FE], np.float32).reshape(600, 8) * np.asarray(qr.scales[FE], np.float32)[None]); rows = []; W = []
    for n in FILMS:
        w = sd[n].numpy().astype(np.float64); nz = np.where(np.abs(w).sum(1) > 0)[0]; rows.append(nz); W.append(w[nz])
    return Z.astype(np.float64), rows, W


def make_film(qr, r, b0=3, b123=4, zbits=4):
    Z, rows, W = film_matrices(qr); Wall = np.concatenate(W, 0); Y = Z @ Wall.T; U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    Zp = U[:, :r] * S[:r][None]; Wp = Vt[:r].T                                                                        # Y ~ Zp Wp^T ; Wp [rows,r]
    qm = sc.qmax(zbits); zs = np.maximum(np.abs(Zp).max(0) / qm, 1e-8).astype(np.float16); zc = np.clip(np.round(Zp / zs.astype(np.float32)[None]), -qm, qm).astype(np.int8)
    q2 = qr.copy(); q2.codes[FE] = zc.reshape(-1); q2.scales[FE] = zs; q2.depth[FE] = zbits; off = 0; shapes = {FE: (600, r)}
    for l, n in enumerate(FILMS):
        k = len(rows[l]); w = Wp[off:off + k]; off += k; bits = b0 if l == 0 else b123
        if l == 0:
            c, s = _qrows(w, bits); q2.codes[n] = c.reshape(-1); q2.scales[n] = s; q2.depth[n] = bits; shapes[n] = (192, r)
        else:
            c, s = _qrows(w, bits); q2.codes[n] = c.reshape(-1); q2.scales[n] = s; q2.depth[n] = bits; shapes[n] = (192, r)      # rowmask unchanged (same active rows)
    return q2, shapes, dict(sv=S.tolist(), var_kept=float((S[:r] ** 2).sum() / (S ** 2).sum()), Zp_codes=zc, Zp_scale=zs)


def expand_film(sd, r):
    """decoder side: new-shaped frame_embed/film tensors -> parent-shaped (zero-padded columns)."""
    out = dict(sd); fe = torch.zeros(600, 8); fe[:, :r] = sd[FE]; out[FE] = fe
    for n in FILMS: w = torch.zeros(192, 8); w[:, :r] = sd[n]; out[n] = w
    return out


def make_head(qr, r, bits=4):
    Wh = qr.state_dict()["head.weight"].numpy().astype(np.float64)                      # [3,96,3,3]
    M = Wh.transpose(1, 0, 2, 3).reshape(96, 27); U, S, Vt = np.linalg.svd(M, full_matrices=False)              # M = sum_j U_j S_j V_j : input-channel unfolding
    A = (U[:, :r] * np.sqrt(S[:r])[None]).T                                                                         # [r,96]   1x1 (96 -> r)
    B = (np.sqrt(S[:r])[:, None] * Vt[:r]).reshape(r, 3, 3, 3).transpose(1, 0, 2, 3)                              # [3,r,3,3] 3x3 (r -> 3)
    ca, sa = _qrows(A, bits); cb, sb = _qrows(B.reshape(3, -1), bits); q2 = qr.copy()
    for n in ("head.weight",): q2.depth.pop(n); q2.scales.pop(n); q2.codes.pop(n)
    q2.depth.update({"head.u.weight": bits, "head.v.weight": bits}); q2.scales.update({"head.u.weight": sa, "head.v.weight": sb}); q2.codes.update({"head.u.weight": ca.reshape(-1), "head.v.weight": cb.reshape(-1)})
    return q2, {"head.u.weight": (r, 96, 1, 1), "head.v.weight": (3, r, 3, 3)}, dict(sv=S.tolist(), var_kept=float((S[:r] ** 2).sum() / (S ** 2).sum()))


def compose_head(sd):
    A = sd["head.u.weight"][:, :, 0, 0]; B = sd["head.v.weight"]; out = dict(sd); out["head.weight"] = torch.einsum("ojyx,jc->ocyx", B, A).contiguous()
    out.pop("head.u.weight"); out.pop("head.v.weight"); return out


def make_qr(qr, opts):
    """apply the structural options (film=r, head=r, fb0/fb1/hb bit depths) -> modified RendererQ (new-shaped tensors).  Uses the PARENT qr for all SVDs."""
    q = qr
    if opts.get("fold"): q, _ = fold_affine(qr, gnm=bool(opts.get("gnm")))
    if opts.get("film"): q, _, _ = make_film(qr, opts["film"], b0=opts.get("fb0", 3), b123=opts.get("fb1", 4))
    if opts.get("head"):
        qh, _, _ = make_head(qr, opts["head"], bits=opts.get("hb", 4)); q = q.copy()
        for n in ("head.weight",): q.depth.pop(n, None); q.scales.pop(n, None); q.codes.pop(n, None)
        for n in ("head.u.weight", "head.v.weight"): q.depth[n] = qh.depth[n]; q.scales[n] = qh.scales[n]; q.codes[n] = qh.codes[n]
    return q


def fold_affine(qr, gnm=False):
    """exact algebraic fold (real arithmetic) of redundant affine parameterisation:  dw.bias -> pw.bias;  norm.weight/bias -> FiLM (blocks 1-3 fully, block 0 scale-side only).  Returns a new RendererQ (parent shapes) whose fp16 dict holds float32 merged vectors; dropped vectors are not stored (decoder fills 0/1)."""
    q2 = qr.copy(); sd = qr.state_dict(); info = {}; q2.keep_percent = KEEP
    for l in range(4):
        P = f"blocks.{l}."; Wpw = sd[P + "pw.weight"][:, :, 0, 0].double().numpy(); dwb = qr.fp16[P + "dw.bias"].astype(np.float32).astype(np.float64); pwb = qr.fp16[P + "pw.bias"].astype(np.float32).astype(np.float64)
        nb_pw = pwb + Wpw @ dwb
        if gnm: nb_pw = nb_pw - nb_pw.reshape(12, 8).mean(1, keepdims=True).repeat(8, 1).reshape(96)       # GroupNorm(12 groups of 8) removes any per-group constant exactly
        q2.fp16[P + "pw.bias"] = nb_pw.astype(np.float32)
        g = qr.fp16[P + "norm.weight"].astype(np.float32).astype(np.float64); b = qr.fp16[P + "norm.bias"].astype(np.float32).astype(np.float64); fb = qr.fp16[P + "film.bias"].astype(np.float32).astype(np.float64); bs, bh = fb[:96], fb[96:]
        n = P + "film.weight"; W = sd[n].double().numpy(); Ws, Wh = W[:96], W[96:]
        if l == 0:
            nb = np.concatenate([g * (1 + bs) - 1, bh]); q2.fp16[P + "norm.bias"] = (b / g).astype(np.float32); q2.scales[n] = (qr.scales[n].astype(np.float32) * np.concatenate([g, np.ones(96)]).astype(np.float32)).astype(np.float16)    # codes unchanged; scale rows c<96 *= gamma
        else:
            nb = np.concatenate([g * (1 + bs) - 1, b * (1 + bs) + bh]); Wn = np.concatenate([g[:, None] * Ws, Wh + b[:, None] * Ws], 0); nr = np.abs(Wn).sum(1); K = max(1, round(192 * KEEP / 100.0)); keep = np.zeros(192, bool); keep[np.argsort(-nr, kind='stable')[:K]] = True
            assert (nr[~keep] <= 1e-7).all(), ('more active rows than the frozen reader allows', l, int((nr > 1e-7).sum()), K)
            c, s = _qrows(Wn[keep], qr.depth[n]); q2.rowmask[n] = keep; q2.codes[n] = c.reshape(-1); q2.scales[n] = s; info[n] = int(keep.sum())
        q2.fp16[P + "film.bias"] = nb.astype(np.float32)
    return q2, info


def vlist(q, opts):
    """per-vector bit depths for a uniform absolute quantisation step opts['vstep'] (units of 1e-6); None when not requested.  Order == rid2 storage order."""
    if not opts.get("vstep"): return None
    from folding import rid2
    step = opts["vstep"] * 1e-6; vn = [n for n in rid2.vec_names(rid2.shaped_template(sc.TEMPLATE, opts.get("film", 0), opts.get("head", 0))) if n not in rid2.fold_drop(opts.get("fold", 0))]
    return [int(min(15, max(4, np.ceil(np.log2(np.abs(q.fp16[n].astype(np.float32)).max() / step + 1)) + 1))) for n in vn]


def head_from_AB(qr, A, B, bits=4):
    """A [r,96] (1x1), B [3,r,3,3] (3x3) float -> qr with head.u/head.v quantised per row (parent-style).  Returns (q2, shapes)."""
    r = A.shape[0]; ca, sa = _qrows(np.asarray(A, np.float64), bits); cb, sb = _qrows(np.asarray(B, np.float64).reshape(3, -1), bits); q2 = qr.copy()
    for n in ("head.weight",): q2.depth.pop(n, None); q2.scales.pop(n, None); q2.codes.pop(n, None)
    q2.depth.update({"head.u.weight": bits, "head.v.weight": bits}); q2.scales.update({"head.u.weight": sa, "head.v.weight": sb}); q2.codes.update({"head.u.weight": ca.reshape(-1), "head.v.weight": cb.reshape(-1)}); return q2
