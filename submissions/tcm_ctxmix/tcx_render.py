"""Frame rendering, taken from #153 vendor/decoder/inflate_core.py (render_video37 and inflate), split into an
even-frame part (carrier; needs no tokens) and an odd-frame part (SemanticTokenRenderer; needs the token maps),
so the even frames can be rendered while the TCM token decoder is still running.  The arithmetic of every frame
is the same op sequence, batch size and dtype as in #153.
"""
from __future__ import annotations

import math
import struct

import numpy as np
import torch
from torch.nn import functional

from runtime.carrier_repack import materialize_cpr1, split_frame0_selector_carrier
from runtime.entropy.renderer_weight_codec import decode_wans1
from runtime.f26_inflate import _load_renderer
from runtime.frame0_selector import apply_pixel_mode, decode_selector


def load_renderer_module(pr141_dir):
    """#151's renderer module (external/pr141/cpr1/inflate.py), with #153's RID1/RID2/RID9 SM3R unpackers
    (same logic as vendor/folding/renderer_shim.py, without its HPAC rowpack hook)."""
    module = _load_renderer(pr141_dir / "cpr1")
    if not getattr(module, "_rid_patched", False):
        inner = module.unpack_variant_semantic_or_none

        def unpack(blob, template):
            if bytes(blob[:4]) == b"SM3R" and len(blob) > 5 and blob[5] == 9:
                from folding.rid9 import decode_state_dict
                return decode_state_dict(blob, template)
            if bytes(blob[:4]) == b"SM3R" and len(blob) > 5 and blob[5] == 8:
                from folding.rid2 import decode_state_dict
                return decode_state_dict(blob, template)
            if bytes(blob[:4]) == b"SM3R" and len(blob) > 5 and blob[5] == 7:
                from folding.rid1 import decode_state_dict
                return decode_state_dict(blob, template)
            return inner(blob, template)
        module.unpack_variant_semantic_or_none = unpack
        module._rid_patched = True
    return module


def build_semantic(r, semantic_blob):
    semantic = r.SemanticTokenRenderer(96)
    tagged = r.unpack_variant_semantic_or_none(semantic_blob, semantic.state_dict())
    if tagged is None:
        tagged = {rec.schema.name: torch.from_numpy(np.ascontiguousarray(rec.values, dtype=np.float32))
                  for rec in decode_wans1(semantic_blob)}
    semantic.load_state_dict(tagged, strict=True)
    return semantic


def build_carrier(r, carrier_blob):
    """-> (basis, coefficients, selector_blob | None), as in #153 inflate_core.inflate."""
    carrier, selector_blob = split_frame0_selector_carrier(carrier_blob)
    canon = materialize_cpr1(carrier, r)
    sp = struct.pack("<II", 40_252, len(canon)) + bytes(40_252) + canon
    _, basis, coefficients = r.unpack_semantic_pose(sp)
    return basis, coefficients, selector_blob


def grid_residual(q, step, fam, dev):
    """EXT1 residual [384,512] from integer coefficients (#153 inflate_core.grid_residual)."""
    if fam == "grid4":
        g = torch.tensor(np.asarray(q, dtype=np.float32) * step, device=dev).view(1, 1, 4, 4)
        return functional.interpolate(g, size=(384, 512), mode="bilinear", align_corners=False)[0, 0]
    raise ValueError(f"EXT1 family {fam} not supported")


def render_even(r, basis, coefficients, ext1, selector_blob, output, device):
    """Even frames output[2k]: carrier + EXT1 grid residual, bicubic upscale, then the frame-0 pixel selector."""
    N = r.N
    basis = r.normalized_basis(basis.to(device))
    coefficients = coefficients.to(device)
    resid = {}
    if ext1 is not None:
        for pid, q in ext1["q"].items():
            resid[int(pid)] = grid_residual(q, ext1["step"], ext1["fam"], device)
    with torch.no_grad():
        for start in range(0, N, 64):
            end = min(start + 64, N)
            carrier = torch.einsum("bk,kchw->bchw", coefficients[start:end], basis) / math.sqrt(r.CARRIER_DIM)
            base = 127.5 + r.CARRIER_AMPLITUDE * carrier
            for k in range(start, end):
                if k in resid:
                    base[k - start] = base[k - start] + resid[k][None]
            slave = functional.interpolate(base.clamp(0.0, 255.0).round(), size=(r.CAMERA_H, r.CAMERA_W),
                                           mode="bicubic", align_corners=False).clamp(0.0, 255.0).round()
            snp = slave.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for o in range(end - start):
                output[2 * (start + o)] = snp[o]
    if selector_blob is not None:
        modes, indices = decode_selector(selector_blob)
        for mi, mode in enumerate(modes):
            ids = np.flatnonzero(indices[:N] == mi)
            if ids.size:
                output[2 * ids] = apply_pixel_mode(np.asarray(output[2 * ids]).copy(), mode)


def render_odd(r, semantic, tokens, output, device):
    """Odd frames output[2k+1]: SemanticTokenRenderer(tokens), bilinear upscale (batches of 8, as #153)."""
    N = r.N
    semantic = semantic.eval().to(device)
    with torch.no_grad():
        for start in range(0, N, 8):
            end = min(start + 8, N)
            indices = torch.arange(start, end, device=device)
            sem = semantic(tokens[start:end].long().to(device), indices)
            master = functional.interpolate(sem, size=(r.CAMERA_H, r.CAMERA_W), mode="bilinear",
                                            align_corners=False).clamp(0.0, 255.0).round()
            mnp = master.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for o in range(end - start):
                output[2 * (start + o) + 1] = mnp[o]
