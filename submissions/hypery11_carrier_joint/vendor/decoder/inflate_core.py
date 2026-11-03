"""Integrated decoder ("actual inflate path"): archive.zip -> tokens (warp-aware RC64 decode, inherited PR141 runtime) -> frames (render_video clone with
EXT1 carrier-grid residual and EXT2 master-correction hooks) -> uint8 .raw [1200,874,1164,3].  
python -m decoder.inflate_core <archive.zip> <out.raw> [noext=1]      (noext=1 ignores the extension trailer: EXT-disabled parity)
"""
import hashlib, json, math, os, struct, sys, tempfile, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "external/pr141"))
from decoder import extfmt

def _load(archive):
    import torch
    from container.payload_format import read_archive, split_inner_payload, write_archive
    from container.archive_writer import EXTERNAL_ATTR_PR141
    from runtime import residual_archive as ra
    p = read_archive(archive); core, w2 = split_inner_payload(p); warp, payload = extfmt.split_trailer(w2, 600)
    with tempfile.TemporaryDirectory() as t:
        a = Path(t) / "core.zip"; write_archive(a, core, EXTERNAL_ATTR_PR141); parts = ra.read_residual_archive(a)
    return parts, warp, payload

def grid_residual(q, step, fam, dev):
    """EXT1 residual [384,512] from integer coefficients (decoder-side generic construction)."""
    import torch
    from torch.nn import functional
    if fam == "grid4":
        g = torch.tensor(np.asarray(q, dtype=np.float32) * step, device=dev).view(1, 1, 4, 4); return functional.interpolate(g, size=(384, 512), mode="bilinear", align_corners=False)[0, 0]
    raise extfmt.ExtError(f"family {fam} not supported by the decoder")

def render_video37(renderer, semantic, basis, coefficients, tokens, destination, device, ext1=None, master_fn=None):
    import torch
    from torch.nn import functional
    r = renderer; N = r.N
    semantic = semantic.eval().to(device); basis = r.normalized_basis(basis.to(device)); coefficients = coefficients.to(device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(destination, mode="w+", dtype=np.uint8, shape=(N * 2, r.CAMERA_H, r.CAMERA_W, 3))
    resid = {}
    if ext1 is not None:
        for pid, q in ext1["q"].items(): resid[int(pid)] = grid_residual(q, ext1["step"], ext1["fam"], device)
    with torch.no_grad():
        for start in range(0, N, 8):
            end = min(start + 8, N); indices = torch.arange(start, end, device=device)
            sem = semantic(tokens[start:end].long().to(device), indices)
            if master_fn is not None: sem = master_fn(sem, tokens[start:end].long().to(device), indices)
            master = functional.interpolate(sem, size=(r.CAMERA_H, r.CAMERA_W), mode="bilinear", align_corners=False).clamp(0.0, 255.0).round()
            mnp = master.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for o in range(end - start): output[2 * (start + o) + 1] = mnp[o]
        for start in range(0, N, 64):
            end = min(start + 64, N); carrier = torch.einsum("bk,kchw->bchw", coefficients[start:end], basis) / math.sqrt(r.CARRIER_DIM)
            base = 127.5 + r.CARRIER_AMPLITUDE * carrier
            for k in range(start, end):
                if k in resid: base[k - start] = base[k - start] + resid[k][None]
            slave = functional.interpolate(base.clamp(0.0, 255.0).round(), size=(r.CAMERA_H, r.CAMERA_W), mode="bicubic", align_corners=False).clamp(0.0, 255.0).round()
            snp = slave.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for o in range(end - start): output[2 * (start + o)] = snp[o]
    output.flush(); return destination

def inflate(archive, out_raw, noext=False, master_factory=None):
    import torch
    from container.token_decode import decode_tokens
    from runtime.carrier_repack import materialize_cpr1, split_frame0_selector_carrier
    from runtime.entropy.renderer_weight_codec import decode_wans1
    from runtime.frame0_selector import apply_pixel_mode, decode_selector
    from runtime.f26_inflate import _load_renderer
    archive = Path(archive); out_raw = Path(out_raw); parts, warp, payload = _load(archive); exts = extfmt.parse_payload(payload) if payload and not noext else dict(ext1=None, ext2=None)
    dec, info = decode_tokens(archive, REPO / "external/pr141", "cuda"); tokens = dec
    renderer = _load_renderer(REPO / "external/pr141/cpr1"); device = torch.device("cuda")
    carrier_blob, selector_blob = split_frame0_selector_carrier(parts.carrier_blob); canon = materialize_cpr1(carrier_blob, renderer)
    sp = struct.pack("<II", 40_252, len(canon)) + bytes(40_252) + canon; _, basis, coefficients = renderer.unpack_semantic_pose(sp)
    semantic = renderer.SemanticTokenRenderer(96); tagged = renderer.unpack_variant_semantic_or_none(parts.semantic_blob, semantic.state_dict())
    if tagged is None:
        tagged = {rec.schema.name: torch.from_numpy(np.ascontiguousarray(rec.values, dtype=np.float32)) for rec in decode_wans1(parts.semantic_blob)}
    semantic.load_state_dict(tagged, strict=True)
    master_fn = master_factory(exts["ext2"], device) if (exts["ext2"] is not None and master_factory) else None
    if exts["ext2"] is not None and master_fn is None: raise extfmt.ExtError("EXT2 present but no master decoder available")
    if out_raw.exists(): out_raw.unlink()
    render_video37(renderer, semantic, basis, coefficients, tokens, out_raw, device, exts["ext1"], master_fn)
    if selector_blob is not None:
        modes, indices = decode_selector(selector_blob); out = np.memmap(out_raw, mode="r+", dtype=np.uint8, shape=(renderer.N * 2, renderer.CAMERA_H, renderer.CAMERA_W, 3))
        for mi, mode in enumerate(modes):
            ids = np.flatnonzero(indices[:renderer.N] == mi)
            if ids.size: out[2 * ids] = apply_pixel_mode(np.asarray(out[2 * ids]).copy(), mode)
        out.flush()
    torch.cuda.synchronize()
    return dict(token_sha=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(), ext1_pairs=sorted(exts["ext1"]["q"]) if exts["ext1"] else [], ext2_bytes=len(exts["ext2"]) if exts["ext2"] else 0, payload_bytes=len(payload),
                carrier_sha=hashlib.sha256(parts.carrier_blob).hexdigest(), hpac_sha=hashlib.sha256(parts.hpac_blob).hexdigest(), semantic_sha=hashlib.sha256(parts.semantic_blob).hexdigest(), warp_sha=hashlib.sha256(warp).hexdigest())

def sha_raw(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b: break
            h.update(b)
    return h.hexdigest()

if __name__ == "__main__":
    if True:
        kv = dict(a.split("=", 1) for a in sys.argv[3:]); t0 = time.time(); info = inflate(sys.argv[1], sys.argv[2], noext=kv.get("noext") == "1"); info["raw_sha256"] = sha_raw(sys.argv[2]); info["secs"] = time.time() - t0; print(json.dumps(info))
