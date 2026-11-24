#!/usr/bin/env python3
import hashlib
import os
import sys
import tempfile
import zipfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(VENDOR))
import brotli
import dz_carrier
import dz_dither
import dz_hpac
import dz_motion
import dz_render
import dz_tokens
import dzformat
COEF_STEP = 0.012

def _vendor_modules():
    from canonical.inflate_canonical import install_rp1_shim
    install_rp1_shim()
    from decoder import numerics
    from container import token_decode
    from decoder import extfmt, inflate_core
    return (token_decode, extfmt, inflate_core)

def render_motion(renderer, semantic, knobs, tokens, destination, device, direct=False):
    import torch
    from torch.nn import functional as F
    N, H, W = (renderer.N, renderer.CAMERA_H, renderer.CAMERA_W)
    semantic = semantic.eval().to(device)
    output = np.memmap(destination, mode='w+', dtype=np.uint8, shape=(N * 2, H, W, 3))
    dither = dz_dither.Dither(device)
    knobs = knobs.to(device)
    with torch.no_grad():
        for start in range(0, N, 8):
            end = min(start + 8, N)
            tok = tokens[start:end].long().to(device)
            sem = semantic(tok, torch.arange(start, end, device=device))
            if direct:
                master = dither.full(sem).float()
            else:
                master = F.interpolate(sem, size=(H, W), mode='bilinear', align_corners=False).clamp(0.0, 255.0).round()
            seen = F.interpolate(master, size=(dz_motion.EH, dz_motion.EW), mode='bilinear')
            slave = dither.full(dz_motion.slave(seen, tok, knobs[start:end]))
            output[2 * start + 1:2 * end:2] = master.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            output[2 * start:2 * end:2] = slave.permute(0, 2, 3, 1).cpu().numpy()
    output.flush()

def inflate(archive_dir, out_raw):
    import torch
    p = (Path(archive_dir) / 'p').read_bytes()
    token_decode, extfmt, inflate_core = _vendor_modules()
    from runtime.f26_inflate import _load_renderer
    flags, hpac_stream, semantic_stream, carrier_blob, mc_blob, tail = dzformat.unpack(p)
    from container.payload_format import split_inner_payload
    tail_core, trailer = split_inner_payload(tail)

    def hook():
        from runtime import residual_archive as ra
        from runtime.segment_layout import unpack_segments
        compact = len(ra.FIXED_MAGIC) + 2 + ra.packed_length(ra.FIXED_STATES * ra.NUM_CLASSES, ra.FIXED_BITS) - len(ra.FIXED_MAGIC)
        residual = ra.FIXED_MAGIC + tail_core[:compact]
        if flags & 1:
            from runtime import ihs2
            from runtime.f26_inflate import _load_renderer as load
            hpac = dz_hpac.decode(hpac_stream, ihs2, ihs2.layout_from_runtime(load(VENDOR / 'external/pr141/cpr1')))
        else:
            hpac = brotli.decompress(hpac_stream)
        parts = ra.ResidualArchiveParts(semantic_blob=b'' if flags & dzformat.RENDER else unpack_segments(semantic_stream), carrier_blob=b'', hpac_blob=hpac, token_stream=tail_core[compact:], table=ra._decode_fixed_table(residual), schema=ra.FIXED_SCHEMA, residual_payload=residual, compressed_models=b'', compensation_blob=None)
        ra.read_residual_archive = lambda _path: parts
        hook.parts = parts
    token_decode.POST_PURGE_HOOKS.append(hook)
    with tempfile.TemporaryDirectory() as tmp:
        shell = Path(tmp) / 'archive.zip'
        with zipfile.ZipFile(shell, 'w', zipfile.ZIP_STORED) as zf:
            zf.writestr('p', tail)
        if flags & dzformat.TOKEN_MC:
            tokens, info = dz_tokens.decode_tokens(shell, VENDOR / 'external/pr141', mc_blob, 'cuda')
        else:
            tokens, info = token_decode.decode_tokens(shell, VENDOR / 'external/pr141', 'cuda')
    parts = hook.parts
    from runtime.entropy.renderer_weight_codec import decode_wans1
    renderer = _load_renderer(VENDOR / 'external/pr141/cpr1')
    device = torch.device('cuda')
    if flags & dzformat.RENDER:
        semantic = dz_render.decode(semantic_stream)
    else:
        semantic = renderer.SemanticTokenRenderer(96)
        state = renderer.unpack_variant_semantic_or_none(parts.semantic_blob, semantic.state_dict())
        if state is None:
            state = {r.schema.name: torch.from_numpy(np.ascontiguousarray(r.values, dtype=np.float32)) for r in decode_wans1(parts.semantic_blob)}
        semantic.load_state_dict(state, strict=True)
    _, ext_payload = extfmt.split_trailer(trailer, renderer.N)
    exts = extfmt.parse_payload(ext_payload) if ext_payload else dict(ext1=None, ext2=None)
    if exts['ext2'] is not None:
        raise ValueError('EXT2 records are not supported')
    out_raw = Path(out_raw)
    if out_raw.exists():
        out_raw.unlink()
    if flags & 2:
        if exts['ext1'] is not None:
            raise ValueError('unsupported archive')
        render_motion(renderer, semantic, torch.from_numpy(dz_motion.decode_blob(carrier_blob)), tokens, out_raw, device, bool(flags & dzformat.RENDER))
    else:
        if flags & dzformat.RENDER:
            raise ValueError('unsupported archive')
        codes, q = dz_carrier.decode(carrier_blob)
        basis = torch.from_numpy(codes.astype(np.float32))
        coefficients = torch.from_numpy(q.astype(np.float32)) * COEF_STEP
        inflate_core.render_video37(renderer, semantic, basis, coefficients, tokens, out_raw, device, exts['ext1'], None)
    torch.cuda.synchronize()
    return dict(token_sha=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(), hpac_sha=hashlib.sha256(parts.hpac_blob).hexdigest(), semantic_sha=hashlib.sha256(semantic_stream if flags & dzformat.RENDER else parts.semantic_blob).hexdigest())

def main():
    if len(sys.argv) != 4:
        raise SystemExit('usage: dz_inflate.py <archive_dir> <output_dir> <video_names_file>')
    archive_dir, output_dir, names = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = [line.strip().rsplit('.', 1)[0] for line in names.read_text().splitlines() if line.strip()]
    if stems != ['0']:
        raise SystemExit(f'unsupported video list: {stems}')
    archive_dir, out_raw = (archive_dir.resolve(), output_dir.resolve() / '0.raw')
    os.chdir(VENDOR)
    print(inflate(archive_dir, out_raw), flush=True)
if __name__ == '__main__':
    main()
