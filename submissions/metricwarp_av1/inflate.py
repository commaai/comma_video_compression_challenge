#!/usr/bin/env python
"""Inflate: reconstruct full-res raw frames from our compressed archive.

Self-contained: uses only numpy, torch, av, brotli (all in the harness base env).
Reads <data_dir>/manifest.json + streams, writes <out_dir>/<base>.raw.

Pipeline: PyAV decode -> our YUV->RGB (full-range BT.601, bilinear chroma up)
-> optional parity-drop of the priming frame -> optional downsample to 512x384
-> per-pair even-frame correction (affine warp + bias/gain) -> exact-grid
placement into 1164x874 (metric-input-exact) with bilinear cosmetic fill.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

H, W = 874, 1164
h, w = 384, 512

def _taps(out, inp):
    x = (np.arange(out) + 0.5) * (inp / out) - 0.5
    x = np.clip(x, 0, inp - 1)
    c = np.floor(x).astype(np.int64)
    return c, np.minimum(c + 1, inp - 1)

CY, CY1 = _taps(h, H)
CX, CX1 = _taps(w, W)

def decode_yuv_frames(path):
    import av
    fmt = 'obu' if str(path).endswith('.obu') else None
    container = av.open(str(path), format=fmt)
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        name = frame.format.name
        bits = 10 if '10' in name else 8
        dt = np.uint16 if bits == 10 else np.uint8
        esz = 2 if bits == 10 else 1
        fh, fw = frame.height, frame.width
        c444 = '444' in name
        ch, cw = (fh, fw) if c444 else (fh // 2, fw // 2)
        y = np.frombuffer(frame.planes[0], dtype=dt).reshape(fh, frame.planes[0].line_size // esz)[:, :fw]
        u = np.frombuffer(frame.planes[1], dtype=dt).reshape(ch, frame.planes[1].line_size // esz)[:, :cw]
        v = np.frombuffer(frame.planes[2], dtype=dt).reshape(ch, frame.planes[2].line_size // esz)[:, :cw]
        yield y.copy(), u.copy(), v.copy(), bits
    container.close()

def yuv_to_rgb_f32(y, u, v, bits):
    scale = float(1 << (bits - 8))
    Yf = torch.from_numpy(y.astype(np.float32)) / scale
    Uf = torch.from_numpy(u.astype(np.float32)) / scale
    Vf = torch.from_numpy(v.astype(np.float32)) / scale
    fh, fw = Yf.shape
    if Uf.shape[0] != fh:
        Uf = F.interpolate(Uf[None, None], size=(fh, fw), mode='bilinear', align_corners=False)[0, 0]
        Vf = F.interpolate(Vf[None, None], size=(fh, fw), mode='bilinear', align_corners=False)[0, 0]
    R = Yf + 1.402 * (Vf - 128.0)
    G = Yf - 0.344136 * (Uf - 128.0) - 0.714136 * (Vf - 128.0)
    B = Yf + 1.772 * (Uf - 128.0)
    return torch.stack([R, G, B], dim=-1).clamp(0, 255)  # (fh,fw,3) float

def downsample_512(x_f32, mode):
    t = x_f32.permute(2, 0, 1).unsqueeze(0)
    if mode == 'bilinear':
        y = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
    elif mode == 'area':
        y = F.interpolate(t, size=(h, w), mode='area')
    else:
        raise ValueError(mode)
    return y[0].permute(1, 2, 0)

def apply_correction(x_f32, dx, dy, rot, zoom, bias, gain):
    if dx or dy or rot or zoom:
        fh, fw = x_f32.shape[0], x_f32.shape[1]
        t = x_f32.permute(2, 0, 1).unsqueeze(0)
        r = rot * 1e-3
        z = 1.0 + zoom * 1e-3
        cos, sin = np.cos(r), np.sin(r)
        theta = torch.tensor([[[cos / z, -sin / z * fh / fw, -dx * 2 / fw],
                               [sin / z * fw / fh, cos / z, -dy * 2 / fh]]], dtype=torch.float32)
        grid = F.affine_grid(theta, t.shape, align_corners=False)
        t = F.grid_sample(t, grid, mode='bilinear', padding_mode='border', align_corners=False)
        x_f32 = t[0].permute(1, 2, 0)
    if bias or gain:
        x_f32 = x_f32 * (1.0 + gain * 1e-3) + bias
    return x_f32.clamp(0, 255)

def gridplace_fullres(t512_u8):
    """(384,512,3) uint8 -> (874,1164,3) uint8 with exact metric taps."""
    x = t512_u8.permute(2, 0, 1).unsqueeze(0).float()
    up = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
    up = up.round_().clamp_(0, 255).to(torch.uint8)[0]  # (3,H,W)
    tt = t512_u8.permute(2, 0, 1)
    for a, ys in ((0, CY), (1, CY1)):
        for b, xs in ((0, CX), (1, CX1)):
            up[:, torch.from_numpy(ys)[:, None], torch.from_numpy(xs)[None, :]] = tt
    return up.permute(1, 2, 0).numpy()

def main():
    data_dir, out_dir, file_list = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    man = json.loads((data_dir / 'manifest.json').read_text())
    corr = None
    if man.get('corrections'):
        raw = (data_dir / man['corrections']).read_bytes()
        try:
            import brotli
            raw = brotli.decompress(raw)
        except Exception:
            pass
        arr = np.frombuffer(raw, dtype=np.int8).reshape(-1, 6).astype(np.float32)
        scales = np.array(man['corr_scales'], dtype=np.float32)
        corr = arr * scales  # (600,6): dx,dy,rot,zoom,bias,gain

    chain = man.get('chain', 'grid512')
    segfix = {}
    if man.get('segfix'):
        import brotli as _br
        blob = _br.decompress((data_dir / man['segfix']).read_bytes())
        D = np.array(man['segfix_dirtable'], dtype=np.float32).reshape(25, 3)
        AMPS = man.get('segfix_amps', [6, 12, 18])
        p = 0
        while p < len(blob):
            k = blob[p] | (blob[p + 1] << 8)
            cnt = blob[p + 2]
            p += 3
            acts = []
            for _ in range(cnt):
                t = blob[p] | (blob[p + 1] << 8)
                di, ai = blob[p + 2], blob[p + 3]
                p += 4
                acts.append((t, di, ai))
            segfix[k] = [(t, np.round(D[di] * AMPS[ai]).astype(np.int16)) for (t, di, ai) in acts]
    names = [ln.strip() for ln in file_list.read_text().splitlines() if ln.strip()]
    for name in names:
        base = name.rsplit('.', 1)[0]
        stream = data_dir / man['streams'][base]
        dst = out_dir / f'{base}.raw'
        skip = man.get('skip_first', 0)
        i = -skip
        with open(dst, 'wb') as f:
            for y, u, v, bits in decode_yuv_frames(stream):
                if i < 0:
                    i += 1
                    continue
                x = yuv_to_rgb_f32(y, u, v, bits)
                if chain == 'hybrid2':
                    # evens: grid-placed metric-space pose warp (as searched).
                    # odds WITH seg-fix entries: rounded+edited+grid-placed (as searched);
                    # odds WITHOUT: direct decoded output (float world, as searched).
                    k = i // 2
                    if i % 2 == 1:
                        if k not in segfix:
                            f.write(x.round().clamp(0, 255).to(torch.uint8).numpy().tobytes())
                            i += 1
                            continue
                        xm = downsample_512(x.round().clamp(0, 255), 'bilinear')
                        t512 = xm.round().clamp(0, 255).to(torch.uint8)
                        a = t512.numpy().astype(np.int16)
                        for (t, delta) in segfix[k]:
                            ty, tx = divmod(t, 32)
                            a[ty * 16:(ty + 1) * 16, tx * 16:(tx + 1) * 16] = np.clip(
                                a[ty * 16:(ty + 1) * 16, tx * 16:(tx + 1) * 16] + delta, 0, 255)
                        t512 = torch.from_numpy(a.astype(np.uint8))
                    else:
                        xm = downsample_512(x.round().clamp(0, 255), 'bilinear')
                        if corr is not None and k < len(corr):
                            xm = apply_correction(xm, *corr[k])
                        t512 = xm.round().clamp(0, 255).to(torch.uint8)
                    f.write(gridplace_fullres(t512).tobytes())
                elif chain == 'hybrid':
                    # odd frames: direct decoded output (zero-floor for SegNet+PoseNet)
                    # even frames: correction applied in metric space exactly as searched,
                    # then exact-grid placed (metric reads it verbatim)
                    if i % 2 == 1 or corr is None or i // 2 >= len(corr):
                        f.write(x.round().clamp(0, 255).to(torch.uint8).numpy().tobytes())
                    else:
                        xm = downsample_512(x.round().clamp(0, 255), 'bilinear')
                        xm = apply_correction(xm, *corr[i // 2])
                        t512 = xm.round().clamp(0, 255).to(torch.uint8)
                        f.write(gridplace_fullres(t512).tobytes())
                elif chain == 'fullres':
                    # corrections searched in metric space; scale translation to full-res
                    if corr is not None and i % 2 == 0 and i // 2 < len(corr):
                        dx, dy, rot, zoom, bias, gain = corr[i // 2]
                        x = apply_correction(x, dx * W / w, dy * H / h, rot, zoom, bias, gain)
                    f.write(x.round().clamp(0, 255).to(torch.uint8).numpy().tobytes())
                else:
                    if x.shape[0] != h:
                        x = downsample_512(x, man.get('dec_down', 'bilinear'))
                    if corr is not None and i % 2 == 0 and i // 2 < len(corr):
                        x = apply_correction(x, *corr[i // 2])
                    t512 = x.round().clamp(0, 255).to(torch.uint8)
                    f.write(gridplace_fullres(t512).tobytes())
                i += 1
        print(f"inflated {base}: {i} frames -> {dst}")

if __name__ == '__main__':
    main()
