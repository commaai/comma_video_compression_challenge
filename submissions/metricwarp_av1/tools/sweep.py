#!/usr/bin/env python
"""Encode target_512.raw with a codec config, decode, score vs cached GT.

Usage: python sweep.py --enc svt --crf 40 [--bits 8|10] [--stride 8] [--tag mylabel] [--extra "k=v:k=v"]
Appends results to work/results.csv.
"""
import argparse, subprocess, time, csv, os
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FFMPEG = str(ROOT / 'tools/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg')
import sys
sys.path.insert(0, str(HERE))
from rgbyuv import rgb_to_yuv, decode_video_rgb
from fastscore import score_frames

h, w = 384, 512
GT_H, GT_W = 874, 1164

def prep_target(eh, ew, aa=True):
    """Cache RGB target at encode resolution (bilinear from gt.raw, rounded)."""
    if (eh, ew) == (h, w):
        return HERE / 'target_512.raw'
    import torch
    import torch.nn.functional as F
    out = HERE / f'target_{ew}x{eh}{"" if aa else "_noaa"}.raw'
    if out.exists():
        return out
    gt = np.memmap(HERE / 'gt.raw', dtype=np.uint8, mode='r').reshape(-1, GT_H, GT_W, 3)
    with open(out, 'wb') as f:
        for i in range(0, len(gt), 32):
            x = torch.from_numpy(np.ascontiguousarray(gt[i:i+32])).permute(0, 3, 1, 2).float()
            y = F.interpolate(x, size=(eh, ew), mode='bilinear', align_corners=False, antialias=aa)
            f.write(y.round().clamp(0, 255).permute(0, 2, 3, 1).to(torch.uint8).numpy().tobytes())
    return out

def prep_yuv(bits, css='420', eh=h, ew=w, pack=False, tgt_path=None):
    """Cache YUV planes file for encoder input. pack: stack (even,odd) vertically -> (N/2, 2*eh, ew)."""
    stem = Path(tgt_path).stem + '_' if tgt_path else ''
    out = HERE / f'{stem}yuv{bits}_{css}_{ew}x{eh}{"_pack" if pack else ""}.yuv'
    if out.exists():
        return out
    tgt = np.memmap(tgt_path or prep_target(eh, ew), dtype=np.uint8, mode='r').reshape(-1, eh, ew, 3)
    if pack:
        tgt = tgt.reshape(-1, 2 * eh, ew, 3)
    with open(out, 'wb') as f:
        for i in range(0, len(tgt), 100):
            Y, U, V = rgb_to_yuv(np.ascontiguousarray(tgt[i:i+100]), bits=bits, subsample=(css == '420'))
            for j in range(len(Y)):
                f.write(Y[j].tobytes()); f.write(U[j].tobytes()); f.write(V[j].tobytes())
    return out

def encode(enc, crf, bits, extra, preset, gop, outpath, css='420', eh=h, ew=w, pack=False, tgt_path=None):
    yuv = prep_yuv(bits, css, eh, ew, pack=pack, tgt_path=tgt_path)
    pix = f'yuv{css}p10le' if bits == 10 else f'yuv{css}p'
    ih, fps = (2 * eh, '10') if pack else (eh, '20')
    base = [FFMPEG, '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', pix, '-s', f'{ew}x{ih}', '-r', fps, '-i', str(yuv)]
    if enc in ('svt', 'svtpar'):
        params = f'tune=1:film-grain=0:scd=0:enable-qm=1:qm-min=0:keyint={gop}'
        if enc == 'svtpar':
            params += ':hierarchical-levels=2'
            dup = yuv.with_name(yuv.stem + '_dup.yuv')
            if not dup.exists():
                fb = (ew * ih * 3) // 2 * (2 if bits == 10 else 1)
                with open(yuv, 'rb') as src, open(dup, 'wb') as dst:
                    first = src.read(fb)
                    dst.write(first); dst.write(first)
                    while True:
                        buf = src.read(1 << 24)
                        if not buf:
                            break
                        dst.write(buf)
            base[base.index(str(yuv))] = str(dup)
        if extra:
            params += ':' + extra
        cmd = base + ['-c:v', 'libsvtav1', '-preset', str(preset), '-crf', str(crf),
                      '-svtav1-params', params, '-f', 'ivf', str(outpath)]
    elif enc == 'x265':
        params = f'keyint={gop}:min-keyint={gop}:scenecut=0:psy-rd=0:psy-rdoq=0:aq-mode=2:bframes=8:log-level=warning'
        if extra:
            params += ':' + extra
        cmd = base + ['-c:v', 'libx265', '-preset', preset if isinstance(preset, str) else 'slow',
                      '-crf', str(crf), '-x265-params', params, '-f', 'hevc', str(outpath)]
    elif enc == 'x265par':
        # parity-QP: prepend dup frame so fixed PbPb pattern puts b on even originals;
        # crf arg = QP for odd originals (refs), extra = QP for even originals (b frames)
        qp_lo = int(crf)
        use_factor = float(extra) < 1.0 if extra else False
        qp_hi = (qp_lo + 8) if (not extra or use_factor) else int(extra)
        dup = yuv.with_name(yuv.stem + '_dup.yuv')
        if not dup.exists():
            fb = (ew * ih * 3) // 2 * (2 if bits == 10 else 1)
            with open(yuv, 'rb') as src, open(dup, 'wb') as dst:
                first = src.read(fb)
                dst.write(first)
                dst.write(first)
                while True:
                    buf = src.read(1 << 24)
                    if not buf:
                        break
                    dst.write(buf)
        zones = []
        n_in = 1201
        for k in range(n_in):
            if use_factor:
                zones.append(f"{k},{k},b={1.0 if (k == 0 or k % 2 == 0) else float(extra):g}")
            else:
                zones.append(f"{k},{k},q={qp_lo if (k == 0 or k % 2 == 0) else qp_hi}")
        zstr = '/'.join(zones)
        params = (f'keyint=300:min-keyint=300:scenecut=0:psy-rd=0:psy-rdoq=0:aq-mode=2:'
                  f'bframes=1:b-adapt=0:log-level=warning:zones={zstr}')
        if use_factor:
            cmd_crf = ['-crf', str(crf)]
        else:
            cmd_crf = []
        base[base.index(str(yuv))] = str(dup)
        cmd = base + ['-c:v', 'libx265', '-preset', preset if isinstance(preset, str) else 'slow'] + \
              cmd_crf + ['-x265-params', params, '-f', 'hevc', str(outpath)]
    elif enc == 'aom':
        cmd = base + ['-c:v', 'libaom-av1', '-crf', str(crf), '-b:v', '0',
                      '-cpu-used', str(preset), '-g', str(gop), '-aq-mode', '1',
                      '-enable-cdef', '1', '-row-mt', '1', '-tiles', '1x1', '-f', 'ivf', str(outpath)]
        if extra:
            cmd += extra.split()
    else:
        raise ValueError(enc)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0

def decode_downsample_stream(path, mode):
    """Memory-lean: decode frames one at a time, downsample to metric res immediately."""
    import torch
    import torch.nn.functional as F
    from rgbyuv import decode_video_yuv, yuv420_to_rgb
    import av
    out = []
    container = av.open(str(path))
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        name = frame.format.name
        bits = 10 if '10' in name else 8
        dt = np.uint16 if bits == 10 else np.uint8
        esz = 2 if bits == 10 else 1
        fh, fw = frame.height, frame.width
        y = np.frombuffer(frame.planes[0], dtype=dt).reshape(fh, frame.planes[0].line_size // esz)[:, :fw]
        u = np.frombuffer(frame.planes[1], dtype=dt).reshape(fh // 2, frame.planes[1].line_size // esz)[:, :fw // 2]
        v = np.frombuffer(frame.planes[2], dtype=dt).reshape(fh // 2, frame.planes[2].line_size // esz)[:, :fw // 2]
        rgb = yuv420_to_rgb(y[None], u[None], v[None], bits=bits)  # (1,fh,fw,3) uint8
        x = torch.from_numpy(rgb.astype(np.float32)).permute(0, 3, 1, 2)
        if mode == 'bilinear':
            z = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        elif mode == 'area':
            z = F.interpolate(x, size=(h, w), mode='area')
        else:
            raise ValueError(mode)
        out.append(z.round().clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).numpy())
    container.close()
    return np.stack(out)

def downsample_to_metric(frames, mode, chunk=64):
    import torch
    import torch.nn.functional as F
    out = np.empty((len(frames), h, w, 3), dtype=np.uint8)
    for s in range(0, len(frames), chunk):
        x = torch.from_numpy(np.ascontiguousarray(frames[s:s+chunk])).permute(0, 3, 1, 2).float()
        if mode == 'bilinear':
            y = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        elif mode == 'area':
            y = F.interpolate(x, size=(h, w), mode='area')
        elif mode == 'bilinear-aa':
            y = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False, antialias=True)
        elif mode == 'bicubic-aa':
            y = F.interpolate(x, size=(h, w), mode='bicubic', align_corners=False, antialias=True)
        out[s:s+chunk] = y.round().clamp(0, 255).permute(0, 2, 3, 1).to(torch.uint8).numpy()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', required=True, choices=['svt', 'svtpar', 'x265', 'x265par', 'aom'])
    ap.add_argument('--crf', type=float, required=True)
    ap.add_argument('--css', default='420', choices=['420', '444'])
    ap.add_argument('--bits', type=int, default=8)
    ap.add_argument('--preset', default=None)
    ap.add_argument('--gop', type=int, default=300)
    ap.add_argument('--extra', default='')
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--tag', default='')
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--enc-size', default='512x384', help='WxH encode resolution')
    ap.add_argument('--dec-down', default='bilinear', choices=['bilinear', 'area', 'bilinear-aa', 'bicubic-aa'],
                    help='decode-side downsample to 512x384')
    ap.add_argument('--pack', action='store_true', help='pack pairs vertically (2*eh frames at 10fps)')
    ap.add_argument('--tgt', default=None, help='override target raw path (512x384 only)')
    args = ap.parse_args()

    preset = args.preset
    if preset is None:
        preset = {'svt': 3, 'svtpar': 3, 'x265': 'slow', 'x265par': 'slow', 'aom': 4}[args.enc]

    ew, eh = map(int, args.enc_size.split('x'))
    name = f"{args.enc}_crf{args.crf:g}_b{args.bits}_{args.css}_p{preset}_g{args.gop}_{ew}x{eh}"
    if ew != w:
        name += f"_d{args.dec_down}"
    if args.pack:
        name += '_pack'
    if args.tgt:
        name += '_' + Path(args.tgt).stem.replace('target_512_', '')
    if args.tag:
        name += '_' + args.tag
    outpath = HERE / 'enc' / (name + ('.ivf' if args.enc in ('svt', 'svtpar', 'aom') else '.hevc'))
    outpath.parent.mkdir(exist_ok=True)

    enc_t = encode(args.enc, args.crf, args.bits, args.extra, preset, args.gop, outpath,
                   css=args.css, eh=eh, ew=ew, pack=args.pack, tgt_path=args.tgt)
    size = outpath.stat().st_size

    t0 = time.time()
    if eh > 700 and not args.pack:
        frames = decode_downsample_stream(outpath, args.dec_down)
        if args.enc in ('x265par', 'svtpar'):
            frames = frames[1:]
    else:
        frames = decode_video_rgb(outpath)
        if args.enc in ('x265par', 'svtpar'):
            frames = frames[1:]
        if args.pack:
            frames = frames.reshape(-1, eh, ew, 3)
        if frames.shape[1] != h:
            frames = downsample_to_metric(frames, args.dec_down)
    dec_t = time.time() - t0
    assert frames.shape == (1200, h, w, 3), frames.shape

    r = score_frames(frames, stride=args.stride, archive_size=size)
    row = dict(name=name, enc=args.enc, crf=args.crf, bits=args.bits, css=args.css, preset=str(preset),
               gop=args.gop, extra=args.extra, enc_size=args.enc_size, dec_down=args.dec_down,
               size=size, stride=args.stride,
               pose=round(r['pose_dist'], 8), seg=round(r['seg_dist'], 8),
               pose_term=round(r['pose_term'], 4), seg_term=round(r['seg_term'], 4),
               rate_term=round(r['rate_term'], 4), score=round(r['score'], 5),
               enc_s=round(enc_t, 1), dec_s=round(dec_t, 1))
    csvp = HERE / 'results.csv'
    exists = csvp.exists()
    with open(csvp, 'a', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            wr.writeheader()
        wr.writerow(row)
    print(' '.join(f'{k}={v}' for k, v in row.items()))
    if not args.keep:
        outpath.unlink()

if __name__ == '__main__':
    main()
