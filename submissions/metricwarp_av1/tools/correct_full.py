#!/usr/bin/env python
"""Correction search for the full-res chain.

1) Stream-decode the 874p bitstream, cache A-sampled float16 metric frames.
2) Per-pair search (reuses correct2.search_pair) on float frames.
3) Verify pass: apply warp at 874p, A-sample, recompute pose; report drift.

Usage: python correct_full.py BITSTREAM OUT.npz [--threads 8] [--cache CACHE.npy]
"""
import sys, argparse, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import torch.nn.functional as F
from correct2 import search_pair, eval_candidates
import correct2

H, W = 874, 1164
h, w = 384, 512

def build_cache(bitstream, cache_path):
    import av
    frames = np.empty((1200, h, w, 3), dtype=np.float16)
    fmt = 'obu' if str(bitstream).endswith('.obu') else None
    container = av.open(str(bitstream), format=fmt)
    stream = container.streams.video[0]
    i = 0
    for frame in container.decode(stream):
        name = frame.format.name
        bits = 10 if '10' in name else 8
        dt = np.uint16 if bits == 10 else np.uint8
        esz = 2 if bits == 10 else 1
        fh, fw = frame.height, frame.width
        y = np.frombuffer(frame.planes[0], dtype=dt).reshape(fh, frame.planes[0].line_size // esz)[:, :fw]
        u = np.frombuffer(frame.planes[1], dtype=dt).reshape(fh // 2, frame.planes[1].line_size // esz)[:, :fw // 2]
        v = np.frombuffer(frame.planes[2], dtype=dt).reshape(fh // 2, frame.planes[2].line_size // esz)[:, :fw // 2]
        scale = float(1 << (bits - 8))
        Yf = torch.from_numpy(y.astype(np.float32)) / scale
        Uf = F.interpolate((torch.from_numpy(u.astype(np.float32)) / scale)[None, None], size=(fh, fw),
                           mode='bilinear', align_corners=False)[0, 0]
        Vf = F.interpolate((torch.from_numpy(v.astype(np.float32)) / scale)[None, None], size=(fh, fw),
                           mode='bilinear', align_corners=False)[0, 0]
        R = Yf + 1.402 * (Vf - 128.0)
        G = Yf - 0.344136 * (Uf - 128.0) - 0.714136 * (Vf - 128.0)
        B = Yf + 1.772 * (Uf - 128.0)
        rgb = torch.stack([R, G, B], dim=-1).clamp(0, 255).round()  # uint8 output contract of inflate
        # A-sample to metric space (float, no rounding)
        samp = F.interpolate(rgb.permute(2, 0, 1)[None], size=(h, w), mode='bilinear', align_corners=False)
        frames[i] = samp[0].permute(1, 2, 0).numpy().astype(np.float16)
        i += 1
    container.close()
    assert i == 1200, i
    np.save(cache_path, frames)
    return frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bitstream')
    ap.add_argument('out')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--cache', default=None)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=600)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cache_path = Path(args.cache) if args.cache else HERE / (Path(args.bitstream).stem + '_A.npy')
    if cache_path.exists():
        frames = np.load(cache_path, mmap_mode='r')
    else:
        t0 = time.time()
        frames = build_cache(args.bitstream, cache_path)
        print(f"cache built in {time.time()-t0:.0f}s", flush=True)

    gt_pose = np.load(HERE / 'gt_pose.npy')
    P = np.zeros((600, 6), dtype=np.float32)
    M = np.zeros((600, 2), dtype=np.float32)
    t0 = time.time()
    for k in range(args.start, args.end):
        e = torch.from_numpy(np.asarray(frames[2 * k], dtype=np.float32))
        o = torch.from_numpy(np.asarray(frames[2 * k + 1], dtype=np.float32))
        cur, best, base = search_pair(e, o, gt_pose[k])
        P[k] = [cur['dx'], cur['dy'], cur['rot'], cur['zoom'], cur['bias'], cur['gain']]
        M[k] = [base, best]
        if (k - args.start) % 20 == 19:
            done = k - args.start + 1
            el = time.time() - t0
            print(f"{done} pairs {el:.0f}s ({el/done:.1f}s/pair) "
                  f"mean {M[args.start:k+1,0].mean():.5f}->{M[args.start:k+1,1].mean():.5f}", flush=True)
    np.savez(args.out, params=P, mse=M, start=args.start, end=args.end)
    a, b = M[args.start:args.end, 0].mean(), M[args.start:args.end, 1].mean()
    print(f"saved {args.out}; base {a:.6f} (term {(10*a)**0.5:.4f}) -> corrected {b:.6f} (term {(10*b)**0.5:.4f})")

if __name__ == '__main__':
    main()
