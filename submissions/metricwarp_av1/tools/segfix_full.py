#!/usr/bin/env python
"""Production seg-fix: greedy per-tile integer RGB nudges on odd frames (uint8 domain,
batched SegNet evaluation). Deterministically reproducible at decode.

Usage: python segfix_full.py FRAMES_512.raw OUT.npz [--start 0] [--end 600]
Tile grid: 16x16 px -> 24 rows x 32 cols (id = ty*32+tx). Directions: cmeans[c_gt]-cmeans[c_pr]
normalized, indexed by (c_gt*5+c_pr). Amps: {6,12,18}.
"""
import sys, argparse, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from metric_lib import load_net, setup_threads

T = 16
TH, TW = 384 // T, 512 // T
AMPS = (6, 12, 18)

_net = None
def net():
    global _net
    if _net is None:
        _net = load_net()
    return _net

@torch.inference_mode()
def seg_argmax_batch(frames_f32):
    """frames_f32: (B,384,512,3) float tensor -> (B,384,512) uint8 argmax"""
    n = net()
    x = frames_f32.permute(0, 3, 1, 2).unsqueeze(1)  # (B,1,3,H,W)
    sin = n.segnet.preprocess_input(x)
    return n.segnet(sin).argmax(dim=1).to(torch.uint8)

def class_means(tgt, gt_seg):
    cm = np.zeros((5, 3), dtype=np.float32)
    for c in range(5):
        vals = []
        for k in range(0, 600, 60):
            m = gt_seg[k] == c
            if m.any():
                vals.append(tgt[2 * k + 1][m].mean(axis=0))
        cm[c] = np.mean(vals, axis=0) if vals else 128
    return cm

def dir_table(cm):
    """(25,3) float unit directions for (c_gt*5+c_pr)."""
    D = np.zeros((25, 3), dtype=np.float32)
    for g in range(5):
        for p in range(5):
            d = cm[g] - cm[p]
            n_ = np.linalg.norm(d)
            D[g * 5 + p] = d / n_ if n_ > 1 else np.array([1, 1, 1]) / np.sqrt(3)
    return D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames')
    ap.add_argument('out')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=600)
    ap.add_argument('--tiles', type=int, default=14)
    ap.add_argument('--min-gain', type=int, default=12)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    setup_threads(args.threads)

    frames = np.memmap(args.frames, dtype=np.uint8, mode='r').reshape(-1, 384, 512, 3)
    gt_seg = np.load(HERE / 'gt_seg.npy')
    tgt = np.memmap(HERE / 'target_512.raw', dtype=np.uint8, mode='r').reshape(-1, 384, 512, 3)
    cm = class_means(tgt, gt_seg)
    D = dir_table(cm)
    np.save(HERE / 'segfix_dirtable.npy', D)

    actions_all = {}
    t0 = time.time()
    tot0 = tot1 = 0
    for k in range(args.start, args.end):
        fidx = 2 * k + 1
        cur = frames[fidx].astype(np.float32)
        curt = torch.from_numpy(cur)
        seg = seg_argmax_batch(curt.unsqueeze(0))[0].numpy()
        flips = seg != gt_seg[k]
        n0 = int(flips.sum())
        cur_flips = n0
        acts = []
        tf = flips.reshape(TH, T, TW, T).sum(axis=(1, 3))
        order = np.argsort(tf.ravel())[::-1][:args.tiles]
        for t in order:
            ty, tx = divmod(int(t), TW)
            if tf[ty, tx] < args.min_gain:
                break
            y0, x0 = ty * T, tx * T
            sub_gt = gt_seg[k][y0:y0 + T, x0:x0 + T]
            sub_pr = seg[y0:y0 + T, x0:x0 + T]
            wrong = sub_gt != sub_pr
            if not wrong.any():
                continue
            c_gt = int(np.bincount(sub_gt[wrong].ravel(), minlength=5).argmax())
            c_pr = int(np.bincount(sub_pr[wrong].ravel(), minlength=5).argmax())
            di = c_gt * 5 + c_pr
            # integer deltas per amp, applied in uint8 domain (exact decode reproduction)
            cands = []
            for a_i, amp in enumerate(AMPS):
                delta = np.round(D[di] * amp).astype(np.int16)
                v = cur.copy()
                v[y0:y0 + T, x0:x0 + T] = np.clip(v[y0:y0 + T, x0:x0 + T] + delta, 0, 255)
                cands.append(v)
            segs = seg_argmax_batch(torch.from_numpy(np.stack(cands))).numpy()
            nf = [(s != gt_seg[k]).sum() for s in segs]
            best = int(np.argmin(nf))
            gain = cur_flips - int(nf[best])
            if gain > args.min_gain:
                cur = cands[best]
                seg = segs[best]
                cur_flips = int(nf[best])
                acts.append((int(t), di, best))
        if acts:
            actions_all[k] = acts
        tot0 += n0; tot1 += cur_flips
        if (k - args.start) % 25 == 24:
            done = k - args.start + 1
            el = time.time() - t0
            print(f"{done} frames {el:.0f}s ({el/done:.1f}s/f) flips {tot0}->{tot1} (-{(tot0-tot1)/max(tot0,1)*100:.0f}%)", flush=True)
    # serialize: per frame with actions: [k_lo,k_hi,count, then per-action tile_lo,tile_hi,di,amp]
    blob = bytearray()
    for k in sorted(actions_all):
        acts = actions_all[k]
        blob += bytes([k & 0xFF, k >> 8, len(acts)])
        for (t, di, a) in acts:
            blob += bytes([t & 0xFF, t >> 8, di, a])
    np.savez(args.out, blob=np.frombuffer(bytes(blob), dtype=np.uint8),
             dirtable=D, start=args.start, end=args.end,
             stats=np.array([tot0, tot1]))
    print(f"saved {args.out}: {len(actions_all)} frames with actions, blob {len(blob)}B, "
          f"flips {tot0}->{tot1} ({(tot0-tot1)/max(tot0,1)*100:.0f}% fixed)")

if __name__ == '__main__':
    main()
