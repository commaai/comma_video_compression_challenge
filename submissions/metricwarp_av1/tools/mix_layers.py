#!/usr/bin/env python
"""Per-pair mix of seg-fix vs no-seg-fix worlds.

For each pair k choose:
  A (no segfix): odd = float decode (corrB world), pose mse = corrB[k]
  B (segfix):    odd = rounded+edited (corrC world), pose mse = corrC[k]
by comparing seg term gain vs pose term cost around the mixed operating point.

Outputs mixed segfix npz (kept frames only) + mixed corr npz (per-pair params from the
matching search) + report.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from segfix_full import seg_argmax_batch
from metric_lib import setup_threads

def parse_blob(blob, D, AMPS=(6, 12, 18)):
    acts = {}
    p = 0
    while p < len(blob):
        k = blob[p] | (blob[p + 1] << 8); cnt = blob[p + 2]; p += 3
        a = []
        for _ in range(cnt):
            t = blob[p] | (blob[p + 1] << 8); di, ai = blob[p + 2], blob[p + 3]; p += 4
            a.append((t, di, ai, np.round(D[di] * AMPS[ai]).astype(np.int16)))
        acts[k] = a
    return acts

def main():
    setup_threads()
    corrB = np.load(HERE / 'corrB_final.npz')
    corrC = np.load(HERE / 'corrC_final.npz')
    sf = np.load(HERE / 'segfix_final.npz')
    D = sf['dirtable']
    acts = parse_blob(sf['blob'].tobytes(), D)
    gt_seg = np.load(HERE / 'gt_seg.npy')

    cacheA = np.load(HERE / 'finalA_A.npy', mmap_mode='r')      # float odds (world A)
    cacheC = np.load(HERE / 'finalA_A_v3.npy', mmap_mode='r')   # edited odds (world B)

    # seg flips per odd frame in both worlds (batch through segnet)
    flipsA = np.zeros(600, dtype=np.int32)
    flipsB = np.zeros(600, dtype=np.int32)
    for s in range(0, 600, 8):
        ks = range(s, min(s + 8, 600))
        fa = torch.from_numpy(np.stack([np.asarray(cacheA[2 * k + 1], dtype=np.float32) for k in ks]))
        fb = torch.from_numpy(np.stack([np.asarray(cacheC[2 * k + 1], dtype=np.float32) for k in ks]))
        sa = seg_argmax_batch(fa).numpy()
        sb = seg_argmax_batch(fb).numpy()
        for j, k in enumerate(ks):
            flipsA[k] = (sa[j] != gt_seg[k]).sum()
            flipsB[k] = (sb[j] != gt_seg[k]).sum()
        if s % 200 == 0:
            print(f"{s}/600 flips computed", flush=True)

    mseA = corrB['mse'][:, 1].astype(np.float64)
    mseB = corrC['mse'][:, 1].astype(np.float64)
    NPIX = 384 * 512

    keep = np.zeros(600, dtype=bool)
    for _ in range(6):
        m = np.where(keep, mseB, mseA).mean()
        w = np.sqrt(10.0) / (2.0 * np.sqrt(m)) / 600.0     # d(pose term)/d(pair mse)
        seg_gain = 100.0 / 600.0 * (flipsA - flipsB) / NPIX  # score gain from segfix
        pose_cost = w * (mseB - mseA)
        keep_new = seg_gain > pose_cost
        if (keep_new == keep).all():
            break
        keep = keep_new

    segA = flipsA.sum() / NPIX / 600
    seg_mix = np.where(keep, flipsB, flipsA).sum() / NPIX / 600
    pose_mix = np.where(keep, mseB, mseA).mean()
    poseA = mseA.mean()
    print(f"kept segfix on {keep.sum()}/600 pairs")
    print(f"seg: allA {segA:.6f} -> mixed {seg_mix:.6f} (term {100*seg_mix:.4f})")
    print(f"pose: allA {poseA:.6f} (term {np.sqrt(10*poseA):.4f}) -> mixed {pose_mix:.6f} (term {np.sqrt(10*pose_mix):.4f})")

    # mixed segfix blob
    blob = bytearray()
    for k in sorted(acts):
        if not keep[k]:
            continue
        a = acts[k]
        blob += bytes([k & 0xFF, k >> 8, len(a)])
        for (t, di, ai, _) in a:
            blob += bytes([t & 0xFF, t >> 8, di, ai])
    np.savez(HERE / 'segfix_mixed.npz', blob=np.frombuffer(bytes(blob), dtype=np.uint8),
             dirtable=D, start=0, end=600, stats=np.array([flipsA.sum(), int(np.where(keep, flipsB, flipsA).sum())]))

    P = np.where(keep[:, None], corrC['params'], corrB['params']).astype(np.float32)
    M = np.where(keep[:, None], corrC['mse'], corrB['mse']).astype(np.float32)
    np.savez(HERE / 'corr_mixed.npz', params=P, mse=M, start=0, end=600)
    print(f"mixed blob {len(blob)}B; wrote segfix_mixed.npz + corr_mixed.npz")

if __name__ == '__main__':
    main()
