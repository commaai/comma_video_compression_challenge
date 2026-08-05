#!/usr/bin/env python
"""Fast scorer: evaluate a candidate .raw against cached GT metric outputs.

Usage: python score.py CANDIDATE.raw [--stride N] [--archive-size BYTES] [--per-pair OUT.npz] [--batch 8]
"""
import argparse, math, time
import numpy as np
from pathlib import Path
from metric_lib import load_net, raw_pairs, net_outputs, setup_threads

HERE = Path(__file__).resolve().parent
ORIG_SIZE = 37545489

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate')
    ap.add_argument('--stride', type=int, default=1, help='evaluate every Nth pair')
    ap.add_argument('--archive-size', type=int, default=0)
    ap.add_argument('--per-pair', type=str, default=None, help='save per-pair distortions to this .npz')
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()

    setup_threads()
    gt_pose = np.load(HERE / 'gt_pose.npy')   # (600,6)
    gt_seg = np.load(HERE / 'gt_seg.npy')     # (600,384,512)
    n_pairs = len(gt_pose)
    pair_idx = np.arange(0, n_pairs, args.stride)

    net = load_net()
    pose_d = np.zeros(len(pair_idx))
    seg_d = np.zeros(len(pair_idx))
    t0 = time.time()
    done = 0
    for pos, (chunk, batch) in enumerate(raw_pairs(args.candidate, batch_size=args.batch, pair_indices=pair_idx)):
        pose, seg = net_outputs(net, batch)
        sl = slice(done, done + len(chunk))
        pose_d[sl] = ((pose - gt_pose[chunk]) ** 2).mean(axis=1)
        seg_d[sl] = (seg != gt_seg[chunk]).mean(axis=(1, 2))
        done += len(chunk)
    el = time.time() - t0

    pose_dist = pose_d.mean()
    seg_dist = seg_d.mean()
    rate = args.archive_size / ORIG_SIZE
    score = 100 * seg_dist + math.sqrt(10 * pose_dist) + 25 * rate
    print(f"pairs evaluated: {len(pair_idx)} (stride {args.stride}), {el:.1f}s")
    print(f"posenet_dist: {pose_dist:.8f}  (term {math.sqrt(10*pose_dist):.4f})")
    print(f"segnet_dist:  {seg_dist:.8f}  (term {100*seg_dist:.4f})")
    print(f"rate:         {rate:.8f}  (term {25*rate:.4f})  [{args.archive_size:,} bytes]")
    print(f"SCORE: {score:.5f}")
    if args.per_pair:
        np.savez(args.per_pair, pair_idx=pair_idx, pose_d=pose_d, seg_d=seg_d)

if __name__ == '__main__':
    main()
