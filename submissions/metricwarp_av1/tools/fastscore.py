#!/usr/bin/env python
"""Score candidate frames (metric-space 384x512 or full-res 874x1164) vs cached GT outputs."""
import math
import numpy as np
import torch
from pathlib import Path
from metric_lib import load_net, net_outputs, setup_threads

HERE = Path(__file__).resolve().parent
ORIG_SIZE = 37545489

_net = None
_gt = None

def get_net():
    global _net
    if _net is None:
        setup_threads()
        _net = load_net()
    return _net

def get_gt():
    global _gt
    if _gt is None:
        _gt = (np.load(HERE / 'gt_pose.npy'), np.load(HERE / 'gt_seg.npy'))
    return _gt

def score_frames(frames, stride=1, batch=8, archive_size=0, per_pair=False, pair_indices=None):
    """frames: uint8 array (N, H, W, 3) with H,W in {(384,512),(874,1164)}. N must be even.

    Returns dict with pose_dist, seg_dist, score(with given archive_size), per-pair arrays."""
    net = get_net()
    gt_pose, gt_seg = get_gt()
    n_pairs_total = len(gt_pose)
    if pair_indices is None:
        pair_indices = np.arange(0, n_pairs_total, stride)
    else:
        pair_indices = np.asarray(pair_indices)
    pose_d = np.zeros(len(pair_indices))
    seg_d = np.zeros(len(pair_indices))
    for start in range(0, len(pair_indices), batch):
        chunk = pair_indices[start:start + batch]
        b = np.stack([frames[2 * i:2 * i + 2] for i in chunk])
        pose, seg = net_outputs(net, torch.from_numpy(b))
        sl = slice(start, start + len(chunk))
        pose_d[sl] = ((pose - gt_pose[chunk]) ** 2).mean(axis=1)
        seg_d[sl] = (seg != gt_seg[chunk]).mean(axis=(1, 2))
    pose_dist = pose_d.mean()
    seg_dist = seg_d.mean()
    rate = archive_size / ORIG_SIZE
    score = 100 * seg_dist + math.sqrt(10 * pose_dist) + 25 * rate
    out = dict(pose_dist=pose_dist, seg_dist=seg_dist, rate=rate, score=score,
               pose_term=math.sqrt(10 * pose_dist), seg_term=100 * seg_dist, rate_term=25 * rate,
               n_pairs=len(pair_indices))
    if per_pair:
        out['pair_indices'] = pair_indices
        out['pose_d'] = pose_d
        out['seg_d'] = seg_d
    return out

def load_raw(path):
    path = Path(path)
    sz = path.stat().st_size
    for (H, W) in [(384, 512), (874, 1164)]:
        fb = H * W * 3
        if sz % fb == 0:
            n = sz // fb
            if n in (1200, 1199, 1198):
                return np.memmap(path, dtype=np.uint8, mode='r', shape=(n, H, W, 3))
    raise ValueError(f"cannot infer shape from size {sz}")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--archive-size', type=int, default=0)
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()
    frames = load_raw(args.candidate)
    r = score_frames(frames, stride=args.stride, batch=args.batch, archive_size=args.archive_size)
    print(f"pairs: {r['n_pairs']}  pose_dist {r['pose_dist']:.8f} (term {r['pose_term']:.4f})  "
          f"seg_dist {r['seg_dist']:.8f} (term {r['seg_term']:.4f})  rate_term {r['rate_term']:.4f}  SCORE {r['score']:.5f}")
