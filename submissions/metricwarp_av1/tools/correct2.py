#!/usr/bin/env python
"""Production correction search: per-pair even-frame affine+bias+gain vs PoseNet, batched.

Usage: python correct2.py FRAMES_512.raw OUT.npz [--start 0] [--end 600] [--threads 4]
FRAMES can be uint8 raw (1200,384,512,3).
"""
import sys, argparse, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import torch.nn.functional as F
from metric_lib import load_net, setup_threads

_net = None
def net():
    global _net
    if _net is None:
        _net = load_net()
    return _net

def make_thetas(params):
    """params: list of dicts with dx,dy,rot(mrad),zoom(1e-3). -> (B,2,3) theta for affine_grid (H=384,W=512)."""
    H, W = 384, 512
    ths = []
    for p in params:
        r = p.get('rot', 0.0) * 1e-3
        z = 1.0 + p.get('zoom', 0.0) * 1e-3
        cos, sin = np.cos(r), np.sin(r)
        ths.append([[cos / z, -sin / z * H / W, -p.get('dx', 0.0) * 2 / W],
                    [sin / z * W / H, cos / z, -p.get('dy', 0.0) * 2 / H]])
    return torch.tensor(ths, dtype=torch.float32)

@torch.inference_mode()
def eval_candidates(even_f32, odd_f32, gt_vec, params):
    """Evaluate B candidate corrections of the even frame. Returns (B,) pose MSEs."""
    B = len(params)
    H, W = 384, 512
    x = even_f32.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)  # (B,3,H,W)
    needs_warp = any(p.get('dx') or p.get('dy') or p.get('rot') or p.get('zoom') for p in params)
    if needs_warp:
        theta = make_thetas(params)
        grid = F.affine_grid(theta, (B, 3, H, W), align_corners=False)
        xw = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
    else:
        xw = x.clone()
    bias = torch.tensor([p.get('bias', 0.0) for p in params], dtype=torch.float32).view(B, 1, 1, 1)
    gain = torch.tensor([1.0 + p.get('gain', 0.0) * 1e-3 for p in params], dtype=torch.float32).view(B, 1, 1, 1)
    # round: decode grid-places the corrected even frame as uint8, so optimize post-rounding
    xw = (xw * gain + bias).clamp(0, 255).round()
    odd = odd_f32.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
    pair = torch.stack([xw, odd], dim=1)  # (B,2,3,H,W)
    n = net()
    pin = n.posenet.preprocess_input(pair)
    out = n.posenet(pin)['pose'][:, :6]
    gt = torch.from_numpy(gt_vec).unsqueeze(0)
    return ((out - gt) ** 2).mean(dim=1).numpy()

def search_pair(even_f32, odd_f32, gt_vec, hard_threshold=0.002):
    cur = dict(dx=0.0, dy=0.0, rot=0.0, zoom=0.0, bias=0.0, gain=0.0)

    def try_set(cands_list):
        nonlocal cur, best
        mses = eval_candidates(even_f32, odd_f32, gt_vec, cands_list)
        i = int(np.argmin(mses))
        if mses[i] < best:
            best = float(mses[i])
            cur = dict(cands_list[i])
        return best

    base = float(eval_candidates(even_f32, odd_f32, gt_vec, [cur])[0])
    best = base
    # stage 1: joint coarse dx,dy
    cands = [dict(cur, dx=a, dy=b) for a in (-2, -1, 0, 1, 2) for b in (-2, -1, 0, 1, 2)]
    try_set(cands)
    # stage 2: fine dx,dy around best
    cands = [dict(cur, dx=cur['dx'] + a, dy=cur['dy'] + b)
             for a in (-0.5, -0.25, 0, 0.25, 0.5) for b in (-0.5, -0.25, 0, 0.25, 0.5)]
    try_set(cands)
    # stage 3+: per-param refine, 2 passes
    for _ in range(2):
        for k, deltas in [('rot', (-2, -1, -0.5, 0.5, 1, 2)), ('zoom', (-4, -2, -1, 1, 2, 4)),
                          ('bias', (-1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5)), ('gain', (-8, -4, -2, 2, 4, 8)),
                          ('dx', (-0.25, -0.125, 0.125, 0.25)), ('dy', (-0.25, -0.125, 0.125, 0.25))]:
            cands = [dict(cur, **{k: cur[k] + d}) for d in deltas]
            try_set(cands)
    # hard-pair extension: wider translation + zoom/rot range
    if best > hard_threshold:
        cands = [dict(cur, dx=cur['dx'] + a, dy=cur['dy'] + b)
                 for a in (-4, -3, -2, 2, 3, 4) for b in (-2, 0, 2)]
        cands += [dict(cur, zoom=cur['zoom'] + z) for z in (-12, -8, 8, 12)]
        cands += [dict(cur, rot=cur['rot'] + r) for r in (-6, -4, 4, 6)]
        try_set(cands)
        for _ in range(2):
            for k, deltas in [('dx', (-1, -0.5, 0.5, 1)), ('dy', (-1, -0.5, 0.5, 1)),
                              ('zoom', (-2, -1, 1, 2)), ('rot', (-1, 1)),
                              ('bias', (-0.5, 0.5)), ('gain', (-4, 4))]:
                cands = [dict(cur, **{k: cur[k] + d}) for d in deltas]
                try_set(cands)
    return cur, best, base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames')
    ap.add_argument('out')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=600)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    frames = np.memmap(args.frames, dtype=np.uint8, mode='r').reshape(-1, 384, 512, 3)
    gt_pose = np.load(HERE / 'gt_pose.npy')
    P = np.zeros((600, 6), dtype=np.float32)
    M = np.zeros((600, 2), dtype=np.float32)
    t0 = time.time()
    for k in range(args.start, args.end):
        e = torch.from_numpy(frames[2 * k].astype(np.float32))
        o = torch.from_numpy(frames[2 * k + 1].astype(np.float32))
        cur, best, base = search_pair(e, o, gt_pose[k])
        P[k] = [cur['dx'], cur['dy'], cur['rot'], cur['zoom'], cur['bias'], cur['gain']]
        M[k] = [base, best]
        if (k - args.start) % 20 == 19:
            done = k - args.start + 1
            el = time.time() - t0
            print(f"{done} pairs {el:.0f}s ({el/done:.1f}s/pair) mean {M[args.start:k+1,0].mean():.5f}->{M[args.start:k+1,1].mean():.5f}", flush=True)
    np.savez(args.out, params=P, mse=M, start=args.start, end=args.end)
    print(f"saved {args.out}; base {M[args.start:args.end,0].mean():.6f} -> corrected {M[args.start:args.end,1].mean():.6f}")

if __name__ == '__main__':
    main()
