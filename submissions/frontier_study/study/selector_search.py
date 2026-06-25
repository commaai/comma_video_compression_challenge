#!/usr/bin/env python
"""Per-pair scorer-exploit search (the lever that moved #101->#110).

On top of the FIXED int8 base reconstruction, search a palette of cheap per-pair
frame perturbations against the EXACT frozen SegNet+PoseNet, and pick the best
per pair. First reports the ORACLE (rate-free) distortion as an upper bound.

SegNet sees only frame-1; PoseNet sees the (f0,f1) pair (6-d MSE). So frame-0
perturbations affect pose only; frame-1 perturbations affect seg+pose.

Score = 100*seg + sqrt(10*pose) + 25*rate. seg is separable per-pair; the pose
term is sqrt(mean), so we select per-pair with a linearized pose weight
w = sqrt(10)/(2*sqrt(mean_pose)) and iterate the mean a couple times.
"""
from __future__ import annotations

import sys, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "work")
import beat_top as B

CAMERA_H, CAMERA_W = B.CAMERA_H, B.CAMERA_W


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- palette ---------------------------------------------------------------
def build_palette():
    """List of (name, frame_idx, fn). fn maps a (b,3,H,W) float frame -> perturbed."""
    pal = [("identity", -1, lambda x: x)]

    def luma(k):
        return lambda x: x + k

    def chan(c, k):
        def f(x):
            y = x.clone(); y[:, c] = y[:, c] + k; return y
        return f

    def roll(dy, dx):
        return lambda x: torch.roll(x, shifts=(dy, dx), dims=(2, 3))

    specs = []
    for k in (-4, -3, -2, -1, 1, 2, 3, 4):
        specs.append((f"luma{k:+d}", luma(k)))
    for c, cn in ((0, "r"), (1, "g"), (2, "b")):
        for k in (-3, -2, -1, 1, 2, 3):
            specs.append((f"{cn}{k:+d}", chan(c, k)))
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)):
        specs.append((f"roll{dy}{dx}", roll(dy, dx)))

    for fi in (0, 1):                      # apply to frame 0 or frame 1
        for name, fn in specs:
            pal.append((f"f{fi}_{name}", fi, fn))
    return pal


# ---- evaluation of one candidate over all pairs ----------------------------
@torch.inference_mode()
def search(device_name="mps", batch=16):
    dev = B.get_device(device_name)
    log(f"device={dev}")
    net = B.build_net(dev)
    if dev.type == "mps":
        B.patch_bn_contiguous(net)
    base = B.extract_base()
    seg_gt, pose_gt = B.compute_gt(net, dev)
    seg_gt = seg_gt.to(dev).long()
    pose_gt = pose_gt.to(dev).float()
    meta = base["meta"]
    dec = B.make_decoder(meta, dev); dec.load_state_dict(base["decoder_sd"]); dec.eval()
    latents = base["latents"]
    eh, ew = dec.eval_size
    n_pairs = latents.shape[0]

    pal = build_palette()
    K = len(pal)
    log(f"palette size K={K}, n_pairs={n_pairs}")

    # per-candidate per-pair distortions
    seg_d = np.zeros((K, n_pairs), dtype=np.float64)
    pose_d = np.zeros((K, n_pairs), dtype=np.float64)

    t0 = time.time()
    for bi, i in enumerate(range(0, n_pairs, batch)):
        z = latents[i:i + batch].to(dev)
        b = z.shape[0]
        # base reconstruction at camera res, float (b,2,3,H,W)
        decoded = dec(z)
        flat = decoded.reshape(b * 2, 3, eh, ew)
        up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False).clamp(0, 255)
        base_frames = up.reshape(b, 2, 3, CAMERA_H, CAMERA_W)  # float

        for c, (name, fi, fn) in enumerate(pal):
            fr = base_frames.clone()
            if fi >= 0:
                fr[:, fi] = fn(fr[:, fi]).clamp(0, 255)
            frames = fr.round().to(torch.uint8).permute(0, 1, 3, 4, 2).contiguous()  # (b,2,H,W,3)
            po, so = net(frames)
            sp = so.argmax(1)
            seg_d[c, i:i + b] = (sp != seg_gt[i:i + b]).float().mean((1, 2)).cpu().numpy()
            pose_d[c, i:i + b] = (po["pose"][..., :6] - pose_gt[i:i + b]).pow(2).mean(1).cpu().numpy()
        if bi % 5 == 0:
            log(f"batch {bi} (pair {i}) [{time.time()-t0:.0f}s]")

    np.savez("work/selector_dists.npz", seg_d=seg_d, pose_d=pose_d,
             names=np.array([p[0] for p in pal]))
    log("saved work/selector_dists.npz")

    # ---- report base (identity) and oracle (free per-pair best) ----
    base_seg = seg_d[0].mean()
    base_pose = pose_d[0].mean()
    base_score_dist = 100 * base_seg + math.sqrt(10 * base_pose)
    log(f"BASE     seg={base_seg:.6f}(100*={100*base_seg:.4f}) "
        f"pose={base_pose:.6f}(sqrt={math.sqrt(10*base_pose):.4f}) distortion={base_score_dist:.4f}")

    # iterate linearized pose weight
    mean_pose = base_pose
    for _ in range(4):
        w = math.sqrt(10) / (2 * math.sqrt(max(mean_pose, 1e-9)))
        J = 100 * seg_d + w * pose_d            # (K, n_pairs)
        choice = J.argmin(0)                    # (n_pairs,)
        sel_seg = seg_d[choice, np.arange(n_pairs)].mean()
        sel_pose = pose_d[choice, np.arange(n_pairs)].mean()
        mean_pose = sel_pose
    oracle_dist = 100 * sel_seg + math.sqrt(10 * sel_pose)
    log(f"ORACLE   seg={sel_seg:.6f}(100*={100*sel_seg:.4f}) "
        f"pose={sel_pose:.6f}(sqrt={math.sqrt(10*sel_pose):.4f}) distortion={oracle_dist:.4f}")
    log(f"distortion gain vs base: {base_score_dist - oracle_dist:.4f}  "
        f"(fec6 selector achieved ~0.073 total distortion vs muon ~0.080)")

    # index distribution + crude rate estimate (brotli of index byte stream)
    import brotli
    idx_bytes = choice.astype(np.uint8).tobytes()
    sel_bytes = len(brotli.compress(idx_bytes, quality=11))
    uniq, cnt = np.unique(choice, return_counts=True)
    nz = (choice != 0).sum()
    log(f"selector: {nz}/{n_pairs} pairs non-identity; ~{sel_bytes} bytes (brotli of indices)")
    log("ALL DONE")


if __name__ == "__main__":
    search(sys.argv[1] if len(sys.argv) > 1 else "mps")
