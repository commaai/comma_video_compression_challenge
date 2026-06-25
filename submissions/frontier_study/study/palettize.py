#!/usr/bin/env python
"""Weight palettization vs int8+brotli for the HNeRV decoder.

Thesis: muon's C1a regularizer clusters weights at few distinct values, so a
K-centroid palette + entropy-coded indices may be near-lossless yet smaller than
the current int8(per-tensor)+brotli baseline (162740 bytes, 5.69 bits/wt).

Steps:
  1. Diagnostic: distinct int8 values per weight tensor (the palettization ceiling).
  2. PTQ palettization sweep (per-tensor 1D k-means) over K: size (brotli of
     centroids+indices) + faithful score.
"""
from __future__ import annotations
import sys, io, struct, math, time
import numpy as np, torch, brotli

sys.path.insert(0, "work")
import beat_top as B


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def kmeans_1d(x, K, iters=25):
    """Simple 1D Lloyd k-means. x: 1D np.float32. Returns (centroids[K], labels)."""
    x = x.astype(np.float64)
    if np.unique(x).size <= K:
        cents = np.unique(x)
        labels = np.searchsorted(cents, x)
        return cents.astype(np.float32), labels.astype(np.int64), cents.size
    # init: K quantiles
    qs = np.linspace(0, 100, K)
    cents = np.percentile(x, qs)
    cents = np.unique(cents)
    for _ in range(iters):
        # assign
        d = np.abs(x[:, None] - cents[None, :])
        labels = d.argmin(1)
        # update
        new = np.array([x[labels == k].mean() if np.any(labels == k) else cents[k]
                        for k in range(len(cents))])
        if np.allclose(new, cents):
            cents = new; break
        cents = new
    d = np.abs(x[:, None] - cents[None, :]); labels = d.argmin(1)
    return cents.astype(np.float32), labels.astype(np.int64), len(cents)


def int8_baseline(sd):
    """per-tensor int8 zigzag+brotli (the current top regime)."""
    buf = io.BytesIO(); buf.write(struct.pack("<I", len(sd)))
    deq = {}
    for name, t in sd.items():
        t = t.detach().cpu().float()
        m = t.abs().max().item(); s = m / 127 if m > 0 else 1.0
        q = (t / s).round().clamp(-127, 127).to(torch.int8).numpy().flatten()
        deq[name] = torch.from_numpy(q.astype(np.float32)).reshape(t.shape) * s
        zz = np.where(q.astype(np.int32) >= 0, 2 * q, -2 * q - 1).astype(np.uint8)
        buf.write(zz.tobytes())
    return len(brotli.compress(buf.getvalue(), quality=11)), deq


def palettize(sd, K):
    """per-tensor K-centroid palette. Returns (size_bytes, deq_sd, avg_bits)."""
    buf = io.BytesIO(); total_w = 0; total_idx_bits = 0.0
    deq = {}
    for name, t in sd.items():
        arr = t.detach().cpu().float().numpy().flatten()
        cents, labels, kk = kmeans_1d(arr, K)
        deq[name] = torch.from_numpy(cents[labels].reshape(t.shape).astype(np.float32))
        buf.write(struct.pack("<H", kk))
        buf.write(cents.astype(np.float16).tobytes())
        buf.write(labels.astype(np.uint8).tobytes())  # K<=256 -> 1 byte; brotli packs it
        # shannon entropy of indices (theoretical index cost)
        _, cnt = np.unique(labels, return_counts=True)
        p = cnt / cnt.sum(); H = -(p * np.log2(p)).sum()
        total_idx_bits += H * labels.size; total_w += labels.size
    size = len(brotli.compress(buf.getvalue(), quality=11))
    return size, deq, total_idx_bits / max(total_w, 1)


def main():
    dev = B.get_device("mps")
    net = B.build_net(dev)
    if dev.type == "mps": B.patch_bn_contiguous(net)
    seg_gt, pose_gt = B.compute_gt(net, dev)
    base = B.extract_base(); sd = base["decoder_sd"]; meta = base["meta"]
    lat_bytes = len(brotli.compress(B.encode_latents(base["latents"]), quality=11))
    nparam = sum(v.numel() for v in sd.values())
    log(f"params={nparam}  lat_bytes={lat_bytes}")

    # 1) distinct int8 values per tensor (the ceiling)
    log("distinct int8 values per weight tensor:")
    for name, t in sd.items():
        if not name.endswith(".weight"):
            continue
        t = t.detach().cpu().float()
        m = t.abs().max().item(); s = m / 127 if m > 0 else 1.0
        q = (t / s).round().clamp(-127, 127).to(torch.int8).numpy()
        log(f"   {name:14s} numel={t.numel():6d} distinct_int8={np.unique(q).size}")

    # 2) int8 baseline
    b_size, b_deq = int8_baseline(sd)
    dec = B.make_decoder(meta, dev); dec.load_state_dict(b_deq)
    r = B.score(net, dec, base["latents"], dev, b_size + lat_bytes + B.META_BYTES, seg_gt, pose_gt)
    log(f"INT8 baseline: dec={b_size} archive={b_size+lat_bytes+B.META_BYTES} "
        f"bits/wt={b_size*8/nparam:.2f}  seg={r['seg']:.6f} SCORE={r['score']:.4f}")

    # 3) palettization sweep
    for K in [64, 32, 16]:
        t0 = time.time()
        size, deq, idx_bits = palettize(sd, K)
        dec = B.make_decoder(meta, dev); dec.load_state_dict(deq)
        r = B.score(net, dec, base["latents"], dev, size + lat_bytes + B.META_BYTES, seg_gt, pose_gt)
        log(f"PALETTE K={K:3d}: dec={size} archive={size+lat_bytes+B.META_BYTES} "
            f"bits/wt={size*8/nparam:.2f} (idxH={idx_bits:.2f})  "
            f"seg={r['seg']:.6f}(100*={100*r['seg']:.4f}) pose={r['pose']:.6f} "
            f"SCORE={r['score']:.4f}  [{time.time()-t0:.0f}s]")
    log("ALL DONE")


if __name__ == "__main__":
    main()
