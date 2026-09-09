"""HNeRV training, restructured for a 2-core CPU budget.

Departures from the reference recipe, all forced by compute:
  * PoseNet runs on every pair every step (cheap: 0.11 s/pair, and it is the
    sensitive term); SegNet runs on a rotating subset (expensive: 0.50 s/pair).
  * The inflate->harness resize round trip is applied as two precomputed dense
    matrices instead of a literal 3M-pixel bicubic: exact to 3e-7, 5x faster.
    Training *through* it lets the decoder pre-compensate for the resampling.
  * Latents are initialised from a PCA of the ground-truth frames rather than
    randomly, so the code is meaningful (and temporally smooth, hence cheap to
    delta-code) from step 0.
  * Wider latents than the reference's 28-d: latent bytes are cheap relative to
    what they buy in convergence speed when steps, not capacity, are the limit.
"""
from __future__ import annotations
import argparse, json, math, time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
torch.set_num_threads(2)
from common import (CACHE, WORK, load_distortion_net, build_roundtrip, apply_roundtrip,
                    Decoder, build_archive, parse_archive, score_of, EVAL_H, EVAL_W)


# ---------------------------------------------------------------- losses
def seg_loss(logits, target, tau=0.3, mode="l7", hard_mult=4.0, hard_thresh=1.0,
             blend=0.25):
    """Margin loss on SegNet logits.

    'smooth' = sigmoid(-margin/tau): gradient peaks exactly at the decision
    boundary.  Correct near convergence, but it SATURATES — measured on this
    model the median disagreeing pixel sits at margin -6.4 and only 7.8% of
    wrong pixels are within one tau of the boundary, so ~92% of the pixels the
    metric charges for receive no gradient at all.

    'l7' = tau*softplus(-margin/tau), reweighted toward small margins: gradient
    magnitude tends to 1 as margin -> -inf, so deeply misclassified pixels keep
    pulling.  Blending a little 'smooth' back in keeps pressure on the boundary
    pixels that are about to flip.
    """
    tgt = logits.gather(1, target.unsqueeze(1))
    masked = logits.masked_fill(
        F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).bool(), -1e9)
    margin = tgt - masked.max(dim=1, keepdim=True)[0]

    if hard_mult > 0:
        with torch.no_grad():
            w = 1.0 + hard_mult * (margin < hard_thresh).float()
            w = w / w.mean()
    else:
        w = 1.0

    smooth = (torch.sigmoid(-margin / tau) * w).mean()
    if mode == "smooth":
        return smooth
    l7 = (tau * F.softplus(-margin / tau) * w).mean()
    return l7 + blend * smooth


def fake_quant(t, n=127):
    m = t.abs().max()
    s = m / n if m > 0 else t.new_tensor(1.0)
    q = (t / s).round().clamp(-n, n)
    return (q * s - t).detach() + t


def apply_qat(dec, n=127):
    orig = {}
    for name, m in dec.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            orig[name] = m.weight.data.clone()
            m.weight.data = fake_quant(m.weight.data, n)
    return orig


def restore_qat(dec, orig):
    for name, m in dec.named_modules():
        if name in orig:
            m.weight.data = orig[name]


# ---------------------------------------------------------------- data
class Targets:
    def __init__(self):
        self.seg = np.load(CACHE / "seg_targets.npy", mmap_mode="r")
        self.gt = np.load(CACHE / "gt384.npy", mmap_mode="r")
        self.pose = torch.from_numpy(np.load(CACHE / "pose_targets.npy"))
        self.n = self.pose.shape[0]

    def batch(self, idx):
        gt = torch.from_numpy(np.ascontiguousarray(self.gt[idx])).float()
        seg = torch.from_numpy(np.ascontiguousarray(self.seg[idx])).long()
        return gt, seg, self.pose[idx]


def init_base_image(dec, tg, n=200):
    """Seed the shared base image with the logit of the mean frame, so step 0
    already predicts the average scene instead of grey."""
    m = torch.from_numpy(np.ascontiguousarray(tg.gt[:n])).float().mean((0, 1)) / 255.0
    m = m.clamp(1e-3, 1 - 1e-3)
    logit = torch.log(m / (1 - m)).unsqueeze(0)
    dec.base.data = F.interpolate(logit, size=dec.base.shape[-2:],
                                  mode="bilinear", align_corners=False)


def pca_init_latents(tg, dim, seed=0):
    """Low-res grayscale of each pair -> PCA -> per-pair code."""
    g = torch.from_numpy(np.ascontiguousarray(tg.gt[:, :, :, ::8, ::8])).float()
    g = g.mean(2).flatten(1)                                  # (N, 2*48*64)
    g = (g - g.mean(0)) / (g.std(0) + 1e-6)
    q = torch.linalg.svd(g - g.mean(0), full_matrices=False)
    z = q.U[:, :dim] * q.S[:dim]
    z = z / (z.std(0, keepdim=True) + 1e-6) * 0.5
    if dim > z.shape[1]:
        pad = torch.randn(z.shape[0], dim - z.shape[1], generator=torch.Generator().manual_seed(seed)) * 0.1
        z = torch.cat([z, pad], 1)
    return z.contiguous()


# ---------------------------------------------------------------- eval
@torch.inference_mode()
def true_score(dec, lat, net, tg, Ah, Aw, idx, batch=4):
    """Real metric on the given pair indices (no surrogates)."""
    segs, poses = [], []
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        out = apply_roundtrip(dec(lat[b]), Ah, Aw)
        out = out.clamp(0, 255).round().permute(0, 1, 3, 4, 2)
        pin, sin = net.preprocess_input(out)
        sl = net.segnet(sin)
        pl = net.posenet(pin)["pose"][:, :6]
        tgt_seg = torch.from_numpy(np.ascontiguousarray(tg.seg[b])).long().to(sl.device)
        segs.append((sl.argmax(1) != tgt_seg).float().mean(dim=(1, 2)))
        poses.append((pl - tg.pose[b].to(pl.device)).square().mean(1))
    return torch.cat(segs).mean().item(), torch.cat(poses).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--latent-dim", type=int, default=96)
    ap.add_argument("--base-channels", type=int, default=40)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--seg-frac", type=float, default=0.25, help="fraction of a batch that gets the (expensive) SegNet loss")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--latent-lr-mult", type=float, default=10.0)
    ap.add_argument("--pixel-w", type=float, default=1.0)
    ap.add_argument("--seg-w", type=float, default=100.0)
    ap.add_argument("--seg-loss", default="l7", choices=["l7", "smooth"])
    ap.add_argument("--seg-tau", type=float, default=0.3)
    ap.add_argument("--seg-blend", type=float, default=0.25)
    ap.add_argument("--pose-w", type=float, default=1.0)
    ap.add_argument("--pixel-decay-min", type=float, default=15, help="minutes over which pixel weight decays to 0")
    ap.add_argument("--warm-pixel-min", type=float, default=8, help="pixel-only warmup minutes (no metric nets)")
    ap.add_argument("--qat-after", type=float, default=0.7, help="fraction of budget after which INT8 QAT turns on")
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--eval-every-min", type=float, default=10)
    ap.add_argument("--eval-pairs", type=int, default=64)
    ap.add_argument("--resume", default="")
    ap.add_argument("--amp", type=int, default=1, help="bf16/AMX for decoder trunk + segnet grads")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = WORK / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "train.log", "a")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); log.write(s + "\n"); log.flush()

    dev = torch.device(args.device if args.device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cpu":
        torch.set_num_threads(args.threads)
    P(f"device: {dev}")

    tg = Targets()
    net = load_distortion_net(dev)
    Ah, Aw = build_roundtrip()
    Ah, Aw = Ah.to(dev), Aw.to(dev)

    dec = Decoder(latent_dim=args.latent_dim, base_channels=args.base_channels)
    init_base_image(dec, tg)
    lat = nn.Parameter(pca_init_latents(tg, args.latent_dim))
    if args.resume:
        rdir = WORK / "runs" / args.resume
        ckpt = rdir / "last.pt"
        if not ckpt.exists():
            ckpt = rdir / "best.pt"          # shipped checkpoints only carry best
        ck = torch.load(ckpt, map_location="cpu")
        dec.load_state_dict(ck["dec"]); lat = nn.Parameter(ck["lat"])
        P(f"resumed from {ckpt}")
    dec = dec.to(dev)
    lat = nn.Parameter(lat.data.to(dev))
    tg.pose = tg.pose.to(dev)

    n_par = sum(p.numel() for p in dec.parameters())
    P(f"[{args.name}] decoder {n_par:,} par, latent {args.latent_dim}d x {tg.n} = {lat.numel():,}")

    opt = torch.optim.AdamW([
        {"params": dec.parameters(), "lr": args.lr},
        {"params": [lat], "lr": args.lr * args.latent_lr_mult},
    ], weight_decay=0.0)

    ema_dec, ema_lat = deepcopy(dec), lat.data.clone()
    eval_idx = np.linspace(0, tg.n - 1, args.eval_pairs).astype(int)

    budget = args.minutes * 60
    t0 = time.time()
    best = {"score": float("inf")}
    step = 0
    next_eval = args.eval_every_min * 60
    perm = np.random.permutation(tg.n); pcur = 0

    while True:
        el = time.time() - t0
        if el > budget:
            break
        frac = el / budget
        # cosine LR
        for g, base_lr in zip(opt.param_groups, (args.lr, args.lr * args.latent_lr_mult)):
            g["lr"] = base_lr * max(0.5 * (1 + math.cos(math.pi * frac)), 0.02)

        if pcur + args.batch > tg.n:
            perm = np.random.permutation(tg.n); pcur = 0
        idx = perm[pcur:pcur + args.batch]; pcur += args.batch

        gt, seg_t, pose_t = tg.batch(idx)
        gt, seg_t = gt.to(dev, non_blocking=True), seg_t.to(dev, non_blocking=True)
        warm = el < args.warm_pixel_min * 60
        use_qat = frac > args.qat_after

        if use_qat:
            orig = apply_qat(dec)
        raw = dec(lat[torch.from_numpy(idx).to(dev)], amp=args.amp)
        rt = apply_roundtrip(raw, Ah, Aw)
        if use_qat:
            restore_qat(dec, orig)

        pw = args.pixel_w * max(0.0, 1.0 - el / (args.pixel_decay_min * 60)) if not warm else args.pixel_w
        loss = 0.0
        parts = {}
        if pw > 0:
            l_pix = F.l1_loss(rt, gt) / 255.0
            loss = loss + pw * l_pix
            parts["pix"] = l_pix.item()

        if not warm:
            clamped = rt.clamp(0, 255)
            ste = clamped + (clamped.round() - clamped).detach()
            bhwc = ste.permute(0, 1, 3, 4, 2)
            pin, sin = net.preprocess_input(bhwc)

            pose_out = net.posenet(pin)["pose"][:, :6]
            pose_mse = F.mse_loss(pose_out, pose_t)
            l_pose = torch.sqrt(10.0 * pose_mse + 1e-12)
            loss = loss + args.pose_w * l_pose
            parts["pose"] = pose_mse.item()

            k = 0 if args.seg_frac <= 0 else max(1, int(round(args.seg_frac * len(idx))))
            if k > 0:
                sub = torch.randperm(len(idx))[:k]
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(args.amp)):
                    sl = net.segnet(sin[sub])
                l_seg = seg_loss(sl.float(), seg_t[sub], tau=args.seg_tau, mode=args.seg_loss, blend=args.seg_blend)
                loss = loss + args.seg_w * l_seg
                parts["seg"] = l_seg.item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(dec.parameters()) + [lat], 1.0)
        opt.step()

        with torch.no_grad():
            d = args.ema
            for ep, p in zip(ema_dec.parameters(), dec.parameters()):
                ep.data.mul_(d).add_(p.data, alpha=1 - d)
            ema_lat.mul_(d).add_(lat.data, alpha=1 - d)

        step += 1
        if step % 50 == 0:
            P(f"  step {step:6d} t={el/60:6.1f}m frac={frac:.2f} " +
              " ".join(f"{k}={v:.5f}" for k, v in parts.items()) +
              f" lr={opt.param_groups[0]['lr']:.2e}")

        if el > next_eval or el > budget:
            next_eval = el + args.eval_every_min * 60
            eval_dec = deepcopy(ema_dec).eval()
            arch = build_archive({k: v.cpu() for k, v in eval_dec.state_dict().items()}, ema_lat.cpu(),
                                 {"latent_dim": args.latent_dim, "base_channels": args.base_channels,
                                  "n_pairs": int(tg.n)})
            sd, lt, _ = parse_archive(arch)
            qdec = Decoder(latent_dim=args.latent_dim, base_channels=args.base_channels).eval()
            qdec.load_state_dict(sd); qdec = qdec.to(dev)
            seg_d, pose_d = true_score(qdec, lt.to(dev), net, tg, Ah, Aw, eval_idx)
            sc = score_of(seg_d, pose_d, len(arch))
            P(f"  >>> t={el/60:.1f}m  score={sc:.4f}  seg={seg_d:.6f} pose={pose_d:.8f} "
              f"archive={len(arch):,}  (rate term {25*len(arch)/37545489:.4f})")
            torch.save({"dec": {k: v.cpu() for k, v in ema_dec.state_dict().items()},
                        "lat": ema_lat.cpu()}, out_dir / "last.pt")
            if sc < best["score"]:
                best = {"score": sc, "seg": seg_d, "pose": pose_d, "bytes": len(arch),
                        "minutes": el / 60, "step": step}
                (out_dir / "best_archive.bin").write_bytes(arch)
                torch.save({"dec": {k: v.cpu() for k, v in ema_dec.state_dict().items()},
                            "lat": ema_lat.cpu()}, out_dir / "best.pt")
                json.dump(best | vars(args), open(out_dir / "best.json", "w"), indent=2, default=str)

    P(f"[{args.name}] BEST {best}")


if __name__ == "__main__":
    main()
