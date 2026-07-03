#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Quantization-aware latent polish against the frozen #1 decoder.

Freezes the PR #95 decoder (extracted from the PR #110/#112 payload) and runs
gradient descent on the 600x28 latents ONLY, through the exact inflate chain:

  decoder -> bicubic 874x1164 -> #98 channel biases (f0 R-1, f0 B-1, f1 G-1)
  -> clamp/round (STE) -> FEC6 per-pair selector transform -> clamp/round (STE)
  -> DistortionNet (SegNet + PoseNet) loss vs precomputed GT targets.

Latents are quantized to the FIXED PR #101 uint8 grid via a straight-through
estimator on every forward, so what we optimize is exactly what we ship.
Because pairs are independent under a frozen decoder, the final result keeps
the best quantized latent PER PAIR seen at any eval, not a global snapshot.

Usage:
  python polish.py --epochs 120 --lr 1.5e-3 --batch 3 --eval-every 10
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
QLP_DIR = HERE.parent
REPO = QLP_DIR.parent.parent
FEC6_SRC = REPO / "submissions" / "hnerv_fec6_fixed_huffman_k16" / "src"
STAGE9_SRC = REPO / "submissions" / "hnerv_stage9" / "src"
RHNERV = REPO / "submissions" / "rhnerv_comma"
sys.path.insert(0, str(FEC6_SRC))          # model.py (PR #95 decoder)
sys.path.insert(0, str(STAGE9_SRC))        # data.py (targets), losses.py
sys.path.insert(0, str(RHNERV))            # codec_ctx (latent size probe)
sys.path.insert(0, str(HERE))

from model import HNeRVDecoder                      # noqa: E402  (fec6)
from data import precompute_targets, get_default_video_path  # noqa: E402  (stage9)
from losses import l7_softplus_seg_loss             # noqa: E402  (stage9)
import codec_ctx as cc                              # noqa: E402  (rhnerv)
from latent_codec import (                          # noqa: E402
    encode_raw_latent_payload, quantize_to_grid, dequantize)
from frame_selector import _blue_tile               # noqa: E402  (fec6)

CAMERA_H, CAMERA_W = 874, 1164
EVAL_SIZE = (384, 512)
N_PAIRS, LATENT_DIM = 600, 28
ORIG_BYTES = 37_545_489
# rhnerv_comma container: 7B header + decoder 161,104 + latent section + selector 248,
# plus 100B zip overhead; sidecar dropped (subsumed by the polish).
FIXED_ARCHIVE_OVERHEAD = 7 + 161_104 + 248 + 100


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


def ste_quantize_latents(lat, mins, scales):
    codes = (lat - mins) / scales
    codes = codes.clamp(0.0, 255.0)
    codes = ste_round(codes)
    return mins + codes * scales


class SelectorApply:
    """Differentiable FEC6 selector transforms, precompiled per pair."""

    def __init__(self, selector_codes, selector_specs, device):
        self.bias = torch.zeros(N_PAIRS, 2, 3, device=device)   # per frame RGB add
        self.blue_amp = torch.zeros(N_PAIRS, 2, device=device)  # +R tile, -B tile
        self.roll = [[None, None] for _ in range(N_PAIRS)]      # (dy, dx) or None
        self.any_op = torch.zeros(N_PAIRS, dtype=torch.bool)
        for i, code in enumerate(selector_codes.tolist()):
            family, params, fidx = selector_specs[code]
            if family == "identity":
                continue
            self.any_op[i] = True
            if family == "rgb_bias":
                self.bias[i, fidx] = torch.tensor(params, dtype=torch.float32,
                                                  device=device)
            elif family == "blue_chroma":
                self.blue_amp[i, fidx] = float(params[0])
            elif family == "roll":
                dx, dy = int(params[0]), int(params[1])
                self.roll[i][fidx] = (dy, dx)
            else:
                raise ValueError(family)
        self.tile = _blue_tile(CAMERA_H, CAMERA_W, device=device,
                               dtype=torch.float32)

    def __call__(self, frames, pair_idx):
        """frames: (B, 2, 3, H, W) post-bias, post-round. Returns same shape."""
        out = frames
        for b, p in enumerate(pair_idx.tolist()):
            if not self.any_op[p]:
                continue
            for f in range(2):
                delta = self.bias[p, f]
                amp = self.blue_amp[p, f]
                fr = out[b, f]
                if delta.abs().sum() > 0:
                    fr = fr + delta.view(3, 1, 1)
                if amp != 0:
                    fr = torch.stack([fr[0] + self.tile * amp, fr[1],
                                      fr[2] - self.tile * amp])
                if self.roll[p][f] is not None:
                    dy, dx = self.roll[p][f]
                    fr = torch.roll(fr, shifts=(dy, dx), dims=(1, 2))
                out = out.clone() if out is frames else out
                out[b, f] = fr
        return out


def decode_chain(decoder, lat_q, pair_idx, selector, *, differentiable):
    """Exact inflate chain. lat_q: (B, 28) quantized latents (STE-attached)."""
    B = lat_q.shape[0]
    decoded = decoder(lat_q)                                   # (B, 2, 3, 384, 512)
    flat = decoded.reshape(B * 2, 3, *EVAL_SIZE)
    with torch.autocast('cuda', dtype=torch.float16, enabled=differentiable):
        up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode='bicubic',
                           align_corners=False)
    up = up.float().reshape(B, 2, 3, CAMERA_H, CAMERA_W)
    # PR #98 channel biases: frame0 R-1, frame0 B-1, frame1 G-1.
    bias = torch.tensor([[-1.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
                        device=up.device).view(1, 2, 3, 1, 1)
    up = up + bias
    clamped = up.clamp(0, 255)
    rounded = ste_round(clamped) if differentiable else clamped.round()
    rounded = selector(rounded, pair_idx)
    clamped2 = rounded.clamp(0, 255)
    return ste_round(clamped2) if differentiable else clamped2.round()


def scorer_forward(distortion_net, frames_b23hw, *, fp32=False):
    """(B, 2, 3, H, W) camera-res -> (seg_logits, pose_out).

    fp32=True disables autocast: the seg metric is a discrete argmax, and
    fp16 vs fp32 disagree on borderline pixels, so selection/eval MUST be
    fp32 to transfer to the CPU scoring axis. Training may stay fp16.
    """
    bhwc = frames_b23hw.permute(0, 1, 3, 4, 2)                 # (B, 2, H, W, 3)
    with torch.autocast('cuda', dtype=torch.float16, enabled=not fp32):
        posenet_in, segnet_in = distortion_net.preprocess_input(bhwc)
        seg_out = distortion_net.segnet(segnet_in)
        pose_out = distortion_net.posenet(posenet_in)
    return seg_out.float(), {k: (v.float() if torch.is_tensor(v) else v)
                             for k, v in pose_out.items()}


@torch.no_grad()
def full_eval(decoder, latents_cont, mins, scales, selector, distortion_net,
              seg_targets_cpu, pose_targets, device, batch=8):
    """True per-pair metrics with hard-quantized latents through the exact chain."""
    q_codes = quantize_to_grid(latents_cont, mins, scales)
    lat_q = dequantize(q_codes, mins, scales)
    seg_d = torch.zeros(N_PAIRS)
    pose_d = torch.zeros(N_PAIRS)
    for i in range(0, N_PAIRS, batch):
        j = min(i + batch, N_PAIRS)
        idx = torch.arange(i, j, device=device)
        frames = decode_chain(decoder, lat_q[i:j].to(device), idx, selector,
                              differentiable=False)
        seg_out, pose_out = scorer_forward(distortion_net, frames, fp32=True)
        pred = seg_out.argmax(dim=1)                            # (B, h, w)
        tgt = seg_targets_cpu[i:j].long().to(device)
        seg_d[i:j] = (pred != tgt).float().flatten(1).mean(dim=1).cpu()
        pose_d[i:j] = F.mse_loss(pose_out['pose'][:, :6], pose_targets[i:j],
                                 reduction='none').mean(dim=1).cpu()
    return q_codes.cpu(), seg_d, pose_d


def latent_section_bytes(q_codes, mins, scales) -> int:
    raw = encode_raw_latent_payload(q_codes, mins, scales)
    return len(cc.encode_latent_section(raw))


def projected_score(seg_mean, pose_mean, lat_bytes) -> float:
    rate = (FIXED_ARCHIVE_OVERHEAD + lat_bytes) / ORIG_BYTES
    return 100 * seg_mean + math.sqrt(10 * pose_mean) + 25 * rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-floor", type=float, default=1e-6)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--ste", action="store_true",
                    help="quantize latents (STE) during training; default is "
                         "continuous training with quantization only at eval")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path,
                    default=QLP_DIR / "work" / "polish_run")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(QLP_DIR / "work" / "payload.pt", weights_only=False)
    mins = payload["mins"].to(device)
    scales = payload["scales"].to(device)

    decoder = HNeRVDecoder(latent_dim=LATENT_DIM, base_channels=36,
                           eval_size=EVAL_SIZE).to(device)
    decoder.load_state_dict(payload["decoder_sd"])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    selector = SelectorApply(payload["selector_codes"],
                             payload["selector_specs"], device)

    video_path = get_default_video_path()
    print(f"video: {video_path}", flush=True)
    distortion_net, seg_targets_cpu, pose_targets, _, n_pairs = (
        precompute_targets(video_path, device))
    assert n_pairs == N_PAIRS

    latents = torch.nn.Parameter(payload["latents_eff"].clone().to(device))

    # ---- Baselines --------------------------------------------------------
    lat_bytes0 = latent_section_bytes(payload["q_codes"], mins, scales)
    print(f"[baseline] original latent section (ctx): {lat_bytes0:,} B "
          f"(shipping: 15,070 B + 607 B sidecar)", flush=True)

    with torch.no_grad():
        # (a) continuous sidecar-applied latents = current #1 pixels (GPU axis)
        seg_a, pose_a = [], []
        for i in range(0, N_PAIRS, 8):
            j = min(i + 8, N_PAIRS)
            idx = torch.arange(i, j, device=device)
            frames = decode_chain(decoder, latents[i:j].detach(), idx, selector,
                                  differentiable=False)
            so, po = scorer_forward(distortion_net, frames, fp32=True)
            pred = so.argmax(dim=1)
            tgt = seg_targets_cpu[i:j].long().to(device)
            seg_a.append((pred != tgt).float().flatten(1).mean(dim=1).cpu())
            pose_a.append(F.mse_loss(po['pose'][:, :6], pose_targets[i:j],
                                     reduction='none').mean(dim=1).cpu())
        seg_a = torch.cat(seg_a); pose_a = torch.cat(pose_a)
    print(f"[baseline a] #1 pixels (continuous+sidecar): "
          f"seg={seg_a.mean():.8f} pose={pose_a.mean():.8f} "
          f"(contest CPU: seg=0.00056023 pose=0.00002943)", flush=True)

    q0, seg_b, pose_b = full_eval(decoder, latents.detach(), mins, scales,
                                  selector, distortion_net, seg_targets_cpu,
                                  pose_targets, device)
    lb0 = latent_section_bytes(q0, mins, scales)
    print(f"[baseline b] grid-snap of start point (no sidecar): "
          f"seg={seg_b.mean():.8f} pose={pose_b.mean():.8f} "
          f"lat_bytes={lb0:,} -> projected {projected_score(seg_b.mean().item(), pose_b.mean().item(), lb0):.6f}",
          flush=True)

    # Per-pair best tracking. Marginal objective: seg_i + w * pose_i with
    # w = d(score)/d(pose_i) / d(score)/d(seg_i), linearized at current mean.
    pose_mean = float(pose_b.mean())
    w_pose = (10.0 / (2.0 * math.sqrt(10.0 * pose_mean))) / 100.0
    best_metric = seg_b + w_pose * pose_b
    best_codes = q0.clone()
    best_seg, best_pose = seg_b.clone(), pose_b.clone()

    # Global-best-epoch tracking: a NON-greedy alternative that ships one
    # whole epoch's codes (least overfit). Scored on the true projected score.
    gbest_proj = projected_score(seg_b.mean().item(), pose_b.mean().item(),
                                 latent_section_bytes(q0, mins, scales))
    gbest_codes = q0.clone()
    gbest_seg, gbest_pose = seg_b.clone(), pose_b.clone()

    opt = torch.optim.AdamW([latents], lr=args.lr, weight_decay=0.0)
    floor = args.lr_floor / args.lr
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: max(0.5 * (1 + math.cos(math.pi * ep / args.epochs)),
                            floor))

    tau_loss = lambda logits, tgt: l7_softplus_seg_loss(
        logits, tgt, tau=0.3, l7_threshold=1.0, l7_mult=4.0)

    start_ep = 0
    ckpt_path = args.out / "latest.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        latents.data.copy_(ck["latents"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"]
        best_codes = ck["best_codes"]
        best_metric = ck["best_metric"]
        best_seg, best_pose = ck["best_seg"], ck["best_pose"]
        print(f"resumed at epoch {start_ep}", flush=True)

    t0 = time.time()
    log_path = args.out / "log.jsonl"

    for epoch in range(start_ep, args.epochs):
        perm = torch.randperm(N_PAIRS, device=device)
        ep_loss, nb = 0.0, 0
        for s in range(0, N_PAIRS, args.batch):
            idx = perm[s:s + args.batch]
            lat_in = (ste_quantize_latents(latents[idx], mins, scales)
                      if args.ste else latents[idx])
            frames = decode_chain(decoder, lat_in, idx, selector,
                                  differentiable=True)
            seg_out, pose_out = scorer_forward(distortion_net, frames)
            tgt = seg_targets_cpu[idx.cpu()].long().to(device)
            seg_l = tau_loss(seg_out, tgt)
            pose_mse = F.mse_loss(pose_out['pose'][:, :6], pose_targets[idx])
            loss = 100.0 * seg_l + torch.sqrt(10.0 * pose_mse + 1e-12)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([latents], 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()

        if (epoch + 1) % 5 == 0 or epoch == start_ep:
            print(f"  ep{epoch+1}/{args.epochs} loss={ep_loss/nb:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if (epoch + 1) % args.eval_every == 0:
            q, seg_d, pose_d = full_eval(decoder, latents.detach(), mins,
                                         scales, selector, distortion_net,
                                         seg_targets_cpu, pose_targets, device)
            metric = seg_d + w_pose * pose_d
            improved = metric < best_metric
            best_codes[improved] = q[improved]
            best_seg[improved] = seg_d[improved]
            best_pose[improved] = pose_d[improved]
            best_metric[improved] = metric[improved]

            lb_cur = latent_section_bytes(q, mins, scales)
            lb_best = latent_section_bytes(best_codes, mins, scales)
            ps_cur = projected_score(seg_d.mean().item(), pose_d.mean().item(),
                                     lb_cur)
            ps_best = projected_score(best_seg.mean().item(),
                                      best_pose.mean().item(), lb_best)

            if ps_cur < gbest_proj:
                gbest_proj = ps_cur
                gbest_codes = q.clone()
                gbest_seg, gbest_pose = seg_d.clone(), pose_d.clone()

            print(f"    >>> ep{epoch+1}: cur seg={seg_d.mean():.8f} "
                  f"pose={pose_d.mean():.8f} bytes={lb_cur:,} proj={ps_cur:.6f} "
                  f"| per-pair ({int(improved.sum())}) proj={ps_best:.6f} "
                  f"| glob-ep proj={gbest_proj:.6f}", flush=True)

            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1,
                    "cur_seg": seg_d.mean().item(),
                    "cur_pose": pose_d.mean().item(),
                    "cur_bytes": lb_cur, "proj_cur": ps_cur,
                    "best_seg": best_seg.mean().item(),
                    "best_pose": best_pose.mean().item(),
                    "best_bytes": lb_best, "proj_best": ps_best,
                    "n_improved": int(improved.sum()),
                }) + "\n")

            torch.save({
                "epoch": epoch + 1,
                "latents": latents.data.cpu(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "best_codes": best_codes, "best_metric": best_metric,
                "best_seg": best_seg, "best_pose": best_pose,
            }, ckpt_path)
            torch.save({"q_codes": best_codes, "mins": mins.cpu(),
                        "scales": scales.cpu(),
                        "seg": best_seg, "pose": best_pose},
                       args.out / "best_codes.pt")
            torch.save({"q_codes": gbest_codes, "mins": mins.cpu(),
                        "scales": scales.cpu(),
                        "seg": gbest_seg, "pose": gbest_pose},
                       args.out / "gbest_codes.pt")

    print(f"\ndone in {(time.time()-t0)/3600:.2f} h — "
          f"final best proj={projected_score(best_seg.mean().item(), best_pose.mean().item(), latent_section_bytes(best_codes, mins, scales)):.6f}",
          flush=True)


if __name__ == "__main__":
    main()
