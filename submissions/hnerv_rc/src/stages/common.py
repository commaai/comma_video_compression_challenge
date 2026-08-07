"""Shared training loop used by every stage.

Per-stage variations are passed in via a `StageConfig` dataclass and a
`compute_seg_loss` callable.

Each stage writes:
  <output_dir>/decoder_f32.pt   — EMA weights at best-eval-score epoch
  <output_dir>/latents_f32.pt   — EMA latents at best-eval-score epoch
  <output_dir>/best_archive.bin — built from the best EMA
  <output_dir>/best_meta.json   — score / archive_bytes / epoch
  <output_dir>/final_decoder.pt — EMA weights at LAST epoch (for next stage)
  <output_dir>/final_latents.pt — EMA latents at LAST epoch (for next stage)

Inter-stage transitions read `final_*.pt`; the codec stage reads `*_f32.pt`
from the final training stage.
"""
from __future__ import annotations

import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent))

from model import HNeRVDecoder
from optim import Muon, partition_params_for_muon
from losses import (
    cat_entropy_v2,
    apply_qat, restore_qat, ema_update,
)
from data import precompute_targets, get_default_video_path, EVAL_SIZE
from codec import build_archive, parse_archive
from score import evaluate_decoder, compute_score, total_video_bytes


@dataclass
class StageConfig:
    name: str
    seg_loss_fn: Callable
    epochs: int
    eval_every: int = 25
    batch_size: int = 8
    ema_decay: float = 0.999

    use_muon: bool = False
    adamw_lr: float = 3e-5
    muon_lr: float = 2e-4
    muon_weight_decay: float = 0.0
    latent_lr_mult: float = 10.0
    grad_clip: float = 1.0
    grad_clip_muon: Optional[float] = 1.0
    lr_floor_ratio: float = 5e-6

    seg_weight: float = 100.0
    pose_weight: float = 1.0
    cat_lambda: float = 0.0
    cat_sigma: float = 0.2

    use_qat: bool = False

    resume_from: Optional[Path] = None
    output_dir: Optional[Path] = None
    init_latents_random: bool = False


def train_stage(cfg: StageConfig, device: torch.device,
                video_path: Optional[Path] = None,
                shared_state: Optional[dict] = None):
    """Train one stage. Returns dict with 'best_score', 'best_ep', 'archive_size'."""
    if video_path is None:
        video_path = get_default_video_path()

    print(f"\n{'='*80}\n[{cfg.name}] {cfg.epochs} ep, batch={cfg.batch_size}, "
          f"adamw_lr={cfg.adamw_lr}, muon_lr={cfg.muon_lr if cfg.use_muon else 'n/a'}, "
          f"lambda={cfg.cat_lambda}, sigma={cfg.cat_sigma}\n{'='*80}", flush=True)

    if cfg.output_dir is not None:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

    decoder = HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=EVAL_SIZE).to(device)

    if shared_state and 'distortion_net' in shared_state and shared_state.get('video_path') == video_path:
        distortion_net = shared_state['distortion_net']
        seg_targets_hard = shared_state['seg_targets_hard']
        pose_targets = shared_state['pose_targets']
        n_pairs = shared_state['n_pairs']
    else:
        distortion_net, seg_targets_hard, pose_targets, _, n_pairs = (
            precompute_targets(video_path, device))
        if shared_state is not None:
            shared_state.update({
                'distortion_net': distortion_net,
                'seg_targets_hard': seg_targets_hard,
                'pose_targets': pose_targets,
                'n_pairs': n_pairs,
                'video_path': video_path,
            })

    if cfg.resume_from is not None:
        sd_path = cfg.resume_from / "final_decoder.pt"
        lat_path = cfg.resume_from / "final_latents.pt"
        if not sd_path.exists() or not lat_path.exists():
            raise FileNotFoundError(
                f"Inter-stage resume needs final_decoder.pt + final_latents.pt at {cfg.resume_from}")
        decoder.load_state_dict(torch.load(sd_path, map_location=device))
        latents = nn.Parameter(torch.load(lat_path, map_location=device))
        print(f"  Loaded EMA from {cfg.resume_from.name}/final_*.pt", flush=True)
    elif cfg.init_latents_random:
        latents = nn.Parameter(torch.randn(n_pairs, 28, device=device) * 0.1)
        print("  Random init for decoder + latents.", flush=True)
    else:
        raise ValueError(f"Stage {cfg.name} has no resume_from and init_latents_random=False.")

    ema_decoder = deepcopy(decoder)
    ema_latents = latents.data.clone()

    if cfg.use_muon:
        muon_params, adamw_params = partition_params_for_muon(decoder)
        muon_opt = Muon(muon_params, lr=cfg.muon_lr, momentum=0.95, nesterov=True,
                        ns_steps=5, weight_decay=cfg.muon_weight_decay)
        adamw_opt = torch.optim.AdamW(
            [{'params': adamw_params, 'lr': cfg.adamw_lr},
             {'params': [latents], 'lr': cfg.adamw_lr * cfg.latent_lr_mult}],
            weight_decay=0.0,
        )
        print(f"  Muon: {sum(p.numel() for p in muon_params):,} params"
              f" ({len(muon_params)} tensors, wd={cfg.muon_weight_decay})", flush=True)
        print(f"  AdamW: {sum(p.numel() for p in adamw_params):,} decoder + {latents.numel():,} latent",
              flush=True)
    else:
        muon_opt = None
        muon_params = []
        adamw_params = list(decoder.parameters())
        adamw_opt = torch.optim.AdamW(
            [{'params': decoder.parameters(), 'lr': cfg.adamw_lr},
             {'params': [latents], 'lr': cfg.adamw_lr * cfg.latent_lr_mult}],
            weight_decay=0.0,
        )
        print(f"  AdamW only: {sum(p.numel() for p in decoder.parameters()):,} decoder + "
              f"{latents.numel():,} latent params", flush=True)

    eta_min_ratio = max(cfg.lr_floor_ratio / cfg.adamw_lr, 1e-3)
    def lr_lambda(epoch):
        return max(0.5 * (1 + math.cos(math.pi * epoch / cfg.epochs)), eta_min_ratio)
    adamw_sched = torch.optim.lr_scheduler.LambdaLR(adamw_opt, lr_lambda)
    muon_sched = (torch.optim.lr_scheduler.LambdaLR(muon_opt, lr_lambda)
                  if muon_opt is not None else None)

    tvb = total_video_bytes(video_path)

    best_score = float('inf'); best_ep = 0; best_archive_size = 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0; epoch_pose = 0.0; nb = 0
        pair_indices = torch.randperm(n_pairs, device=device)

        for batch_start in range(0, n_pairs, cfg.batch_size):
            idx = pair_indices[batch_start:batch_start + cfg.batch_size]
            B = len(idx)

            if cfg.use_qat:
                originals = apply_qat(decoder)
            decoded_pair = decoder(latents[idx])
            if cfg.use_qat:
                restore_qat(decoder, originals)

            flat = decoded_pair.reshape(B * 2, 3, EVAL_SIZE[0], EVAL_SIZE[1])
            up = F.interpolate(flat, size=(874, 1164), mode='bicubic', align_corners=False)
            down = F.interpolate(up, size=(384, 512), mode='bilinear', align_corners=False)
            decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)

            decoded_clamped = decoded_bhwc.clamp(0, 255)
            decoded_rounded = decoded_clamped.round()
            decoded_bhwc = decoded_clamped + (decoded_rounded - decoded_clamped).detach()

            posenet_in, segnet_in = distortion_net.preprocess_input(decoded_bhwc)
            seg_out = distortion_net.segnet(segnet_in)
            pose_out = distortion_net.posenet(posenet_in)

            seg_l = cfg.seg_loss_fn(seg_out, seg_targets_hard[idx])
            pose_mse = F.mse_loss(pose_out['pose'][:, :6], pose_targets[idx])
            pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)

            loss = cfg.seg_weight * seg_l + cfg.pose_weight * pose_l
            if cfg.cat_lambda > 0:
                ent = cat_entropy_v2(decoder, sigma=cfg.cat_sigma, sample_size=2000,
                                     device=device)
                loss = loss + cfg.cat_lambda * ent

            adamw_opt.zero_grad()
            if muon_opt is not None:
                muon_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adamw_params + [latents], cfg.grad_clip)
            if muon_opt is not None and cfg.grad_clip_muon is not None:
                torch.nn.utils.clip_grad_norm_(muon_params, cfg.grad_clip_muon)
            adamw_opt.step()
            if muon_opt is not None:
                muon_opt.step()

            ema_update(ema_decoder, decoder, ema_latents, latents, decay=cfg.ema_decay)

            epoch_loss += loss.item()
            epoch_pose += pose_mse.item()
            nb += 1

        adamw_sched.step()
        if muon_opt is not None:
            muon_sched.step()

        if (epoch + 1) % 10 == 0:
            print(f"  [{cfg.name}] ep{epoch+1}/{cfg.epochs} "
                  f"loss={epoch_loss/nb:.4f} pose_mse={epoch_pose/nb:.6f} "
                  f"lr={adamw_opt.param_groups[0]['lr']:.2e} ({time.time()-t0:.0f}s)", flush=True)

        if (epoch + 1) % cfg.eval_every == 0:
            archive = build_archive(
                ema_decoder.state_dict(), ema_latents.cpu(),
                meta_dict={"n_pairs": n_pairs, "latent_dim": 28, "base_channels": 36,
                           "eval_size": list(EVAL_SIZE)})
            archive_size = len(archive)
            eval_decoder_sd, eval_lat, _ = parse_archive(archive)
            eval_dec = HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=EVAL_SIZE).to(device)
            eval_dec.load_state_dict(eval_decoder_sd)
            eval_dec.eval()
            dist = evaluate_decoder(eval_dec, eval_lat.to(device), distortion_net,
                                    video_path, batch_pairs=8, device=device)
            result = compute_score(dist['seg_distortion'], dist['pose_distortion'],
                                   archive_size, tvb)
            del eval_dec
            torch.cuda.empty_cache()

            print(f"    >>> ep{epoch+1}: score={result['score']:.4f} "
                  f"seg={result['seg_distortion']:.5f} pose={result['pose_distortion']:.6f} "
                  f"arch={archive_size:,}", flush=True)

            if result['score'] < best_score:
                best_score = result['score']; best_ep = epoch + 1
                best_archive_size = archive_size
                if cfg.output_dir is not None:
                    with open(cfg.output_dir / "best_archive.bin", "wb") as f:
                        f.write(archive)
                    torch.save(ema_decoder.state_dict(), cfg.output_dir / "decoder_f32.pt")
                    torch.save(ema_latents.cpu(), cfg.output_dir / "latents_f32.pt")
                    with open(cfg.output_dir / "best_meta.json", "w") as f:
                        json.dump({"stage": cfg.name, "score": result['score'],
                                   "seg_distortion": result['seg_distortion'],
                                   "pose_distortion": result['pose_distortion'],
                                   "archive_bytes": archive_size,
                                   "epoch": epoch + 1}, f, indent=2)

    if cfg.output_dir is not None:
        torch.save(ema_decoder.state_dict(), cfg.output_dir / "final_decoder.pt")
        torch.save(ema_latents.cpu(), cfg.output_dir / "final_latents.pt")

    print(f"\n[{cfg.name}] BEST: {best_score:.4f} at ep{best_ep}", flush=True)
    return {
        'stage': cfg.name,
        'best_score': best_score,
        'best_ep': best_ep,
        'archive_size': best_archive_size,
        'output_ckpt_dir': cfg.output_dir,
    }
