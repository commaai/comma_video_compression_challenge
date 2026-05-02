"""
Train the ScaleHyperpriorCompressor on driving videos.

Usage:
    python train.py \
        --video_dir /path/to/test_videos \
        --segnet_ckpt /path/to/models/segnet.safetensors \
        --posenet_ckpt /path/to/models/posenet.safetensors \
        --out_dir ./checkpoints \
        --crop 256 256 --batch_size 8 --epochs 10

A note on training resolution:
  Both comma SegNet and PoseNet downsample inputs to 384x512 internally, so the
  loss is measured at that scale anyway. Training on 256x256 crops is a great
  speed/quality tradeoff. Full 1164x874 frames are only needed at eval time.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports (this file lives next to compressor.py and loss.py)
from compressor import ScaleHyperpriorCompressor, num_params
from dataset import FramePairDataset, list_videos
from loss import CommaScoreLoss


# --- Loading the official comma models ---------------------------------------
# Their modules.py defines `SegNet`, `PoseNet`, and `DistortionNet`.
# We assume this script is run from inside the comma_video_compression_challenge
# repo (so `modules.py` is importable). Adjust sys.path if not.
import sys

def load_comma_networks(repo_root: str, segnet_ckpt: str, posenet_ckpt: str, device: torch.device):
    """Import comma's SegNet/PoseNet from the challenge repo and load weights."""
    sys.path.insert(0, repo_root)
    from modules import SegNet, PoseNet  # noqa: E402
    from safetensors.torch import load_file  # noqa: E402

    segnet = SegNet().to(device)
    segnet.load_state_dict(load_file(segnet_ckpt, device=str(device)))
    segnet.eval()

    posenet = PoseNet().to(device)
    posenet.load_state_dict(load_file(posenet_ckpt, device=str(device)))
    posenet.eval()
    return segnet, posenet


def configure_optimizers(model: nn.Module, lr: float = 1e-4, aux_lr: float = 1e-3):
    """compressai EntropyBottleneck has 'quantiles' aux params — needs its own optimizer."""
    main_params = {n: p for n, p in model.named_parameters()
                   if p.requires_grad and not n.endswith(".quantiles")}
    aux_params = {n: p for n, p in model.named_parameters()
                  if p.requires_grad and n.endswith(".quantiles")}

    optimizer = optim.Adam(main_params.values(), lr=lr)
    aux_optimizer = optim.Adam(aux_params.values(), lr=aux_lr)
    return optimizer, aux_optimizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", type=str, default=".",
                   help="Path to the comma_video_compression_challenge repo (for modules.py).")
    p.add_argument("--video_dir", type=str, required=True,
                   help="Directory of training videos (the 2.4 GB test_videos.zip extracted).")
    p.add_argument("--segnet_ckpt", type=str, default="models/segnet.safetensors")
    p.add_argument("--posenet_ckpt", type=str, default="models/posenet.safetensors")
    p.add_argument("--out_dir", type=str, default="./checkpoints")

    p.add_argument("--N", type=int, default=64, help="Conv channel width")
    p.add_argument("--M", type=int, default=128, help="Latent channels")
    p.add_argument("--N_hyp", type=int, default=64, help="Hyperprior channels")

    p.add_argument("--crop", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps_per_epoch", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--aux_lr", type=float, default=1e-3)
    p.add_argument("--lam_rate", type=float, default=0.01,
                   help="Lagrangian weight on bpp. Sweep this {0.001, 0.01, 0.1}.")
    p.add_argument("--use_official_weights", action="store_true",
                   help="Use the exact 100/25/sqrt(10*) recipe instead of Lagrangian.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # ---- data ----
    paths = list_videos(args.video_dir)
    if not paths:
        raise RuntimeError(f"No videos found in {args.video_dir}")
    print(f"Found {len(paths)} training videos")

    ds = FramePairDataset(paths, crop_size=tuple(args.crop), shuffle_videos=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ---- models ----
    print("Loading comma SegNet/PoseNet (frozen)...")
    segnet, posenet = load_comma_networks(args.repo_root, args.segnet_ckpt, args.posenet_ckpt, device)

    print("Building compressor...")
    compressor = ScaleHyperpriorCompressor(N=args.N, M=args.M, N_hyp=args.N_hyp).to(device)
    print(f"  params: {num_params(compressor):,}  "
          f"(~{num_params(compressor)*4/1e6:.1f} MB fp32, ~{num_params(compressor)*2/1e6:.1f} MB fp16)")

    loss_fn = CommaScoreLoss(
        segnet, posenet,
        lam_rate=args.lam_rate,
        use_official_weights=args.use_official_weights,
    ).to(device)

    optimizer, aux_optimizer = configure_optimizers(compressor, lr=args.lr, aux_lr=args.aux_lr)

    # ---- train ----
    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        loader_iter = iter(loader)
        for _ in range(args.steps_per_epoch):
            try:
                pair = next(loader_iter)            # (B, 2, 3, H, W) uint8
            except StopIteration:
                loader_iter = iter(loader)
                pair = next(loader_iter)

            pair = pair.to(device, non_blocking=True)
            B, T, C, H, W = pair.shape
            assert T == 2

            # Compress each frame independently (you can extend to conditional coding later)
            x_in = pair.view(B * T, C, H, W).float()       # (2B, 3, H, W)  in [0,255]
            out = compressor(x_in)                          # forward (with noise)
            x_hat = out["x_hat"].view(B, T, C, H, W)        # (B, 2, 3, H, W)
            x_orig = pair.float()                           # (B, 2, 3, H, W)

            losses = loss_fn(
                x_orig=x_orig,
                x_recon=x_hat,
                y_likelihoods=out["y_likelihoods"],
                z_likelihoods=out["z_likelihoods"],
                num_pixels=out["num_pixels"],
            )

            optimizer.zero_grad(set_to_none=True)
            aux_optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()

            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(compressor.parameters(), args.grad_clip)
            optimizer.step()

            # auxiliary loss for entropy bottleneck quantiles (compressai-specific)
            aux_loss = compressor.entropy_bottleneck.loss()
            aux_loss.backward()
            aux_optimizer.step()

            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                # Approximate "score" using the official weights for monitoring
                approx_score = (
                    100.0 * losses["seg"].item()
                    + 25.0 * (losses["bpp"].item() / 24.0)
                    + (10.0 * losses["pose"].item() + 1e-8) ** 0.5
                )
                print(
                    f"[ep {epoch} step {global_step}] "
                    f"loss={losses['loss'].item():.4f} "
                    f"seg={losses['seg'].item():.4f} "
                    f"pose={losses['pose'].item():.4f} "
                    f"bpp={losses['bpp'].item():.3f} "
                    f"aux={aux_loss.item():.3f} "
                    f"~score={approx_score:.3f} "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )

            if global_step > 0 and global_step % args.save_every == 0:
                ckpt_path = out_dir / f"compressor_step{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "model": compressor.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                print(f"  saved {ckpt_path}")

            global_step += 1

        # save at end of every epoch
        ckpt_path = out_dir / f"compressor_ep{epoch}.pt"
        torch.save({
            "step": global_step,
            "model": compressor.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        print(f"end of epoch {epoch}: saved {ckpt_path}")

    # ---- final: build CDF tables for arithmetic coding ----
    print("Building entropy CDFs (compressor.update)...")
    compressor.update(force=True)
    final_path = out_dir / "compressor_final.pt"
    torch.save({"model": compressor.state_dict(), "args": vars(args)}, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()