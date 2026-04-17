#!/usr/bin/env python
"""
Inflation: decodes seed video, loads stored targets, optimizes pixel values
at model input resolution (384x512) via gradient descent through SegNet/PoseNet,
then upsamples to full camera resolution for output.
"""
import sys, os, struct, bz2, time
import numpy as np
import torch
import torch.nn.functional as F
import einops

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from frame_utils import camera_size, segnet_model_input_size
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

import av as av_lib


def rgb_to_yuv6_diff(rgb_chw):
    """Differentiable rgb_to_yuv6 (no @torch.no_grad)."""
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, :2*H2, :2*W2]
    R, G, B = rgb[..., 0, :, :], rgb[..., 1, :, :], rgb[..., 2, :, :]
    Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
    U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] + U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
    V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] + V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
    return torch.stack([Y[..., 0::2, 0::2], Y[..., 1::2, 0::2], Y[..., 0::2, 1::2], Y[..., 1::2, 1::2], U_sub, V_sub], dim=-3)


def decode_seed_video(video_path):
    container = av_lib.open(video_path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray(format='rgb24')
        frames.append(torch.from_numpy(arr))
    container.close()
    return frames


def load_class_maps(path):
    with open(path, 'rb') as f:
        N, H, W = struct.unpack('<III', f.read(12))
        compressed = f.read()
    data = np.frombuffer(bz2.decompress(compressed), dtype=np.uint8).reshape(N, H, W)
    return torch.from_numpy(data.copy())


def load_pose_targets(path):
    with open(path, 'rb') as f:
        N, D = struct.unpack('<II', f.read(8))
        compressed = f.read()
    data = np.frombuffer(bz2.decompress(compressed), dtype=np.float32).reshape(N, D)
    return torch.from_numpy(data.copy())


def optimize_batch(init_384, class_targets, pose_targets, dn, device,
                   n_steps=80, lr=1.0, seg_weight=10.0, pose_weight=3.0):
    """
    Two-phase optimization at 384x512 resolution.
    Phase 1: Direct model-res optimization (fast convergence)
    Phase 2: Round-trip through full resolution (matches eval chain)

    init_384: (B, 2, 3, mH, mW) float tensor - initialization at model res
    class_targets: (B, mH, mW) long - target SegNet classes
    pose_targets: (B, 6) float - target PoseNet pose values
    """
    B = init_384.shape[0]
    W, H = camera_size
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    use_amp = device.type == 'cuda'

    # Phase 1: Fast optimization at model resolution (no round-trip)
    pixels = init_384.clone().detach().requires_grad_(True)
    p1_steps = n_steps // 2
    optimizer = torch.optim.Adam([pixels], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p1_steps, eta_min=lr * 0.1)

    for step in range(p1_steps):
        optimizer.zero_grad()
        x = pixels.clamp(0, 255)
        with torch.amp.autocast('cuda', enabled=use_amp):
            seg_out = dn.segnet(x[:, -1])
            seg_loss = F.cross_entropy(seg_out, class_targets)
            bt = einops.rearrange(x, 'b t c h w -> (b t) c h w')
            yuv = einops.rearrange(rgb_to_yuv6_diff(bt), '(b t) c h w -> b (t c) h w', b=B, t=2, c=6)
            pose_loss = ((dn.posenet(yuv)['pose'][:, :6] - pose_targets) ** 2).mean()
            loss = seg_weight * seg_loss + pose_weight * pose_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            pixels.data.clamp_(0, 255)

    # Phase 2: Refine with round-trip to match actual evaluation
    p2_steps = n_steps - p1_steps
    optimizer2 = torch.optim.Adam([pixels], lr=lr * 0.3)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=p2_steps, eta_min=lr * 0.02)

    for step in range(p2_steps):
        optimizer2.zero_grad()
        x = pixels.clamp(0, 255)
        with torch.amp.autocast('cuda', enabled=use_amp):
            bt = einops.rearrange(x, 'b t c h w -> (b t) c h w')
            full = F.interpolate(bt, size=(H, W), mode='bicubic', align_corners=False).clamp(0, 255)
            eval_384 = F.interpolate(full, size=(mH, mW), mode='bilinear')
            eval_pairs = einops.rearrange(eval_384, '(b t) c h w -> b t c h w', b=B, t=2)
            seg_loss = F.cross_entropy(dn.segnet(eval_pairs[:, -1]), class_targets)
            yuv = einops.rearrange(rgb_to_yuv6_diff(eval_384), '(b t) c h w -> b (t c) h w', b=B, t=2, c=6)
            pose_loss = ((dn.posenet(yuv)['pose'][:, :6] - pose_targets) ** 2).mean()
            loss = seg_weight * seg_loss + pose_weight * pose_loss
        loss.backward()
        optimizer2.step()
        scheduler2.step()
        with torch.no_grad():
            pixels.data.clamp_(0, 255)

    # Generate full-res output
    with torch.no_grad():
        x = pixels.detach().clamp(0, 255)
        bt = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        full = F.interpolate(bt, size=(H, W), mode='bicubic', align_corners=False)
        full = full.clamp(0, 255).round().to(torch.uint8)
        full = einops.rearrange(full, '(b t) c h w -> b t h w c', b=B, t=2)

    return full


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('archive_dir')
    parser.add_argument('output_dir')
    parser.add_argument('video_names_file')
    parser.add_argument('--device', default=None)
    parser.add_argument('--n-steps', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seg-weight', type=float, default=10.0)
    parser.add_argument('--pose-weight', type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    t0 = time.time()
    W, H = camera_size
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    print(f"Device: {device}, Model res: {mH}x{mW}, Camera: {H}x{W}")

    # Load models
    print("Loading models...")
    dn = DistortionNet().eval().to(device)
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in dn.parameters():
        p.requires_grad_(False)

    with open(args.video_names_file) as f:
        video_names = [l.strip() for l in f if l.strip()]

    for vname in video_names:
        base = os.path.splitext(vname)[0]
        print(f"\nProcessing {vname}...")

        # Decode seed video
        seed_path = os.path.join(args.archive_dir, f'{base}.mkv')
        seed_frames = decode_seed_video(seed_path)
        n_frames = len(seed_frames)
        print(f"  {n_frames} seed frames decoded")

        # Resize seed frames to model input resolution
        init_frames = []
        for sf in seed_frames:
            x = sf.float().permute(2, 0, 1).unsqueeze(0).to(device)
            x = F.interpolate(x, size=(mH, mW), mode='bicubic', align_corners=False).clamp(0, 255)
            init_frames.append(x.squeeze(0))
        del seed_frames

        # Load targets
        class_maps = load_class_maps(os.path.join(args.archive_dir, f'{base}.segmap')).to(device)
        pose_targets = load_pose_targets(os.path.join(args.archive_dir, f'{base}.pose')).to(device)
        print(f"  {len(class_maps)} class maps, {len(pose_targets)} pose targets")

        # Process in batches
        n_pairs = n_frames // 2
        all_frames = []

        for start in range(0, n_pairs, args.batch_size):
            end = min(start + args.batch_size, n_pairs)
            B = end - start

            # Build initialization batch: (B, 2, 3, mH, mW)
            batch_init = torch.stack([
                torch.stack([init_frames[i * 2], init_frames[i * 2 + 1]])
                for i in range(start, end)
            ])

            seg_tgt = class_maps[start:end].long()
            pose_tgt = pose_targets[start:end].float()

            # Optimize
            optimized = optimize_batch(
                batch_init, seg_tgt, pose_tgt, dn, device,
                n_steps=args.n_steps, lr=args.lr,
                seg_weight=args.seg_weight, pose_weight=args.pose_weight
            )

            # Collect frames: (B, 2, H, W, 3) uint8
            for i in range(B):
                all_frames.append(optimized[i, 0].cpu())
                all_frames.append(optimized[i, 1].cpu())

            elapsed = time.time() - t0
            batch_num = start // args.batch_size + 1
            total_batches = (n_pairs + args.batch_size - 1) // args.batch_size
            eta = elapsed / batch_num * total_batches - elapsed
            print(f"  Batch {batch_num}/{total_batches} ({elapsed:.0f}s, ETA {eta:.0f}s)")

        # Save raw output
        raw_path = os.path.join(args.output_dir, f'{base}.raw')
        raw_data = torch.stack(all_frames).numpy()
        raw_data.tofile(raw_path)
        print(f"  Saved {raw_path}: {os.path.getsize(raw_path):,} bytes, {len(all_frames)} frames")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
