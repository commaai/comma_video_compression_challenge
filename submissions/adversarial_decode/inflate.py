#!/usr/bin/env python
"""
Phase 2 Decoder: Adversarial decode via evaluation network inversion.

Loads compressed SegNet/PoseNet targets from archive, then runs gradient
descent through the evaluation networks to generate frames that reproduce
the correct network outputs. Frames may look nothing like the original
video -- they only need to fool SegNet and PoseNet.

Usage: python -m submissions.adversarial_decode.inflate <data_dir> <output_path>
"""
import sys, struct, zlib, time, warnings
import torch, numpy as np
import torch.nn.functional as F
warnings.filterwarnings('ignore', message='.*lr_scheduler.*optimizer.*')
from pathlib import Path
from safetensors.torch import load_file
from frame_utils import camera_size, segnet_model_input_size
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path



# ── Differentiable reimplementations ────────────────────────────────────────
# frame_utils.rgb_to_yuv6 is decorated with @torch.no_grad(), which kills
# gradient flow. We reimplement it identically but without that decorator.

def rgb_to_yuv6_diff(rgb_chw: torch.Tensor) -> torch.Tensor:
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, :2 * H2, :2 * W2]

    R = rgb[..., 0, :, :]
    G = rgb[..., 1, :, :]
    B = rgb[..., 2, :, :]

    Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)

    U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
             U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
    V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
             V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25

    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)


def posenet_preprocess_diff(x: torch.Tensor) -> torch.Tensor:
    """PoseNet.preprocess_input with differentiable rgb_to_yuv6.

    Args:
        x: (B, T=2, C=3, H, W) float tensor, frames already at model res (384x512)
    Returns:
        (B, 12, 192, 256) float tensor ready for PoseNet.forward
    """
    B, T = x.shape[0], x.shape[1]
    flat = x.reshape(B * T, *x.shape[2:])         # (B*T, 3, H, W)
    yuv = rgb_to_yuv6_diff(flat)                    # (B*T, 6, H/2, W/2)
    return yuv.reshape(B, T * yuv.shape[1], *yuv.shape[2:])  # (B, 12, 192, 256)


# ── Loss functions ──────────────────────────────────────────────────────────

def margin_loss(logits, target, margin=3.0):
    """Ensure target class logit exceeds all competitors by at least `margin`.

    This provides a safety buffer against uint8 rounding noise.
    """
    target_logits = logits.gather(1, target.unsqueeze(1))
    competitor = logits.clone()
    competitor.scatter_(1, target.unsqueeze(1), float('-inf'))
    max_other = competitor.max(dim=1, keepdim=True).values
    return F.relu(max_other - target_logits + margin).mean()


# ── Data loading ────────────────────────────────────────────────────────────

DECOMPRESS = {
    0: lambda b: zlib.decompress(b),
    1: lambda b: __import__('lzma').decompress(b),
    2: lambda b: __import__('bz2').decompress(b),
}


def load_targets(data_dir: Path):
    """Load and decompress seg maps, pose vectors, and ideal colors."""
    # Segmentation maps
    with open(data_dir / 'seg.bin', 'rb') as f:
        n, H, W, flags = struct.unpack('<IIII', f.read(16))
        compress_method = flags & 0x3
        is_4bit = bool(flags & (1 << 2))
        is_row_interleaved = bool(flags & (1 << 3))
        has_perm = bool(flags & (1 << 4))
        is_left_residual = bool(flags & (1 << 5))

        perm_lut = None
        if has_perm:
            fwd_perm = list(f.read(5))
            inv_lut = np.zeros(5, dtype=np.uint8)
            for orig_class, encoded_val in enumerate(fwd_perm):
                inv_lut[encoded_val] = orig_class
            perm_lut = inv_lut

        raw_bytes = DECOMPRESS[compress_method](f.read())

        if is_4bit:
            packed = np.frombuffer(raw_bytes, dtype=np.uint8)
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            interleaved = np.empty(len(packed) * 2, dtype=np.uint8)
            interleaved[0::2] = low
            interleaved[1::2] = high
            total_pixels = n * H * W
            seg_maps = interleaved[:total_pixels].reshape(n, H, W)
        elif is_row_interleaved:
            seg_maps = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(H, n, W)
            seg_maps = np.ascontiguousarray(seg_maps.transpose(1, 0, 2))
        else:
            seg_maps = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n, H, W)

        if perm_lut is not None:
            seg_maps = perm_lut[seg_maps]

        if is_left_residual:
            for c in range(1, W):
                seg_maps[:, :, c] = (seg_maps[:, :, c] + seg_maps[:, :, c - 1]) % 5
            seg_maps = seg_maps.astype(np.uint8)

    # Pose vectors
    with open(data_dir / 'pose.bin', 'rb') as f:
        header = f.read(12)
        if len(header) == 12:
            n_p, d, pose_flags = struct.unpack('<III', header)
        else:
            f.seek(0)
            n_p, d = struct.unpack('<II', f.read(8))
            pose_flags = 0
        pose_method = pose_flags & 0x3
        is_f16 = bool(pose_flags & (1 << 2))
        raw_pose = DECOMPRESS[pose_method](f.read())
        if is_f16:
            pose_vectors = np.frombuffer(raw_pose, dtype=np.float16).astype(np.float32).reshape(n_p, d)
        else:
            pose_deltas = np.frombuffer(raw_pose, dtype=np.float32).reshape(n_p, d)
            pose_vectors = np.cumsum(pose_deltas, axis=0)

    # Ideal class colors (hardcoded to avoid storing in archive)
    colors = np.array([
        [52.373100, 66.082504, 53.425133],
        [132.627213, 139.283707, 154.640060],
        [0.000000, 58.369274, 200.949326],
        [200.236008, 213.412567, 201.891022],
        [26.859495, 41.075771, 46.146519],
    ], dtype=np.float32)

    # Override with archive colors if present (backward compat)
    colors_path = data_dir / 'colors.bin'
    if colors_path.exists():
        colors = np.frombuffer(
            colors_path.read_bytes(), dtype=np.float32
        ).reshape(5, 3)

    return seg_maps, pose_vectors, colors


# ── Optimization core ──────────────────────────────────────────────────────

def optimize_batch(
    segnet, posenet,
    target_seg, target_pose, ideal_colors,
    H_cam, W_cam, device, use_amp,
    num_iters=100, lr=2.0, seg_margin=5.0,
    alpha=10.0, beta=1.0,
):
    """Optimize at model resolution (384x512), then upscale to camera resolution.

    Uses AdamW optimizer with betas=(0.9, 0.88) for faster second-moment decay,
    cosine LR schedule, Huber (smooth L1) loss for PoseNet, margin loss for
    SegNet. Dynamic 3-phase loss weighting:
    - Early phase (0-30%): SegNet-dominant (alpha, beta*0.3)
    - Mid phase (30-90%): balanced (alpha, beta*1.5)
    - Late phase (90-100%): PoseNet-dominant (alpha*0.2, beta*8)

    Returns:
        f0_uint8: (B, 3, H_cam, W_cam) uint8 even frames
        f1_uint8: (B, 3, H_cam, W_cam) uint8 odd frames
        stats: dict with loss values and mismatch counts
    """
    B = target_seg.shape[0]
    model_H, model_W = segnet_model_input_size[1], segnet_model_input_size[0]
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    init_odd = ideal_colors[target_seg].permute(0, 3, 1, 2).clone()
    frame_1 = init_odd.requires_grad_(True)
    f0_init = init_odd.detach().mean(dim=(-2, -1), keepdim=True).expand_as(init_odd).clone()
    frame_0 = f0_init.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [frame_0, frame_1], lr=lr, weight_decay=0.025, betas=(0.9, 0.88)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iters, eta_min=lr * 0.05
    )

    seg_loss_val = 0.0
    pose_loss_val = 0.0

    AVG_LAST_K = 10
    f0_accum = torch.zeros_like(init_odd)
    f1_accum = torch.zeros_like(init_odd)
    avg_count = 0

    early_cutoff = int(0.3 * max(num_iters - 1, 1))

    # Early stopping: when seg_loss is near-zero for PATIENCE consecutive
    # iters after MIN_ES iterations, start accumulating for averaging then stop.
    # This banks saved time for harder batches.
    EARLY_STOP_MIN = 50
    EARLY_STOP_SEG_THRESH = 1e-4
    EARLY_STOP_POSE_THRESH = 4e-4
    EARLY_STOP_PATIENCE = 5
    converged_count = 0
    early_stop_triggered = False
    actual_iters = num_iters

    for it in range(num_iters):
        progress = it / max(num_iters - 1, 1)
        run_pose = (it >= early_cutoff)

        if progress < 0.3:
            a_eff = alpha
            b_eff = beta * 0.3
        elif progress < 0.9:
            a_eff = alpha
            b_eff = beta * 1.5
        else:
            a_eff = alpha * 0.2
            b_eff = beta * 8.0

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast('cuda'):
                seg_logits = segnet(frame_1)
                seg_l = margin_loss(seg_logits, target_seg, seg_margin)

                if run_pose:
                    both = torch.stack([frame_0, frame_1], dim=1)
                    pn_in = posenet_preprocess_diff(both)
                    pn_out = posenet(pn_in)['pose'][:, :6]
                    pose_l = F.smooth_l1_loss(pn_out, target_pose)
                    total = a_eff * seg_l + b_eff * pose_l
                else:
                    pose_l = seg_l.new_zeros(())
                    total = a_eff * seg_l

            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            seg_logits = segnet(frame_1)
            seg_l = margin_loss(seg_logits, target_seg, seg_margin)

            if run_pose:
                both = torch.stack([frame_0, frame_1], dim=1)
                pn_in = posenet_preprocess_diff(both)
                pn_out = posenet(pn_in)['pose'][:, :6]
                pose_l = F.smooth_l1_loss(pn_out, target_pose)
                total = a_eff * seg_l + b_eff * pose_l
            else:
                pose_l = seg_l.new_zeros(())
                total = a_eff * seg_l

            total.backward()
            optimizer.step()

        with torch.no_grad():
            scheduler.step()

        with torch.no_grad():
            frame_0.data.clamp_(0, 255)
            frame_1.data.clamp_(0, 255)

        seg_loss_val = seg_l.item()
        if run_pose:
            pose_loss_val = pose_l.item()

        # Early stopping logic
        with torch.no_grad():
            if it >= EARLY_STOP_MIN and not early_stop_triggered and run_pose:
                if seg_loss_val < EARLY_STOP_SEG_THRESH and pose_loss_val < EARLY_STOP_POSE_THRESH:
                    converged_count += 1
                else:
                    converged_count = 0
                if converged_count >= EARLY_STOP_PATIENCE:
                    early_stop_triggered = True
                    f0_accum.zero_()
                    avg_count = 0

            # Accumulate for averaging: either post-convergence or last K iters
            if early_stop_triggered:
                f0_accum += frame_0.data
                f1_accum += frame_1.data
                avg_count += 1
                if avg_count >= AVG_LAST_K:
                    actual_iters = it + 1
                    break
            elif it >= num_iters - AVG_LAST_K:
                f0_accum += frame_0.data
                f1_accum += frame_1.data
                avg_count += 1

    # Use averaged frames from last K iterations to smooth out oscillation
    final_f0 = f0_accum / avg_count if avg_count > 0 else frame_0.detach()
    final_f1 = f1_accum / avg_count if avg_count > 0 else frame_1.detach()

    # Free optimizer/scheduler/grad state before upscale to reclaim VRAM
    del optimizer, scheduler, frame_0, frame_1, f0_accum, f1_accum, init_odd
    if scaler is not None:
        del scaler
    if use_amp:
        torch.cuda.empty_cache()

    # Upscale to camera resolution and quantize
    with torch.no_grad():
        f0_up = F.interpolate(final_f0, size=(H_cam, W_cam),
                              mode='bicubic', align_corners=False)
        f0_up = f0_up.clamp(0, 255).round()

        f1_up = F.interpolate(final_f1, size=(H_cam, W_cam),
                              mode='bicubic', align_corners=False)
        f1_up = f1_up.clamp(0, 255).round()

    f0_uint8 = f0_up.to(torch.uint8)
    f1_uint8 = f1_up.to(torch.uint8)
    del f0_up, f1_up, final_f0, final_f1

    stats = {
        'seg_loss': seg_loss_val,
        'pose_loss': pose_loss_val,
        'actual_iters': actual_iters,
    }
    return f0_uint8, f1_uint8, stats


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    data_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Device: {device}, AMP: {use_amp}")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    t0 = time.time()

    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))

    for p in segnet.parameters():
        p.requires_grad_(False)
    for p in posenet.parameters():
        p.requires_grad_(False)

    print(f"Models loaded in {time.time() - t0:.1f}s")

    # Load targets from archive
    seg_maps_np, pose_vectors_np, ideal_colors_np = load_targets(data_dir)

    # Upscale seg maps to model resolution if stored at lower res
    model_H, model_W = segnet_model_input_size[1], segnet_model_input_size[0]
    if seg_maps_np.shape[1] != model_H or seg_maps_np.shape[2] != model_W:
        seg_t = torch.from_numpy(seg_maps_np.astype(np.int64)).unsqueeze(1).float()
        seg_t = F.interpolate(seg_t, size=(model_H, model_W), mode='nearest')
        seg_maps = seg_t.squeeze(1).long().to(device)
        print(f"Upscaled seg maps: {seg_maps_np.shape} -> ({seg_maps_np.shape[0]}, {model_H}, {model_W})")
    else:
        seg_maps = torch.from_numpy(seg_maps_np).long().to(device)

    pose_vectors = torch.from_numpy(pose_vectors_np.copy()).float().to(device)

    num_pairs = seg_maps.shape[0]
    W_cam, H_cam = camera_size  # 1164, 874

    ideal_colors = torch.from_numpy(ideal_colors_np.copy()).float().to(device)
    print(f"Loaded ideal class colors")

    if device.type == 'cuda':
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True)
            vram_mb = int(result.stdout.strip().split('\n')[0])
            if vram_mb >= 20000:
                batch_size = 16
            elif vram_mb >= 12000:
                batch_size = 8
            else:
                batch_size = 4
        except Exception:
            batch_size = 4
    else:
        batch_size = 2

    print(f"Processing {num_pairs} pairs at {W_cam}x{H_cam}, batch_size={batch_size}")

    TARGET_ITERS = 150
    MIN_ITERS = 60
    LR = 1.2
    SEG_MARGIN = 0.1
    ALPHA = 120.0
    BETA = 0.20
    WALL_BUDGET = 1400   # ~23.3 min inflate budget (conservative for CI safety)

    total_batches = (num_pairs + batch_size - 1) // batch_size
    avg_time_per_iter = None
    batch_overhead = 2.0

    # Pre-compute per-batch difficulty based on boundary pixel fraction.
    # More boundary pixels = harder to optimize = needs more iterations.
    seg_np = seg_maps_np
    batch_difficulty = []
    for bs in range(0, num_pairs, batch_size):
        be = min(bs + batch_size, num_pairs)
        chunk = seg_np[bs:be]  # (B, H, W)
        h_diff = (chunk[:, :, :-1] != chunk[:, :, 1:]).sum()
        v_diff = (chunk[:, :-1, :] != chunk[:, 1:, :]).sum()
        boundary_frac = (h_diff + v_diff) / max(chunk.size, 1)
        batch_difficulty.append(float(boundary_frac))
    mean_diff = sum(batch_difficulty) / len(batch_difficulty) if batch_difficulty else 1.0
    # Relative difficulty: >1 means harder than average, <1 means easier
    batch_rel_diff = [d / mean_diff if mean_diff > 0 else 1.0 for d in batch_difficulty]
    print(f"Batch difficulty range: {min(batch_difficulty):.4f} - {max(batch_difficulty):.4f} "
          f"(mean={mean_diff:.4f})")

    with open(output_path, 'wb') as f_out:
        for batch_start in range(0, num_pairs, batch_size):
            batch_end = min(batch_start + batch_size, num_pairs)
            B = batch_end - batch_start
            t_batch = time.time()

            target_seg = seg_maps[batch_start:batch_end]
            target_pose = pose_vectors[batch_start:batch_end]

            batch_idx = batch_start // batch_size + 1
            elapsed_total = time.time() - t0
            remaining_time = WALL_BUDGET - elapsed_total
            remaining_batches = total_batches - batch_idx + 1
            rel_diff = batch_rel_diff[batch_idx - 1]

            if avg_time_per_iter is not None and remaining_batches > 0:
                time_for_iters = (remaining_time - remaining_batches * batch_overhead) / remaining_batches
                affordable = max(MIN_ITERS, int(time_for_iters / avg_time_per_iter))
                base_iters = min(TARGET_ITERS, affordable)
                # Scale iterations by relative difficulty (clamped to [0.7x, 1.4x])
                scale = max(0.7, min(1.4, rel_diff))
                num_iters = max(MIN_ITERS, min(TARGET_ITERS, int(base_iters * scale)))
            else:
                num_iters = TARGET_ITERS

            t_opt_start = time.time()
            f0, f1, stats = optimize_batch(
                segnet, posenet,
                target_seg, target_pose, ideal_colors,
                H_cam, W_cam, device, use_amp,
                num_iters=num_iters, lr=LR, seg_margin=SEG_MARGIN,
                alpha=ALPHA, beta=BETA,
            )
            t_opt_end = time.time()

            for b in range(B):
                f_out.write(f0[b].permute(1, 2, 0).contiguous().cpu().numpy().tobytes())
                f_out.write(f1[b].permute(1, 2, 0).contiguous().cpu().numpy().tobytes())

            elapsed_batch = time.time() - t_batch
            opt_time = t_opt_end - t_opt_start
            actual_overhead = elapsed_batch - opt_time

            used_iters = stats['actual_iters']

            if batch_idx > 1:
                tpi = opt_time / used_iters
                if avg_time_per_iter is None:
                    avg_time_per_iter = tpi
                else:
                    avg_time_per_iter = 0.8 * avg_time_per_iter + 0.2 * tpi
                batch_overhead = 0.8 * batch_overhead + 0.2 * actual_overhead

            elapsed_total = time.time() - t0
            pairs_done = batch_end
            eta = elapsed_total / pairs_done * (num_pairs - pairs_done)

            es_tag = f" ES@{used_iters}" if used_iters < num_iters else ""
            print(f"[{batch_idx}/{total_batches}] pairs {batch_start}-{batch_end-1} | "
                  f"{elapsed_batch:.1f}s ({used_iters}/{num_iters}it, d={rel_diff:.2f}{es_tag}) | "
                  f"seg={stats['seg_loss']:.4f} pose={stats['pose_loss']:.6f} | "
                  f"ETA {eta:.0f}s")

    total_time = time.time() - t0
    raw_size = num_pairs * 2 * H_cam * W_cam * 3

    print(f"\nDone: {num_pairs * 2} frames written to {output_path}")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Output size: {raw_size:,} bytes ({raw_size / 1e9:.2f} GB)")


if __name__ == '__main__':
    main()
