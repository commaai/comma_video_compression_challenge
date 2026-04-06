#!/usr/bin/env python
"""Gradient-optimized inflation with on-the-fly label computation.

Instead of storing SegNet labels in the archive (costs ~500KB), we compute
them on-the-fly from the ORIGINAL video available at ROOT/videos/.
This gives us the same SegNet optimization benefit at no archive cost.

Key insight: SegNet evaluates only the LAST frame of each 2-frame pair.
With 1200 decoded frames, there are 600 pairs: (0,1), (2,3), (4,5)...
SegNet uses frames 1, 3, 5, ..., 1199 (odd-indexed).
We optimize only those 600 frames against computed labels.
"""
import av, torch, sys, os, time
import numpy as np
import torch.nn.functional as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from frame_utils import camera_size, yuv420_to_rgb, segnet_model_input_size

UNSHARP_KERNEL = torch.tensor([
    [1., 8., 28., 56., 70., 56., 28., 8., 1.],
    [8., 64., 224., 448., 560., 448., 224., 64., 8.],
    [28., 224., 784., 1568., 1960., 1568., 784., 224., 28.],
    [56., 448., 1568., 3136., 3920., 3136., 1568., 448., 56.],
    [70., 560., 1960., 3920., 4900., 3920., 1960., 560., 70.],
    [56., 448., 1568., 3136., 3920., 3136., 1568., 448., 56.],
    [28., 224., 784., 1568., 1960., 1568., 784., 224., 28.],
    [8., 64., 224., 448., 560., 448., 224., 64., 8.],
    [1., 8., 28., 56., 70., 56., 28., 8., 1.],
], dtype=torch.float32) / 65536.0


def decode_video(video_path, target_h, target_w):
    """Decode compressed video, upscale bicubic, apply unsharp mask."""
    fmt = 'hevc' if video_path.endswith('.hevc') else None
    container = av.open(video_path, format=fmt)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        t = yuv420_to_rgb(frame)
        H, W, _ = t.shape
        x = t.permute(2, 0, 1).unsqueeze(0).float()
        if H != target_h or W != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bicubic',
                              align_corners=False)
        kernel = UNSHARP_KERNEL.expand(3, 1, 9, 9)
        blur = F.conv2d(x, kernel, padding=4, groups=3)
        x = x + 0.85 * (x - blur)
        x = x.clamp(0, 255)
        frames.append(x.squeeze(0))
    container.close()
    return frames


def compute_segnet_labels(original_video_path, device, batch_size=16):
    """Compute SegNet class labels from the original video on-the-fly.

    SegNet uses the LAST frame of each 2-frame pair.
    Returns: (n_pairs, seg_h, seg_w) uint8 array of class indices.
    """
    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file

    seg_h, seg_w = segnet_model_input_size[1], segnet_model_input_size[0]

    segnet = SegNet().eval().to(device)
    sd = load_file(str(segnet_sd_path), device=str(device))
    segnet.load_state_dict(sd)

    # Decode original video frames
    fmt = 'hevc' if original_video_path.endswith('.hevc') else None
    container = av.open(original_video_path, format=fmt)
    stream = container.streams.video[0]

    all_labels = []
    frame_idx = 0
    with torch.inference_mode():
        batch_frames = []
        for frame in container.decode(stream):
            # Only process odd-indexed frames (last frame of each pair)
            if frame_idx % 2 == 1:
                t = yuv420_to_rgb(frame)
                # (H, W, 3) -> (3, H, W)
                x = t.permute(2, 0, 1).unsqueeze(0).float()
                # Resize to SegNet input size
                x = F.interpolate(x, size=(seg_h, seg_w), mode='bilinear',
                                  align_corners=False)
                batch_frames.append(x.squeeze(0))

                if len(batch_frames) == batch_size:
                    batch = torch.stack(batch_frames).to(device)
                    if device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            logits = segnet(batch)
                    else:
                        logits = segnet(batch)
                    labels = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
                    all_labels.append(labels)
                    batch_frames = []
            frame_idx += 1

        if batch_frames:
            batch = torch.stack(batch_frames).to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = segnet(batch)
            else:
                logits = segnet(batch)
            labels = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            all_labels.append(labels)

    container.close()

    seg_labels = np.concatenate(all_labels, axis=0)
    del segnet, sd
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

    return seg_labels


def gradient_optimize_segnet(frames, seg_labels, device, n_steps=30, lr=2.0,
                             batch_size=4, reg=0.01, delta_clamp=20):
    """Optimize odd-indexed frames (SegNet-evaluated) to match class predictions.

    Uses DELTA approach: optimize delta correction at 512x384, upscale, add to
    full-res frame. This preserves texture detail for PoseNet while fixing
    SegNet boundaries.
    """
    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file

    seg_h, seg_w = segnet_model_input_size[1], segnet_model_input_size[0]
    target_h, target_w = camera_size[1], camera_size[0]

    segnet = SegNet().eval().to(device)
    sd = load_file(str(segnet_sd_path), device=str(device))
    segnet.load_state_dict(sd)

    n_total = len(frames)
    n_pairs = n_total // 2
    assert n_pairs == seg_labels.shape[0], \
        f"Mismatch: {n_pairs} pairs but {seg_labels.shape[0]} labels"

    odd_indices = list(range(1, n_total, 2))  # [1, 3, 5, ..., 1199]
    total_batches = (len(odd_indices) + batch_size - 1) // batch_size

    for batch_start in range(0, len(odd_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(odd_indices))
        indices = odd_indices[batch_start:batch_end]
        B = len(indices)
        batch_num = batch_start // batch_size + 1

        # Downscale to SegNet resolution for optimization
        batch = torch.stack([frames[i] for i in indices]).to(device)
        batch_small = F.interpolate(batch, size=(seg_h, seg_w),
                                    mode='bilinear', align_corners=False)

        # Labels for these frames (pair index = frame_index // 2)
        label_indices = [i // 2 for i in indices]
        targets = torch.from_numpy(seg_labels[label_indices].copy()).long().to(device)

        # DELTA APPROACH: optimize a small correction at 512x384
        original_small = batch_small.clone().detach()
        delta = torch.zeros_like(batch_small, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()
            corrected = original_small + delta
            logits = segnet(corrected)
            loss = F.cross_entropy(logits, targets)
            loss = loss + reg * (delta ** 2).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.clamp_(-delta_clamp, delta_clamp)
                (original_small + delta).clamp_(0, 255)

        # Apply delta: upscale correction and add to original full-res frames
        with torch.no_grad():
            delta_small = delta.detach()
            delta_full = F.interpolate(delta_small, size=(target_h, target_w),
                                       mode='bicubic', align_corners=False)
            for j, idx in enumerate(indices):
                corrected = (batch[j] + delta_full[j]).clamp(0, 255)
                frames[idx] = corrected.cpu()

        del delta, targets, batch, batch_small, original_small, optimizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

        if batch_num % 10 == 0 or batch_num == 1 or batch_num == total_batches:
            print(f"  Batch {batch_num}/{total_batches} "
                  f"(frames {indices[0]}-{indices[-1]})", flush=True)

    del segnet, sd
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

    return frames


def optimize_frame(frame_chw, seg_label, segnet, device, n_steps=30, lr=2.0, reg=0.01):
    """Optimize a single frame using delta approach."""
    seg_h, seg_w = segnet_model_input_size[1], segnet_model_input_size[0]
    target_h, target_w = camera_size[1], camera_size[0]

    frame_gpu = frame_chw.unsqueeze(0).to(device)
    frame_small = F.interpolate(frame_gpu, size=(seg_h, seg_w),
                                 mode='bilinear', align_corners=False).squeeze(0)
    original_small = frame_small.clone().detach()
    target = torch.from_numpy(seg_label[np.newaxis].copy()).long().to(device)

    delta = torch.zeros_like(original_small, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    use_amp = device.type == 'cuda'
    for step in range(n_steps):
        optimizer.zero_grad()
        corrected = (original_small + delta).unsqueeze(0)
        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = segnet(corrected)
                loss = F.cross_entropy(logits, target) + reg * (delta ** 2).mean()
        else:
            logits = segnet(corrected)
            loss = F.cross_entropy(logits, target) + reg * (delta ** 2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-20, 20)

    with torch.no_grad():
        delta_full = F.interpolate(delta.unsqueeze(0), size=(target_h, target_w),
                                    mode='bicubic', align_corners=False).squeeze(0)
        result = (frame_chw.to(device) + delta_full).clamp(0, 255).cpu()
    del delta, target, frame_gpu, frame_small, original_small
    return result


def decode_frame_single(frame, target_h, target_w):
    """Decode one frame: YUV->RGB, upscale, unsharp."""
    t = yuv420_to_rgb(frame)
    H, W, _ = t.shape
    x = t.permute(2, 0, 1).unsqueeze(0).float()
    if H != target_h or W != target_w:
        x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
    kernel = UNSHARP_KERNEL.expand(3, 1, 9, 9)
    blur = F.conv2d(x, kernel, padding=4, groups=3)
    x = (x + 0.85 * (x - blur)).clamp(0, 255)
    return x.squeeze(0)


def main(compressed_path, dst, video_name):
    """Streaming version: process frame-by-frame, no OOM."""
    target_w, target_h = camera_size
    t_start = time.time()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        n_steps = 25
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        n_steps = 30
    else:
        device = torch.device('cpu')
        n_steps = 15
    print(f"Device: {device}, steps={n_steps}", flush=True)
    TIME_BUDGET = 20 * 60  # 20 minutes max for inflate (leave 10 for setup+eval)

    # Step 1: Compute SegNet labels from original video (streaming, low memory)
    original_path = str(ROOT / "videos" / video_name)
    print(f"Computing labels from original: {original_path}", flush=True)
    t_label = time.time()
    seg_labels = compute_segnet_labels(original_path, device, batch_size=16)
    print(f"Labels: {seg_labels.shape} ({time.time()-t_label:.1f}s)", flush=True)

    # Step 2: Load SegNet for optimization
    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file
    segnet = SegNet().eval().to(device)
    sd = load_file(str(segnet_sd_path), device=str(device))
    segnet.load_state_dict(sd)
    del sd
    print("SegNet loaded for optimization", flush=True)

    # Step 3: Stream-process compressed video
    fmt = 'hevc' if compressed_path.endswith('.hevc') else None
    container = av.open(compressed_path, format=fmt)
    stream = container.streams.video[0]

    pair_idx = 0
    frame_idx = 0
    n_written = 0
    with open(dst, 'wb') as f:
        for frame in container.decode(stream):
            decoded = decode_frame_single(frame, target_h, target_w)

            if frame_idx % 2 == 1 and pair_idx < seg_labels.shape[0]:
                elapsed = time.time() - t_start
                if elapsed < TIME_BUDGET:
                    # Reduce steps if running low on time
                    remaining_pairs = seg_labels.shape[0] - pair_idx
                    time_left = TIME_BUDGET - elapsed
                    if remaining_pairs > 0:
                        time_per_pair = elapsed / max(pair_idx, 1)
                        if time_per_pair * remaining_pairs > time_left and n_steps > 10:
                            adaptive_steps = max(10, int(n_steps * time_left / (time_per_pair * remaining_pairs)))
                        else:
                            adaptive_steps = n_steps
                    else:
                        adaptive_steps = n_steps
                    decoded = optimize_frame(decoded, seg_labels[pair_idx], segnet,
                                              device, n_steps=adaptive_steps, lr=2.0, reg=0.01)
                pair_idx += 1

            frame_hwc = decoded.permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
            f.write(frame_hwc.contiguous().numpy().tobytes())
            n_written += 1
            frame_idx += 1

            if frame_idx % 100 == 0:
                elapsed = time.time() - t_start
                fps = frame_idx / elapsed
                eta = (1200 - frame_idx) / fps if fps > 0 else 0
                print(f"  Frame {frame_idx}, {fps:.1f} fps, ETA {eta:.0f}s, steps={n_steps}", flush=True)

    container.close()
    del segnet
    print(f"Done: {n_written} frames, total {time.time()-t_start:.1f}s", flush=True)
    return n_written


if __name__ == "__main__":
    compressed_path = sys.argv[1]
    dst = sys.argv[2]
    video_name = sys.argv[3]

    n = main(compressed_path, dst, video_name)
    print(f"Saved {n} frames", flush=True)
