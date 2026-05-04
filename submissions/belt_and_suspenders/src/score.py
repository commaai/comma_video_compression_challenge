"""Score computation matching the official challenge metric.

Score = 100 · seg_distortion + sqrt(10 · pose_distortion) + 25 · rate

  - seg_distortion: from DistortionNet.compute_distortion (frozen SegNet)
  - pose_distortion: from DistortionNet.compute_distortion (frozen PoseNet)
  - rate: archive_bytes / sum(video file sizes) per challenge evaluate.py:65

Eval runs the decoder, bicubic-upsamples (384,512) outputs to (874,1164) camera
size, and calls DistortionNet.compute_distortion against the GT video.

Stream-decoded so we never hold all 1200 full-res frames in memory at once
(challenge frames at 1164×874×3 uint8 = ~3 MB each).
"""
import math
from pathlib import Path

import av
import torch
import torch.nn.functional as F


CAMERA_H, CAMERA_W = 874, 1164
EVAL_H, EVAL_W = 384, 512


def _decoded_to_camera(decoded_native, target_h=CAMERA_H, target_w=CAMERA_W):
    """Resample (B*2, 3, EVAL_H, EVAL_W) -> (B*2, 3, CAMERA_H, CAMERA_W) bicubic.
    Matches eval harness resample chain."""
    return F.interpolate(decoded_native, size=(target_h, target_w),
                         mode='bicubic', align_corners=False)


@torch.inference_mode()
def evaluate_decoder(decoder, latents, distortion_net, video_path,
                     batch_pairs=8, device='cuda'):
    """Stream-decode the GT video and the decoder output simultaneously, accumulate
    per-pair distortions via DistortionNet.compute_distortion.

    Returns:
        dict with seg_distortion (mean), pose_distortion (mean)
    """
    from frame_utils import yuv420_to_rgb

    decoder.eval()
    n_pairs = latents.shape[0]

    # Stream GT pairs from video
    container = av.open(str(video_path))
    gt_pairs_iter_state = {'prev': None, 'pair_idx': 0}
    seg_total = 0.0
    pose_total = 0.0
    count = 0

    def next_gt_pair():
        """Yield one (2, H, W, 3) uint8 GT pair from the stream, or None when done."""
        # Drain frames into pairs lazily
        for frame in gt_pairs_iter_state.get('frames', iter(())):
            f = yuv420_to_rgb(frame)
            if gt_pairs_iter_state['prev'] is None:
                gt_pairs_iter_state['prev'] = f
                continue
            f0, f1 = gt_pairs_iter_state['prev'], f
            gt_pairs_iter_state['prev'] = None
            return torch.stack([f0, f1])
        return None

    # Build flat frame iterator
    gt_pairs_iter_state['frames'] = container.decode(container.streams.video[0])

    pair_idx = 0
    while pair_idx < n_pairs:
        # Collect a batch of GT pairs
        batch_gt = []
        for _ in range(min(batch_pairs, n_pairs - pair_idx)):
            pair = next_gt_pair()
            if pair is None:
                break
            batch_gt.append(pair)
        if not batch_gt:
            break
        batch_gt = torch.stack(batch_gt).to(device)  # (B, 2, H, W, 3) uint8
        B = batch_gt.shape[0]

        # Run decoder on the matching latent batch
        idx = torch.arange(pair_idx, pair_idx + B, device=device)
        z = latents[idx]
        decoded = decoder(z)  # (B, 2, 3, EVAL_H, EVAL_W) float in [0,255]
        flat = decoded.reshape(B * 2, 3, EVAL_H, EVAL_W)
        up = _decoded_to_camera(flat)
        decoded_bhwc = (up.reshape(B, 2, 3, CAMERA_H, CAMERA_W)
                          .permute(0, 1, 3, 4, 2)
                          .clamp(0, 255).round().to(torch.uint8))

        pose_d, seg_d = distortion_net.compute_distortion(batch_gt, decoded_bhwc)
        seg_total += seg_d.sum().item()
        pose_total += pose_d.sum().item()
        count += B
        pair_idx += B

    container.close()
    return {
        'seg_distortion': seg_total / max(count, 1),
        'pose_distortion': pose_total / max(count, 1),
    }


def compute_score(seg_dist, pose_dist, archive_bytes, total_video_bytes):
    """Final challenge metric. Lower is better."""
    rate = archive_bytes / total_video_bytes
    seg_component = 100.0 * seg_dist
    pose_component = math.sqrt(10.0 * pose_dist + 1e-12)
    rate_component = 25.0 * rate
    return {
        'score': seg_component + pose_component + rate_component,
        'seg_component': seg_component,
        'pose_component': pose_component,
        'rate_component': rate_component,
        'seg_distortion': seg_dist,
        'pose_distortion': pose_dist,
        'rate': rate,
        'archive_bytes': archive_bytes,
    }


def total_video_bytes(video_paths):
    """Rate denominator per challenge eval (evaluate.py:64): sum of source video
    file sizes."""
    if isinstance(video_paths, (str, Path)):
        video_paths = [video_paths]
    return sum(Path(p).stat().st_size for p in video_paths)
