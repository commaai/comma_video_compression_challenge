"""Video frame loading and ground-truth precompute.

Reads the comma.ai challenge video (single hevc file, 1164×874, ~1200 frames),
groups consecutive frames into pairs (n_pairs = 600), and runs the frozen
SegNet/PoseNet (DistortionNet from the challenge repo) once to cache:
  - seg_targets_hard:  (n_pairs, 384, 512) int64 — SegNet argmax labels
  - pose_targets:      (n_pairs, 6) float32     — PoseNet pose vectors
  - gt_pairs_half:     (n_pairs, 2, 3, 192, 256) float32 CPU — half-res GT for multi-res L1

The challenge repo is imported as a stable third-party (provides the official
SegNet/PoseNet weights and the YUV420 decode path that matches the eval harness).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _resolve_challenge_root():
    """Find the comma_video_compression_challenge repo. Search up from this file
    and from CWD; allow override via env var COMMA_CHALLENGE_ROOT."""
    import os
    if 'COMMA_CHALLENGE_ROOT' in os.environ:
        return Path(os.environ['COMMA_CHALLENGE_ROOT']).resolve()
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        cand = parent / 'comma_video_compression_challenge'
        if (cand / 'frame_utils.py').exists():
            return cand
    cand = Path.cwd() / 'comma_video_compression_challenge'
    if (cand / 'frame_utils.py').exists():
        return cand.resolve()
    raise FileNotFoundError(
        "comma_video_compression_challenge/ not found. Set COMMA_CHALLENGE_ROOT or "
        "place the challenge repo as a sibling of my_submission/.")


CHALLENGE_ROOT = _resolve_challenge_root()
sys.path.insert(0, str(CHALLENGE_ROOT.parent))
sys.path.insert(0, str(CHALLENGE_ROOT))

import av
import frame_utils  # noqa: E402
import modules  # noqa: E402
from frame_utils import yuv420_to_rgb  # noqa: E402
from modules import DistortionNet, segnet_sd_path, posenet_sd_path  # noqa: E402


def _rgb_to_yuv6_differentiable(rgb_chw):
    """Differentiable BT.601 RGB->YUV6 (matches frame_utils.rgb_to_yuv6 numerically
    but without the @torch.no_grad() decorator and with out-of-place clamp).

    The challenge's frame_utils.rgb_to_yuv6 is wrapped in @torch.no_grad() and
    uses clamp_() in-place, which severs the autograd graph. PoseNet's
    preprocess_input calls rgb_to_yuv6 on every forward — so without this
    patch the pose loss gradient never reaches the decoder, and pose stays
    pinned at its random-init value through training. This was a real bug
    in v1/v2 (pose plateaued at 142 across 2500+ epochs).
    """
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, :2*H2, :2*W2]
    R, G, B = rgb[..., 0, :, :], rgb[..., 1, :, :], rgb[..., 2, :, :]
    Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
    U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2]
             + U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
    V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2]
             + V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
    return torch.stack([Y[..., 0::2, 0::2], Y[..., 1::2, 0::2],
                        Y[..., 0::2, 1::2], Y[..., 1::2, 1::2],
                        U_sub, V_sub], dim=-3)


# Apply the patch at import time. modules.py already imported rgb_to_yuv6 from
# frame_utils, so we have to overwrite BOTH module-level references.
frame_utils.rgb_to_yuv6 = _rgb_to_yuv6_differentiable
modules.rgb_to_yuv6 = _rgb_to_yuv6_differentiable


CAMERA_SIZE = (1164, 874)  # (W, H) — challenge convention
EVAL_SIZE = (384, 512)     # (H, W) — decoder native output size
MULTIRES_SIZE = (192, 256)


def load_distortion_net(device):
    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in net.parameters():
        p.requires_grad = False
    return net


def precompute_targets(video_path, device, multires_size=MULTIRES_SIZE):
    """Stream-decode one video, compute SegNet/PoseNet targets per pair, return
    (seg_targets_hard, pose_targets, gt_pairs_half_cpu, n_pairs).

    The half-res GT pairs are kept on CPU to bound memory; per-batch transfer is
    cheap. seg/pose targets are kept on GPU.
    """
    distortion_net = load_distortion_net(device)
    container = av.open(str(video_path))
    seg_targets_hard, pose_targets, gt_pairs_half = [], [], []
    prev = None
    with torch.inference_mode():
        for frame in container.decode(container.streams.video[0]):
            f = yuv420_to_rgb(frame)
            if prev is None:
                prev = f
                continue
            f0, f1 = prev, f
            prev = None
            pair = torch.stack([f0, f1]).unsqueeze(0).to(device)
            po, so = distortion_net(pair)
            seg_targets_hard.append(so.argmax(dim=1).squeeze(0).clone())
            pose_targets.append(po['pose'][:, :6].float().squeeze(0).clone())
            gt2 = torch.stack([f0, f1]).float().permute(0, 3, 1, 2).contiguous()
            gt2_half = F.interpolate(gt2, size=multires_size, mode='bilinear',
                                     align_corners=False)
            gt_pairs_half.append(gt2_half)
            del f0, f1, f, pair, po, so, gt2, gt2_half
    container.close()
    torch.cuda.empty_cache()

    seg_targets_hard = torch.stack(seg_targets_hard)
    pose_targets = torch.stack(pose_targets)
    gt_pairs_half_cpu = torch.stack(gt_pairs_half).contiguous()
    n_pairs = seg_targets_hard.shape[0]
    return distortion_net, seg_targets_hard, pose_targets, gt_pairs_half_cpu, n_pairs


def video_paths_from_names_file(names_file: Path, videos_dir: Path):
    with open(names_file) as f:
        names = [l.strip() for l in f if l.strip()]
    return [videos_dir / n for n in names]


def get_default_video_path():
    """Returns the canonical first video for training (single-video memorization regime).

    Tries (in order): video_names.txt, public_test_video_names.txt; else falls
    back to the first .hevc file in videos/.
    """
    videos_dir = CHALLENGE_ROOT / 'videos'
    for cand in ('video_names.txt', 'public_test_video_names.txt'):
        names_file = CHALLENGE_ROOT / cand
        if names_file.exists():
            paths = video_paths_from_names_file(names_file, videos_dir)
            existing = [p for p in paths if p.exists()]
            if existing:
                return existing[0]
    # Fallback: any .hevc in videos/
    hevcs = sorted(videos_dir.glob('*.hevc'))
    if hevcs:
        return hevcs[0]
    raise FileNotFoundError(f"No videos found in {videos_dir}")
