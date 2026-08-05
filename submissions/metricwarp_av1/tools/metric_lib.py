#!/usr/bin/env python
"""Shared helpers: load metric nets, iterate raw-file pairs, compute per-pair metric outputs.

Matches evaluate.py numerics exactly (CPU float32, batch of pairs, same preprocessing).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import einops
from modules import DistortionNet, segnet_sd_path, posenet_sd_path
from frame_utils import camera_size

W, H = camera_size  # 1164, 874
FRAME_BYTES = H * W * 3

_torch_threads_set = False

def setup_threads(n=8):
    global _torch_threads_set
    if not _torch_threads_set:
        torch.set_num_threads(n)
        _torch_threads_set = True

def load_net(device='cpu'):
    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device(device))
    return net

def raw_pairs(path, batch_size=8, pair_indices=None):
    """Yield (pair_idx_array, batch tensor (B,2,H,W,3) uint8) from a .raw file.

    pair_indices: optional sorted list of pair indices to evaluate (subset eval).
    """
    path = Path(path)
    size = path.stat().st_size
    n_frames = size // FRAME_BYTES
    n_pairs = n_frames // 2
    mm = np.memmap(path, dtype=np.uint8, mode='r', shape=(n_frames, H, W, 3))
    idxs = np.arange(n_pairs) if pair_indices is None else np.asarray(pair_indices)
    for start in range(0, len(idxs), batch_size):
        chunk = idxs[start:start + batch_size]
        frames = np.stack([mm[2 * i:2 * i + 2] for i in chunk])  # (B,2,H,W,3)
        yield chunk, torch.from_numpy(frames)

@torch.inference_mode()
def net_outputs(net, batch):
    """batch: (B,2,H,W,3) uint8 tensor -> (pose (B,6) float32, seg_argmax (B,384,512) uint8)"""
    posenet_out, segnet_out = net(batch)
    pose = posenet_out['pose'][..., :6].float().cpu().numpy()
    seg = segnet_out.argmax(dim=1).to(torch.uint8).cpu().numpy()
    return pose, seg
