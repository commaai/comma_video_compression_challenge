"""
Dataset for training the neural compressor.

Yields pairs of consecutive frames (x_t, x_{t+1}) shape (3, H, W) uint8 RGB,
sampled randomly from the comma2k19 driving videos.

We use PyAV (cross-platform). For CUDA-only DALI, swap in DaliVideoDataset
from the parent repo's frame_utils.py.
"""
import os
import random
from pathlib import Path
from typing import List, Tuple

import av
import numpy as np
import torch
from torch.utils.data import IterableDataset

# Match the official challenge constants
CAMERA_W, CAMERA_H = 1164, 874


def yuv420_to_rgb_np(frame) -> np.ndarray:
    """Convert PyAV YUV420 frame to RGB uint8 (H, W, 3) using BT.601 limited."""
    H, W = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(H, frame.planes[0].line_size)[:, :W]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(H // 2, frame.planes[1].line_size)[:, :W // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(H // 2, frame.planes[2].line_size)[:, :W // 2]

    # bilinear chroma upsampling
    u_up = np.kron(u, np.ones((2, 2), dtype=np.uint8))[:H, :W]
    v_up = np.kron(v, np.ones((2, 2), dtype=np.uint8))[:H, :W]

    yf = (y.astype(np.float32) - 16.0) * (255.0 / 219.0)
    uf = (u_up.astype(np.float32) - 128.0) * (255.0 / 224.0)
    vf = (v_up.astype(np.float32) - 128.0) * (255.0 / 224.0)

    r = np.clip(yf + 1.402 * vf, 0, 255)
    g = np.clip(yf - 0.344136 * uf - 0.714136 * vf, 0, 255)
    b = np.clip(yf + 1.772 * uf, 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


class FramePairDataset(IterableDataset):
    """
    Streams consecutive frame pairs (x_t, x_{t+1}) from a directory of videos.

    Each item is a tensor of shape (2, 3, H, W) uint8 — two consecutive frames.
    The collate_fn stacks these into (B, 2, 3, H, W).
    """

    def __init__(
        self,
        video_paths: List[str],
        crop_size: Tuple[int, int] = None,  # (H, W); None = full frame
        shuffle_videos: bool = True,
        max_pairs_per_video: int = None,
    ):
        super().__init__()
        self.video_paths = list(video_paths)
        self.crop_size = crop_size
        self.shuffle_videos = shuffle_videos
        self.max_pairs_per_video = max_pairs_per_video

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        paths = list(self.video_paths)
        if self.shuffle_videos:
            random.shuffle(paths)

        # split work across DataLoader workers
        if worker_info is not None:
            paths = paths[worker_info.id::worker_info.num_workers]

        for path in paths:
            try:
                yield from self._iter_video(path)
            except Exception as e:
                print(f"[FramePairDataset] skipping {path}: {e}")

    def _iter_video(self, path: str):
        fmt = "hevc" if path.endswith(".hevc") else None
        container = av.open(path, format=fmt)
        stream = container.streams.video[0]

        prev = None
        emitted = 0
        for frame in container.decode(stream):
            cur = yuv420_to_rgb_np(frame)  # (H, W, 3) uint8
            if prev is not None:
                pair = np.stack([prev, cur], axis=0)  # (2, H, W, 3)
                pair = self._maybe_crop(pair)
                # → (2, 3, H, W)
                yield torch.from_numpy(pair).permute(0, 3, 1, 2).contiguous()
                emitted += 1
                if self.max_pairs_per_video and emitted >= self.max_pairs_per_video:
                    break
            prev = cur

        container.close()

    def _maybe_crop(self, pair: np.ndarray) -> np.ndarray:
        if self.crop_size is None:
            return pair
        ch, cw = self.crop_size
        _, h, w, _ = pair.shape
        if h == ch and w == cw:
            return pair
        top = random.randint(0, max(0, h - ch))
        left = random.randint(0, max(0, w - cw))
        return pair[:, top : top + ch, left : left + cw, :]


def list_videos(data_dir: str, exts=(".hevc", ".mkv", ".mp4")) -> List[str]:
    p = Path(data_dir)
    out = []
    for ext in exts:
        out.extend(sorted(str(x) for x in p.rglob(f"*{ext}")))
    return out


if __name__ == "__main__":
    # Quick sanity check
    import sys
    video_dir = sys.argv[1] if len(sys.argv) > 1 else "./test_videos"
    paths = list_videos(video_dir)
    print(f"Found {len(paths)} videos in {video_dir}")
    if not paths:
        sys.exit(0)
    ds = FramePairDataset(paths[:1], crop_size=(256, 256), max_pairs_per_video=3)
    for i, pair in enumerate(ds):
        print(f"pair {i}: {pair.shape} {pair.dtype}  min={pair.min()} max={pair.max()}")
        if i >= 2:
            break