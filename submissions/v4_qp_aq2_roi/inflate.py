#!/usr/bin/env python
import sys, torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from frame_utils import camera_size, yuv420_to_rgb
import av


def decode_and_resize_to_file(video_path, dst):
    target_w, target_h = camera_size
    container = av.open(video_path)
    stream = container.streams.video[0]
    n = 0
    with open(dst, "wb") as f:
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame)
            h, w, _ = t.shape
            if h != target_h or w != target_w:
                x = t.permute(2, 0, 1).unsqueeze(0).float()
                x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = decode_and_resize_to_file(src, dst)
    print(f"saved {n} frames")
