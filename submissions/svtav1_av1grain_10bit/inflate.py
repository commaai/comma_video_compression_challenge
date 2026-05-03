#!/usr/bin/env python
"""Decode compressed AV1 video -> raw uint8 RGB frames.

Handles both 8-bit and 10-bit YUV420 AV1 content.
Uses CUDA if available, falls back to CPU.
"""
import sys
import logging
import numpy as np
import torch
import torch.nn.functional as F
import av

log = logging.getLogger(__name__)

TARGET_W, TARGET_H = 1164, 874
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def yuv420_to_rgb(frame, device):
    """BT.601 limited-range YUV420p -> RGB uint8 (H, W, 3). Handles 8-bit and 10-bit."""
    H, W = frame.height, frame.width
    y_plane = frame.planes[0]
    u_plane = frame.planes[1]
    v_plane = frame.planes[2]

    fmt = frame.format.name
    is_10bit = "10" in fmt or "p10" in fmt

    if is_10bit:
        dtype = np.uint16
        y = np.frombuffer(y_plane, dtype=dtype).reshape(H, y_plane.line_size // 2)[:, :W]
        u = np.frombuffer(u_plane, dtype=dtype).reshape(H // 2, u_plane.line_size // 2)[:, :W // 2]
        v = np.frombuffer(v_plane, dtype=dtype).reshape(H // 2, v_plane.line_size // 2)[:, :W // 2]
        s = 255.0 / 1023.0
        y_t = torch.from_numpy(y.copy().astype(np.float32) * s).to(device)
        u_t = torch.from_numpy(u.copy().astype(np.float32) * s).to(device).unsqueeze(0).unsqueeze(0)
        v_t = torch.from_numpy(v.copy().astype(np.float32) * s).to(device).unsqueeze(0).unsqueeze(0)
    else:
        y = np.frombuffer(y_plane, dtype=np.uint8).reshape(H, y_plane.line_size)[:, :W]
        u = np.frombuffer(u_plane, dtype=np.uint8).reshape(H // 2, u_plane.line_size)[:, :W // 2]
        v = np.frombuffer(v_plane, dtype=np.uint8).reshape(H // 2, v_plane.line_size)[:, :W // 2]
        y_t = torch.from_numpy(y.copy()).to(device).float()
        u_t = torch.from_numpy(u.copy()).to(device).float().unsqueeze(0).unsqueeze(0)
        v_t = torch.from_numpy(v.copy()).to(device).float().unsqueeze(0).unsqueeze(0)

    u_up = F.interpolate(u_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()
    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)
    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


def decode_and_resize_to_file(video_path, dst):
    """Decode compressed video, upscale, sharpen, and write raw RGB frames."""
    device = DEVICE
    log.info("Using device: %s", device)

    container = av.open(video_path)
    stream = container.streams.video[0]

    # Pre-compute unsharp kernel (5x5, sigma=1.0)
    ax = torch.arange(5, dtype=torch.float32, device=device) - 2
    k1d = torch.exp(-0.5 * (ax / 1.0) ** 2)
    k1d = k1d / k1d.sum()
    kernel = (k1d.unsqueeze(1) * k1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
    amount = 2.0

    n = 0
    with open(dst, "wb") as f:
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame, device)
            H, W, _ = t.shape
            if H != TARGET_H or W != TARGET_W:
                x = t.permute(2, 0, 1).unsqueeze(0).float()
                x = F.interpolate(x, size=(TARGET_H, TARGET_W),
                                  mode="bicubic", align_corners=False)
                blurred = F.conv2d(x, kernel, padding=2, groups=3)
                x = (x + amount * (x - blurred)).clamp(0, 255)
                t = x.squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
            f.write(t.cpu().contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    src, dst = sys.argv[1], sys.argv[2]
    n = decode_and_resize_to_file(src, dst)
    print(f"saved {n} frames")
