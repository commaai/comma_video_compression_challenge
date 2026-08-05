#!/usr/bin/env python
"""Full-range BT.601 RGB<->YUV420 with box chroma down / bilinear chroma up.

The DECODE direction defined here is the contract for inflate.py; the encode
direction is chosen to minimize roundtrip error under that decode.
Supports 8-bit and 10-bit YUV.
"""
import numpy as np
import torch
import torch.nn.functional as F

def rgb_to_yuv(rgb, bits=10, subsample=True):
    """rgb: (N,H,W,3) uint8. Returns (Y, U, V) planes; chroma half-res if subsample."""
    x = rgb.astype(np.float32)
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = (B - Y) / 1.772 + 128.0
    V = (R - Y) / 1.402 + 128.0
    if subsample:
        U = (U[:, 0::2, 0::2] + U[:, 1::2, 0::2] + U[:, 0::2, 1::2] + U[:, 1::2, 1::2]) * 0.25
        V = (V[:, 0::2, 0::2] + V[:, 1::2, 0::2] + V[:, 0::2, 1::2] + V[:, 1::2, 1::2]) * 0.25
    scale = (1 << (bits - 8))
    dt = np.uint16 if bits > 8 else np.uint8
    maxv = (1 << bits) - 1
    to = lambda p: np.clip(np.round(p * scale), 0, maxv).astype(dt)
    return to(Y), to(U), to(V)

def rgb_to_yuv420(rgb, bits=10):
    return rgb_to_yuv(rgb, bits=bits, subsample=True)

def yuv420_to_rgb(Y, U, V, bits=10, chunk=64):
    """Y: (N,H,W), U/V: (N,H,W) or (N,H/2,W/2) int arrays -> (N,H,W,3) uint8.
    Bilinear chroma upsample if subsampled (align_corners=False), full-range BT.601 inverse.
    Chunked to bound memory."""
    scale = float(1 << (bits - 8))
    N, H, W = np.asarray(Y).shape
    out = np.empty((N, H, W, 3), dtype=np.uint8)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        Yf = torch.from_numpy(np.asarray(Y[s:e], dtype=np.float32)) / scale
        Uf = torch.from_numpy(np.asarray(U[s:e], dtype=np.float32)) / scale
        Vf = torch.from_numpy(np.asarray(V[s:e], dtype=np.float32)) / scale
        if Uf.shape[1] == H:
            Uu, Vu = Uf, Vf
        else:
            Uu = F.interpolate(Uf.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
            Vu = F.interpolate(Vf.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
        R = Yf + 1.402 * (Vu - 128.0)
        G = Yf - 0.344136 * (Uu - 128.0) - 0.714136 * (Vu - 128.0)
        B = Yf + 1.772 * (Uu - 128.0)
        rgb = torch.stack([R, G, B], dim=-1).clamp(0, 255).round().to(torch.uint8)
        out[s:e] = rgb.numpy()
    return out

def write_y4m(path, Y, U, V, bits=10, fps=20):
    """Write a y4m file from YUV planes ((N,H,W),(N,h2,w2),(N,h2,w2))."""
    N, H, W = Y.shape
    csp = 'C420p10' if bits == 10 else '420mpeg2'
    hdr = f"YUV4MPEG2 W{W} H{H} F{fps}:1 Ip A1:1 {csp} XYSCSS={'420P10' if bits==10 else '420MPEG2'}\n"
    dt = np.uint16 if bits > 8 else np.uint8
    with open(path, 'wb') as f:
        f.write(hdr.encode())
        for i in range(N):
            f.write(b'FRAME\n')
            f.write(np.ascontiguousarray(Y[i], dtype=dt).tobytes())
            f.write(np.ascontiguousarray(U[i], dtype=dt).tobytes())
            f.write(np.ascontiguousarray(V[i], dtype=dt).tobytes())

def decode_video_yuv(path):
    """Decode a video file with PyAV; return (Y,U,V) planes as int arrays plus bit depth."""
    import av
    container = av.open(str(path))
    stream = container.streams.video[0]
    Ys, Us, Vs = [], [], []
    bits = 8
    for frame in container.decode(stream):
        name = frame.format.name
        if '10' in name:
            bits = 10
            dt = np.uint16
        else:
            dt = np.uint8
        H, Wd = frame.height, frame.width
        esz = 2 if bits == 10 else 1
        c444 = '444' in name
        ch, cw = (H, Wd) if c444 else (H // 2, Wd // 2)
        y = np.frombuffer(frame.planes[0], dtype=dt).reshape(H, frame.planes[0].line_size // esz)[:, :Wd]
        u = np.frombuffer(frame.planes[1], dtype=dt).reshape(ch, frame.planes[1].line_size // esz)[:, :cw]
        v = np.frombuffer(frame.planes[2], dtype=dt).reshape(ch, frame.planes[2].line_size // esz)[:, :cw]
        Ys.append(y.copy()); Us.append(u.copy()); Vs.append(v.copy())
    container.close()
    return np.stack(Ys), np.stack(Us), np.stack(Vs), bits

def decode_video_rgb(path):
    """Decode with PyAV and convert to RGB target space via our inverse transform."""
    Y, U, V, bits = decode_video_yuv(path)
    return yuv420_to_rgb(Y, U, V, bits=bits)
