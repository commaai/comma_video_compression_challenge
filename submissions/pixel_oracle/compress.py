#!/usr/bin/env python
"""
Compression: extracts SegNet class maps and PoseNet pose targets,
creates a tiny seed video for initialization.
"""
import sys, os, struct, bz2
import numpy as np
import torch
import torch.nn.functional as F
import einops

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from frame_utils import yuv420_to_rgb, camera_size, segnet_model_input_size
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

import av as av_lib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')


def rgb_to_yuv6_diff(rgb_chw):
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


def extract_all_frames(video_path):
    container = av_lib.open(video_path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = yuv420_to_rgb(frame)
        frames.append(rgb)
    container.close()
    return frames


def compute_targets(frames, device):
    dn = DistortionNet().eval().to(device)
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    n_frames = len(frames)
    n_pairs = n_frames // 2
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]

    segnet_classes = []
    posenet_poses = []

    batch_size = 8
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        B = end - start
        batch_pairs = []
        for i in range(start, end):
            f0 = frames[i * 2].to(device)
            f1 = frames[i * 2 + 1].to(device)
            pair = torch.stack([f0, f1])
            batch_pairs.append(pair)
        batch = torch.stack(batch_pairs).float()

        with torch.no_grad():
            x = einops.rearrange(batch, 'b t h w c -> b t c h w')
            segnet_in = F.interpolate(x[:, -1], size=(mH, mW), mode='bilinear')
            segnet_out = dn.segnet(segnet_in)
            classes = segnet_out.argmax(dim=1).cpu()

            px = einops.rearrange(x, 'b t c h w -> (b t) c h w')
            px = F.interpolate(px, size=(mH, mW), mode='bilinear')
            px_yuv = einops.rearrange(rgb_to_yuv6_diff(px), '(b t) c h w -> b (t c) h w', b=B, t=2, c=6)
            poses = dn.posenet(px_yuv)['pose'][:, :6].cpu()

        for j in range(B):
            segnet_classes.append(classes[j])
            posenet_poses.append(poses[j])

        if (start // batch_size) % 10 == 0:
            print(f"  Pairs {start}-{end}/{n_pairs}")

    return torch.stack(segnet_classes), torch.stack(posenet_poses)


def encode_seed_video(frames, output_path, scale=0.20, crf=50):
    """Tiny seed video for initialization — quality handled by optimization."""
    H, W = frames[0].shape[0], frames[0].shape[1]
    new_H = int(H * scale) // 2 * 2
    new_W = int(W * scale) // 2 * 2

    container = av_lib.open(output_path, mode='w')
    stream = container.add_stream('libsvtav1', rate=20)
    stream.width = new_W
    stream.height = new_H
    stream.pix_fmt = 'yuv420p'
    stream.options = {
        'crf': str(crf),
        'preset': '8',
        'svtav1-params': 'keyint=240:scd=0'
    }

    from PIL import Image
    for i, frame_tensor in enumerate(frames):
        img = frame_tensor.numpy()
        pil_img = Image.fromarray(img).resize((new_W, new_H), Image.LANCZOS)
        vf = av_lib.VideoFrame.from_ndarray(np.array(pil_img), format='rgb24')
        vf.pts = i
        for pkt in stream.encode(vf):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    print(f"  Seed video: {os.path.getsize(output_path)} bytes at {new_W}x{new_H}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=os.path.join(ROOT, 'videos', '0.mkv'))
    parser.add_argument('--output-dir', default=os.path.join(HERE, 'archive_contents'))
    parser.add_argument('--scale', type=float, default=0.30)
    parser.add_argument('--crf', type=int, default=45)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Extracting frames...")
    frames = extract_all_frames(args.video)
    print(f"  {len(frames)} frames")

    print("Computing targets...")
    class_maps, pose_targets = compute_targets(frames, device)

    print("Encoding seed video...")
    encode_seed_video(frames, os.path.join(args.output_dir, '0.mkv'),
                      scale=args.scale, crf=args.crf)

    print("Saving class maps (bz2)...")
    data = class_maps.numpy().astype(np.uint8)
    compressed = bz2.compress(data.tobytes(), 9)
    with open(os.path.join(args.output_dir, '0.segmap'), 'wb') as f:
        f.write(struct.pack('<III', *data.shape))
        f.write(compressed)
    print(f"  Class maps: {12 + len(compressed)} bytes")

    print("Saving pose targets (bz2)...")
    pdata = pose_targets.numpy().astype(np.float32)
    pcomp = bz2.compress(pdata.tobytes(), 9)
    with open(os.path.join(args.output_dir, '0.pose'), 'wb') as f:
        f.write(struct.pack('<II', *pdata.shape))
        f.write(pcomp)
    print(f"  Pose targets: {8 + len(pcomp)} bytes")

    # Create archive.zip
    import zipfile
    archive_path = os.path.join(HERE, 'archive.zip')
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_STORED) as zf:
        for fname in os.listdir(args.output_dir):
            fpath = os.path.join(args.output_dir, fname)
            zf.write(fpath, fname)

    total = os.path.getsize(archive_path)
    orig = os.path.getsize(args.video)
    print(f"\nArchive: {total} bytes ({total/1024:.1f} KB)")
    print(f"Rate: {total / orig:.6f}")
    print(f"Rate score: {25 * total / orig:.4f}")


if __name__ == '__main__':
    main()
