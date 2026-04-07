#!/usr/bin/env python
"""Tensor Inversion Encoder."""
import sys, struct, bz2, time
import torch, einops, numpy as np
from pathlib import Path
from safetensors.torch import load_file
from frame_utils import AVVideoDataset, segnet_model_input_size
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <video_path> <output_dir>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    t0 = time.time()

    # Load models
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))

    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))

    for p in segnet.parameters():
        p.requires_grad_(False)
    for p in posenet.parameters():
        p.requires_grad_(False)

    print(f"Models loaded in {time.time() - t0:.1f}s")

    # Load video (AVVideoDataset requires CPU device)
    video_name = video_path.name
    data_dir = video_path.parent
    ds = AVVideoDataset(
        [video_name], data_dir=data_dir,
        batch_size=16, device=torch.device('cpu')
    )
    ds.prepare_data()

    all_seg_maps = []
    all_pose_vectors = []

    print("Extracting targets...")
    with torch.inference_mode():
        for path, idx, batch in ds:
            batch = batch.to(device)
            x = einops.rearrange(batch, 'b t h w c -> b t c h w').float()

            # SegNet: takes last frame, downscales, forward, argmax
            seg_in = segnet.preprocess_input(x)
            seg_out = segnet(seg_in)
            seg_maps = seg_out.argmax(dim=1).cpu().numpy().astype(np.uint8)
            all_seg_maps.append(seg_maps)

            # PoseNet: takes both frames, converts to YUV6, forward, take first 6 dims
            pn_in = posenet.preprocess_input(x)
            pn_out = posenet(pn_in)['pose'][:, :6].cpu().numpy()
            all_pose_vectors.append(pn_out)

            print(f"  Batch {idx}: {seg_maps.shape[0]} pairs processed")

    seg_maps = np.concatenate(all_seg_maps, axis=0)       # (N, 384, 512)
    pose_vectors = np.concatenate(all_pose_vectors, axis=0)  # (N, 6)
    num_pairs, orig_H, orig_W = seg_maps.shape

    print(f"Extracted {num_pairs} pairs: seg_maps {seg_maps.shape}, "
          f"pose_vectors {pose_vectors.shape}")
    print(f"Seg map class distribution: {np.bincount(seg_maps.flatten(), minlength=5)}")

    # Ideal class colors (pre-computed via gradient ascent, deterministic)
    ideal_colors = torch.tensor([
        [52.3731, 66.0825, 53.4251],
        [132.6272, 139.2837, 154.6401],
        [0.0000, 58.3693, 200.9493],
        [200.2360, 213.4126, 201.8910],
        [26.8595, 41.0758, 46.1465],
    ], device=device)
    print("Using hardcoded ideal class colors")

    # Row-interleaving groups the same row across all frames together, which
    # helps BWT (bz2) exploit temporal redundancy more effectively.
    # Optimal perm and level found via exhaustive search (1080 combos) offline.
    BEST_PERM = (2, 4, 1, 3, 0)
    BEST_LEVEL = 2

    ri_data = np.ascontiguousarray(seg_maps.transpose(1, 0, 2))
    perm_lut = np.array(BEST_PERM, dtype=np.uint8)
    seg_encoded = perm_lut[ri_data].tobytes()
    seg_compressed = bz2.compress(seg_encoded, BEST_LEVEL)
    print(f"Seg map compression: {seg_maps.nbytes:,} -> {len(seg_compressed):,} bytes "
          f"({len(seg_compressed)/1024:.1f} KB), perm={BEST_PERM}, level={BEST_LEVEL}")

    pose_f16 = pose_vectors.astype(np.float16).tobytes()
    pose_compressed = bz2.compress(pose_f16, 9)

    # seg.bin v3: [n:u32, H:u32, W:u32, flags:u32, perm:5xu8] + bz2 data
    # flags: bit 0-1 = compressor (2=bz2), bit 3 = row-interleaved, bit 4 = has permutation
    seg_flags = 2 | (1 << 3) | (1 << 4)
    with open(output_dir / 'seg.bin', 'wb') as f:
        f.write(struct.pack('<IIII', num_pairs, orig_H, orig_W, seg_flags))
        f.write(bytes(BEST_PERM))
        f.write(seg_compressed)

    # pose.bin: [n:u32, dims:u32, flags:u32] + bz2 data
    # flags: bit 0-1 = compressor (2=bz2), bit 2 = float16
    pose_flags = 2 | (1 << 2)
    with open(output_dir / 'pose.bin', 'wb') as f:
        f.write(struct.pack('<III', num_pairs, 6, pose_flags))
        f.write(pose_compressed)

    # Write colors.bin: raw float32, 5 classes x 3 channels
    (output_dir / 'colors.bin').write_bytes(
        ideal_colors.cpu().numpy().astype(np.float32).tobytes()
    )

    # Summary
    seg_file_size = 16 + 5 + len(seg_compressed)  # header + perm + data
    pose_file_size = 12 + len(pose_compressed)     # header + data
    total_data = seg_file_size + pose_file_size + 60
    elapsed = time.time() - t0
    uncompressed_size = 37_545_489

    print(f"\n{'='*50}")
    print(f"Encoding complete in {elapsed:.1f}s")
    print(f"  Seg maps (bz2, row-interleaved, perm): {seg_file_size:,} bytes")
    print(f"  Pose vectors (bz2, float16): {pose_file_size:,} bytes")
    print(f"  Ideal colors: 60 bytes")
    print(f"  Total data: {total_data:,} bytes ({total_data/1024:.1f} KB)")
    print(f"  Expected rate: {total_data / uncompressed_size:.6f}")
    print(f"  Expected rate contribution: {25 * total_data / uncompressed_size:.3f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
