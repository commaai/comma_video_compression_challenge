"""
Compress: input videos -> archive/  (then zipped to archive.zip externally).

archive/ layout:
    archive/
        model.pt                 # trained compressor weights (fp16, no optimizer state)
        meta.json                # arch hyperparams (so decompress can rebuild model)
        streams/<name>.bin       # per-video bitstream (pickle)

Bitstream content per video (one big pickle):
    {
        "frames": [
            {"strings": [y_str, z_str], "shape": (h, w), "pad": (l,r,t,b), "orig_hw": (H,W)},
            ...
        ],
        "frame_count": N,
        "fps_num": ..., "fps_den": ...,   # so we can write the right .mkv on inflate
    }

We encode frames independently (intra-only). Conditional/temporal coding is a
straightforward extension once intra-only works.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import av
import numpy as np
import torch

from compressor import ScaleHyperpriorCompressor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True, help="Dir of original videos.")
    p.add_argument("--video_names", type=str, required=True,
                   help="Text file listing video filenames to process (one per line).")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to compressor_*.pt")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory (becomes archive/).")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dtype", type=str, default="fp16", choices=["fp32", "fp16"])
    return p.parse_args()


def load_compressor(ckpt_path: str, device: torch.device) -> ScaleHyperpriorCompressor:
    blob = torch.load(ckpt_path, map_location=device)
    args = blob.get("args", {})
    N = args.get("N", 64)
    M = args.get("M", 128)
    N_hyp = args.get("N_hyp", 64)
    model = ScaleHyperpriorCompressor(N=N, M=M, N_hyp=N_hyp).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    # Build CDF tables for arithmetic coding
    model.update(force=True)
    return model, dict(N=N, M=M, N_hyp=N_hyp)


def yuv420_to_rgb(frame) -> np.ndarray:
    """Same as in dataset.py — kept here so compress.py is self-contained."""
    H, W = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(H, frame.planes[0].line_size)[:, :W]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(H // 2, frame.planes[1].line_size)[:, :W // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(H // 2, frame.planes[2].line_size)[:, :W // 2]
    u_up = np.kron(u, np.ones((2, 2), dtype=np.uint8))[:H, :W]
    v_up = np.kron(v, np.ones((2, 2), dtype=np.uint8))[:H, :W]
    yf = (y.astype(np.float32) - 16.0) * (255.0 / 219.0)
    uf = (u_up.astype(np.float32) - 128.0) * (255.0 / 224.0)
    vf = (v_up.astype(np.float32) - 128.0) * (255.0 / 224.0)
    r = np.clip(yf + 1.402 * vf, 0, 255)
    g = np.clip(yf - 0.344136 * uf - 0.714136 * vf, 0, 255)
    b = np.clip(yf + 1.772 * uf, 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def iter_frames(video_path: str):
    fmt = "hevc" if video_path.endswith(".hevc") else None
    container = av.open(video_path, format=fmt)
    stream = container.streams.video[0]
    fps = stream.average_rate
    yield ("meta", {"fps_num": fps.numerator, "fps_den": fps.denominator})
    for frame in container.decode(stream):
        yield ("frame", yuv420_to_rgb(frame))  # (H, W, 3) uint8
    container.close()


@torch.no_grad()
def compress_video(model, video_path: str, batch: int, device: torch.device):
    frames = []
    meta = {}
    buf = []

    def flush():
        if not buf:
            return []
        # stack to (B, 3, H, W) uint8
        arr = np.stack(buf, axis=0)
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)  # (B, 3, H, W) uint8
        results = []
        out = model.compress(x)
        # compressai's compress returns lists with one entry per batch item
        y_strings_batch = out["strings"][0]
        z_strings_batch = out["strings"][1]
        for i in range(x.shape[0]):
            results.append({
                "strings": [[y_strings_batch[i]], [z_strings_batch[i]]],
                "shape": out["shape"],
                "pad": out["pad"],
                "orig_hw": out["orig_hw"],
            })
        buf.clear()
        return results

    for kind, item in iter_frames(video_path):
        if kind == "meta":
            meta = item
            continue
        buf.append(item)
        if len(buf) >= batch:
            frames.extend(flush())
    frames.extend(flush())

    return {
        "frames": frames,
        "frame_count": len(frames),
        **meta,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")

    out_dir = Path(args.out_dir)
    (out_dir / "streams").mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}")
    model, arch = load_compressor(args.checkpoint, device)

    # Save model in compact form
    sd = model.state_dict()
    if args.save_dtype == "fp16":
        sd = {k: v.half() if v.dtype == torch.float32 else v for k, v in sd.items()}
    torch.save(sd, out_dir / "model.pt")

    (out_dir / "meta.json").write_text(json.dumps({
        "arch": arch,
        "save_dtype": args.save_dtype,
    }, indent=2))

    names = Path(args.video_names).read_text().splitlines()
    names = [n.strip() for n in names if n.strip()]
    print(f"Compressing {len(names)} videos...")

    total_bytes = 0
    for name in names:
        in_path = Path(args.input_dir) / name
        out_path = out_dir / "streams" / (Path(name).stem + ".bin")
        print(f"  {name} ...", end=" ", flush=True)
        payload = compress_video(model, str(in_path), args.batch, device)
        with open(out_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        sz = out_path.stat().st_size
        total_bytes += sz
        print(f"{payload['frame_count']} frames -> {sz/1024:.1f} KB")

    model_size = (out_dir / "model.pt").stat().st_size
    print(f"\nDone. Streams total: {total_bytes/1024/1024:.2f} MB,"
          f"  model.pt: {model_size/1024/1024:.2f} MB,"
          f"  combined: {(total_bytes+model_size)/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()