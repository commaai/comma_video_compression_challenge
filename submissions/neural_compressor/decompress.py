"""
Decompress: archive/ -> reconstructed video frames.

Output format
-------------
We write reconstructed frames back as LOSSLESS HEVC .hevc streams (libx265
with lossless=1) so the comma evaluator's HEVC dataloader can read them
without introducing extra distortion.

The output filename matches the original (just with .hevc extension), so
evaluate.sh can pair them up.

Usage
-----
    python decompress.py --archive_dir ./archive --out_dir ./reconstructed
"""
import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import torch

from compressor import ScaleHyperpriorCompressor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--archive_dir", type=str, required=True,
                   help="Directory containing model.pt, meta.json, streams/")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ext", type=str, default="hevc", choices=["hevc", "raw"])
    return p.parse_args()


def load_compressor(archive_dir: Path, device: torch.device):
    meta = json.loads((archive_dir / "meta.json").read_text())
    arch = meta["arch"]
    model = ScaleHyperpriorCompressor(N=arch["N"], M=arch["M"], N_hyp=arch["N_hyp"]).to(device)
    sd = torch.load(archive_dir / "model.pt", map_location=device)
    # promote fp16 -> fp32 for inference (decompress is a bit numerically sensitive)
    sd = {k: v.float() if v.dtype == torch.float16 else v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    model.update(force=True)
    return model


@torch.no_grad()
def decode_video(model, payload, device):
    """Yields uint8 (H, W, 3) numpy frames."""
    for frame_info in payload["frames"]:
        x_hat = model.decompress(
            strings=frame_info["strings"],
            shape=frame_info["shape"],
            pad=frame_info["pad"],
            orig_hw=frame_info["orig_hw"],
        )  # (1, 3, H, W) uint8
        yield x_hat[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3) uint8


def write_lossless_hevc(frames_iter, out_path: Path, H: int, W: int, fps_num: int, fps_den: int):
    """Pipe raw RGB frames into ffmpeg, encode as lossless x265, write .hevc."""
    fps = f"{fps_num}/{fps_den}" if fps_den else "20/1"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", fps,
        "-i", "-",
        "-c:v", "libx265",
        "-x265-params", "lossless=1:log-level=error",
        "-pix_fmt", "yuv420p",   # comma's loader expects yuv420
        "-f", "hevc",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames_iter:
        assert frame.shape == (H, W, 3) and frame.dtype == np.uint8
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit {rc}")


def write_raw(frames_iter, out_path: Path):
    """Write a flat .raw file: concatenated (H*W*3 bytes per frame)."""
    with open(out_path, "wb") as f:
        for frame in frames_iter:
            f.write(frame.tobytes())


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    archive_dir = Path(args.archive_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {archive_dir}")
    model = load_compressor(archive_dir, device)

    streams_dir = archive_dir / "streams"
    bin_files = sorted(streams_dir.glob("*.bin"))
    print(f"Inflating {len(bin_files)} videos -> {out_dir}")

    for bin_path in bin_files:
        with open(bin_path, "rb") as f:
            payload = pickle.load(f)

        if payload["frame_count"] == 0:
            continue

        # Peek first frame to know H, W
        first = next(iter(decode_video(model, payload, device)))
        H, W, _ = first.shape

        def all_frames():
            yield first
            yield from decode_video(model, payload, device)
            # Note: we re-decode to be safe; alternative is buffering all frames in memory.
            # For long videos, prefer a single-pass version.

        # Single-pass version: decode again from scratch but yield all
        def all_frames_singlepass():
            yield from decode_video(model, payload, device)

        out_name = bin_path.stem + "." + args.ext
        out_path = out_dir / out_name

        if args.ext == "hevc":
            write_lossless_hevc(
                all_frames_singlepass(),
                out_path,
                H=H, W=W,
                fps_num=payload.get("fps_num", 20),
                fps_den=payload.get("fps_den", 1),
            )
        else:  # raw
            write_raw(all_frames_singlepass(), out_path)

        print(f"  {bin_path.name} -> {out_name}  ({out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()