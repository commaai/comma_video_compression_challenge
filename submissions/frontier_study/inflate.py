#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Inflate our archive payload into raw RGB frames (N, 874, 1164, 3) uint8.

Self-contained at eval time: needs only torch + brotli + src/{model,codec}.py.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from codec import parse_payload          # noqa: E402
from model import HNeRVDecoder           # noqa: E402

CAMERA_H, CAMERA_W = 874, 1164


def inflate(src_path: str, dst_raw: str) -> int:
    decoder_sd, latents, meta = parse_payload(Path(src_path).read_bytes())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dec = HNeRVDecoder(latent_dim=meta["latent_dim"], base_channels=meta["base_channels"],
                       eval_size=tuple(meta["eval_size"])).to(device)
    dec.load_state_dict(decoder_sd)
    dec.eval()
    latents = latents.to(device)
    eh, ew = meta["eval_size"]
    n_pairs = latents.shape[0]
    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            b = j - i
            decoded = dec(latents[i:j])                      # (b,2,3,eh,ew)
            flat = decoded.reshape(b * 2, 3, eh, ew)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            up = up.clamp(0, 255).round().to(torch.uint8)
            frames = up.reshape(b * 2, 3, CAMERA_H, CAMERA_W).permute(0, 2, 3, 1).cpu().numpy()
            fout.write(frames.tobytes())
            n += b * 2
    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <payload.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
