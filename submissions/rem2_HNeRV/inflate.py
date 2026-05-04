#!/usr/bin/env python
import io
import lzma
import struct
import sys
from collections import OrderedDict
from pathlib import Path

import brotli
import constriction
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CAMERA_SIZE = (1164, 874)
LATENT_DIM = 28
BASE_CHANNELS = 36
EVAL_SIZE = (384, 512)


def _ac_decode(coded, count, hist_u16):
    probs = hist_u16.astype(np.float64)
    probs = np.maximum(probs, 1e-10)
    probs /= probs.sum()
    cat = constriction.stream.model.Categorical(probs, perfect=False)
    dec = constriction.stream.queue.RangeDecoder(np.frombuffer(coded, dtype=np.uint32))
    out = np.zeros(count, dtype=np.int32)
    for i in range(count):
        out[i] = dec.decode(cat)
    return out


def _decompress_hist(comp_id, blob):
    if comp_id == 0:
        return lzma.decompress(blob)
    if comp_id == 2:
        return brotli.decompress(blob)
    if comp_id == 1:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(blob)
    raise RuntimeError(f"unknown histogram codec id {comp_id}")


def _decode_decoder(blob):
    h = io.BytesIO(blob)
    br_len, hist_len, meta_len, lengths_len, comp_id = struct.unpack("<IIIIB", h.read(17))
    br_bytes = h.read(br_len)
    hist_bytes = h.read(hist_len)
    meta_bytes = h.read(meta_len)
    lengths_bytes = h.read(lengths_len)
    coded_bytes = h.read()

    quantized = OrderedDict()
    bh = io.BytesIO(brotli.decompress(br_bytes))
    n_br = struct.unpack("<I", bh.read(4))[0]
    for _ in range(n_br):
        nl = struct.unpack("<I", bh.read(4))[0]
        name = bh.read(nl).decode("utf-8")
        nd = struct.unpack("<I", bh.read(4))[0]
        shape = tuple(struct.unpack("<I", bh.read(4))[0] for _ in range(nd))
        scale = struct.unpack("<f", bh.read(4))[0]
        cnt = int(np.prod(shape))
        q = np.frombuffer(bh.read(cnt), dtype=np.int8).copy()
        quantized[name] = (q, scale, shape)

    if hist_len == 0:
        return quantized

    hist_raw = _decompress_hist(comp_id, hist_bytes)
    meta_raw = brotli.decompress(meta_bytes)
    lengths_raw = brotli.decompress(lengths_bytes)
    lengths = struct.unpack(f"<{len(lengths_raw) // 4}I", lengths_raw)
    hists = np.frombuffer(hist_raw, dtype=np.uint16).reshape(-1, 256)

    mh = io.BytesIO(meta_raw)
    n_ac = struct.unpack("<I", mh.read(4))[0]
    off = 0
    for i in range(n_ac):
        nl = struct.unpack("<I", mh.read(4))[0]
        name = mh.read(nl).decode("utf-8")
        nd = struct.unpack("<I", mh.read(4))[0]
        shape = tuple(struct.unpack("<I", mh.read(4))[0] for _ in range(nd))
        scale = struct.unpack("<f", mh.read(4))[0]
        cnt = struct.unpack("<I", mh.read(4))[0]
        coded = coded_bytes[off:off + lengths[i]]
        off += lengths[i]
        ub = _ac_decode(coded, cnt, hists[i])
        quantized[name] = ((ub - 128).astype(np.int8), scale, shape)

    return quantized


def _dequantize(quantized):
    sd = OrderedDict()
    for name, (q, scale, shape) in quantized.items():
        sd[name] = torch.from_numpy(q.astype(np.float32).reshape(shape)) * scale
    return sd


def _decode_latents(blob):
    h = io.BytesIO(blob)
    n_rows, n_dim = struct.unpack("<II", h.read(8))
    mins = torch.from_numpy(np.frombuffer(h.read(n_dim * 2), dtype=np.float16).copy()).float()
    scales = torch.from_numpy(np.frombuffer(h.read(n_dim * 2), dtype=np.float16).copy()).float()
    q = torch.from_numpy(
        np.frombuffer(h.read(n_rows * n_dim), dtype=np.uint8).reshape(n_rows, n_dim).copy()
    ).float()
    return q * scales.unsqueeze(0) + mins.unsqueeze(0)


class Decoder(nn.Module):
    def __init__(self, latent_dim, base_channels, eval_size):
        super().__init__()
        self.eval_size = eval_size
        self.base_h, self.base_w = 6, 8
        c = base_channels
        self.channels = [c, c, c, int(c * 0.75), int(c * 0.58), int(c * 0.5), int(c * 0.5)]
        self.stem = nn.Linear(latent_dim, self.channels[0] * self.base_h * self.base_w)
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(6):
            ic = self.channels[i]
            oc = self.channels[i + 1]
            self.blocks.append(nn.Conv2d(ic, oc * 4, 3, padding=1))
            self.skips.append(nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity())
        self.ps = nn.PixelShuffle(2)
        f = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(f, f // 2, 3, padding=2, dilation=2),
            nn.Conv2d(f // 2, f, 3, padding=1),
        )
        self.rgb_0 = nn.Conv2d(f, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(f, 3, 3, padding=1)

    def forward(self, z):
        b = z.shape[0]
        x = self.stem(z).view(b, self.channels[0], self.base_h, self.base_w)
        x = torch.sin(x)
        for blk, sk in zip(self.blocks, self.skips):
            i = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            i = sk(i)
            x = self.ps(blk(x))
            x = torch.sin(x + i)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: inflate.py <archive_dir> <output_dir> <video_names_file>")
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    video_names_file = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = _dequantize(_decode_decoder((archive_dir / "decoder.bin").read_bytes()))
    latents = _decode_latents((archive_dir / "latents.bin").read_bytes()).to(device)

    decoder = Decoder(LATENT_DIM, BASE_CHANNELS, EVAL_SIZE).to(device)
    decoder.load_state_dict(state_dict)
    decoder.eval()

    width, height = CAMERA_SIZE
    names = [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]
    for name in names:
        out_path = output_dir / f"{Path(name).stem}.raw"
        with torch.inference_mode(), out_path.open("wb") as handle:
            for start in range(0, latents.shape[0], 16):
                end = min(start + 16, latents.shape[0])
                pairs = decoder(latents[start:end])
                flat = pairs.view((end - start) * 2, 3, pairs.shape[-2], pairs.shape[-1])
                up = F.interpolate(flat, size=(height, width), mode="bicubic", align_corners=False)
                nhwc = up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8)
                handle.write(nhwc.cpu().numpy().tobytes())


if __name__ == "__main__":
    main()
