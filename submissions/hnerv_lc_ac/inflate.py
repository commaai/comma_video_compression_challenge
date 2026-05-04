#!/usr/bin/env python
import struct
import sys
from pathlib import Path

import brotli
import constriction
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CAMERA_H, CAMERA_W = 874, 1164
N_PAIRS = 600
LATENT_DIM = 28
BASE_CHANNELS = 36
EVAL_SIZE = (384, 512)
DELTA_SCALE = 0.01

SCHEMA = [
    ("stem.weight", (1728, 28)),
    ("stem.bias", (1728,)),
    ("blocks.0.weight", (144, 36, 3, 3)),
    ("blocks.0.bias", (144,)),
    ("blocks.1.weight", (144, 36, 3, 3)),
    ("blocks.1.bias", (144,)),
    ("blocks.2.weight", (108, 36, 3, 3)),
    ("blocks.2.bias", (108,)),
    ("blocks.3.weight", (80, 27, 3, 3)),
    ("blocks.3.bias", (80,)),
    ("blocks.4.weight", (72, 20, 3, 3)),
    ("blocks.4.bias", (72,)),
    ("blocks.5.weight", (72, 18, 3, 3)),
    ("blocks.5.bias", (72,)),
    ("skips.2.weight", (27, 36, 1, 1)),
    ("skips.2.bias", (27,)),
    ("skips.3.weight", (20, 27, 1, 1)),
    ("skips.3.bias", (20,)),
    ("skips.4.weight", (18, 20, 1, 1)),
    ("skips.4.bias", (18,)),
    ("refine.0.weight", (9, 18, 3, 3)),
    ("refine.0.bias", (9,)),
    ("refine.1.weight", (18, 9, 3, 3)),
    ("refine.1.bias", (18,)),
    ("rgb_0.weight", (3, 18, 3, 3)),
    ("rgb_0.bias", (3,)),
    ("rgb_1.weight", (3, 18, 3, 3)),
    ("rgb_1.bias", (3,)),
]
AC_INDICES = [0, 2, 4, 6, 8, 10, 12, 21]
AC_SYMBOL_COUNTS = [int(np.prod(SCHEMA[i][1])) for i in AC_INDICES]
BR_LEN = 7097
HIST_LEN = 895
MERGED_AC_LEN = 153856
LO_LEN = 15537
HI_HIST_LEN = 15

N_TENSORS = len(SCHEMA)
SCA_LEN = N_TENSORS * 2
LATENT_META_LEN = LATENT_DIM * 4
HI_LEN = N_PAIRS * LATENT_DIM


class HNeRVDecoder(nn.Module):
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
            ic, oc = self.channels[i], self.channels[i + 1]
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


def make_categorical(weights):
    p = weights.astype(np.float64)
    p = np.maximum(p, 1e-10)
    p /= p.sum()
    return constriction.stream.model.Categorical(p, perfect=False)


def parse_archive(blob):
    o = 0
    sca = blob[o:o + SCA_LEN]; o += SCA_LEN
    br = blob[o:o + BR_LEN]; o += BR_LEN
    hists_b = blob[o:o + HIST_LEN]; o += HIST_LEN
    merged_ac = blob[o:o + MERGED_AC_LEN]; o += MERGED_AC_LEN
    mins_scales = blob[o:o + LATENT_META_LEN]; o += LATENT_META_LEN
    lo_b = blob[o:o + LO_LEN]; o += LO_LEN
    hi_hist_b = blob[o:o + HI_HIST_LEN]; o += HI_HIST_LEN
    wrp_b = blob[o:]
    return sca, br, hists_b, merged_ac, mins_scales, lo_b, hi_hist_b, wrp_b


def decode_merged_ac(merged_ac, hists, hi_hist):
    dec = constriction.stream.queue.RangeDecoder(np.frombuffer(merged_ac, dtype=np.uint32))
    weight_arrays = []
    for k, count in enumerate(AC_SYMBOL_COUNTS):
        cat = make_categorical(hists[k])
        out = np.zeros(count, dtype=np.int32)
        for i in range(count):
            out[i] = dec.decode(cat)
        weight_arrays.append(out)
    hi_cat = make_categorical(hi_hist)
    hi_out = np.zeros(HI_LEN, dtype=np.int32)
    for i in range(HI_LEN):
        hi_out[i] = dec.decode(hi_cat)
    return weight_arrays, hi_out


def build_state_dict(br_b, hists_b, merged_ac, sca, hi_hist):
    br_concat = brotli.decompress(br_b)
    hists = np.frombuffer(brotli.decompress(hists_b), dtype=np.uint8).reshape(len(AC_INDICES), 256)
    scales = np.frombuffer(sca, dtype=np.float16)
    weight_arrays, hi_decoded = decode_merged_ac(merged_ac, hists, hi_hist)
    ac_arrays = {}
    for k, idx in enumerate(AC_INDICES):
        shape = SCHEMA[idx][1]
        ac_arrays[idx] = (weight_arrays[k] - 128).astype(np.int8).reshape(shape)
    sd = {}
    br_off = 0
    for idx, (name, shape) in enumerate(SCHEMA):
        if idx in ac_arrays:
            chunk = ac_arrays[idx]
        else:
            n_el = int(np.prod(shape))
            chunk = np.frombuffer(br_concat[br_off:br_off + n_el], dtype=np.int8).reshape(shape)
            br_off += n_el
        sd[name] = torch.from_numpy(chunk.astype(np.float32) * float(scales[idx]))
    return sd, hi_decoded


def decode_latents(mins_scales, lo_b, hi_decoded):
    mins = np.frombuffer(mins_scales[:LATENT_DIM * 2], dtype=np.float16).astype(np.float32)
    scales = np.frombuffer(mins_scales[LATENT_DIM * 2:], dtype=np.float16).astype(np.float32)
    lo = np.frombuffer(brotli.decompress(lo_b), dtype=np.uint8).astype(np.uint16)
    hi = hi_decoded.astype(np.uint16)
    delta_zz = ((hi << 8) | lo).reshape(N_PAIRS, LATENT_DIM)
    delta = np.where(
        delta_zz % 2 == 0,
        delta_zz.astype(np.int32) // 2,
        -(delta_zz.astype(np.int32) // 2) - 1,
    ).astype(np.int16)
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, N_PAIRS):
        q[i] = q[i - 1] + delta[i]
    return torch.from_numpy(q.astype(np.float32) * scales[None, :] + mins[None, :])


def apply_corrections(latents, wrp_brotli):
    if not wrp_brotli:
        return latents
    raw = brotli.decompress(wrp_brotli)
    arr = np.frombuffer(raw, dtype=np.uint8)
    dim28 = arr[:N_PAIRS]
    dqzz = arr[N_PAIRS:N_PAIRS * 2].astype(np.int16)
    delta_q = np.where(dqzz % 2 == 0, dqzz // 2, -(dqzz // 2) - 1).astype(np.int8)
    for p in range(N_PAIRS):
        d = int(dim28[p])
        if d == LATENT_DIM:
            continue
        latents[p, d] = latents[p, d] + float(delta_q[p]) * DELTA_SCALE
    return latents


def inflate(src_bin, dst_raw):
    blob = Path(src_bin).read_bytes()
    sca, br_b, hists_b, merged_ac, ms, lo_b, hi_hist_b, wrp_b = parse_archive(blob)
    hi_hist = np.frombuffer(brotli.decompress(hi_hist_b), dtype=np.uint16)
    state_dict, hi_decoded = build_state_dict(br_b, hists_b, merged_ac, sca, hi_hist)
    latents = decode_latents(ms, lo_b, hi_decoded)
    apply_corrections(latents, wrp_b)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = HNeRVDecoder(LATENT_DIM, BASE_CHANNELS, EVAL_SIZE).to(device)
    decoder.load_state_dict(state_dict)
    decoder.eval()
    latents = latents.to(device)
    eval_h, eval_w = EVAL_SIZE
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, N_PAIRS, 16):
            j = min(i + 16, N_PAIRS); B = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(B * 2, 3, eval_h, eval_w)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            frames = up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            fout.write(frames.tobytes())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
