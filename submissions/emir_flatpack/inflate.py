#!/usr/bin/env python
"""emir_flatpack: lossless tightening of qpose14 (PR #63).

Same renderer, same masks, same pose vectors, same numeric output.
Differences vs qpose14:
  - model.pt is repacked as a flat binary blob (no pickle / torch.save
    overhead). Schema is hardcoded below.
  - mask + model + pose are concatenated and brotli-compressed as one
    stream so the entropy coder can share its window across them.
  - 1-byte order tag in front of the brotli payload picks the best
    of 6 stream orderings (chosen at encode time).

Credit: qpose14 (PR #63), quantizr (PR #55), unified_brotli (PR #64).
"""
import io
import os
import struct
import sys
import tempfile
from math import prod
from pathlib import Path

import av
import brotli
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# -----------------------------
# Hardcoded schema (matches qpose14 model.pt)
# -----------------------------
QUANTIZED_SCHEMA = [
    ('shared_trunk.embedding', 'fp16', (5, 6), False),
    ('shared_trunk.stem_conv.dw', 'fp4_packed', (8, 1, 3, 3), False),
    ('shared_trunk.stem_conv.pw', 'fp4_packed', (56, 8, 1, 1), True),
    ('shared_trunk.stem_block.conv1.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('shared_trunk.stem_block.conv1.pw', 'fp4_packed', (56, 56, 1, 1), True),
    ('shared_trunk.stem_block.conv2.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('shared_trunk.stem_block.conv2.pw', 'fp4_packed', (56, 56, 1, 1), True),
    ('shared_trunk.down_conv.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('shared_trunk.down_conv.pw', 'fp4_packed', (64, 56, 1, 1), True),
    ('shared_trunk.down_block.conv1.dw', 'fp4_packed', (64, 1, 3, 3), False),
    ('shared_trunk.down_block.conv1.pw', 'fp4_packed', (64, 64, 1, 1), True),
    ('shared_trunk.down_block.conv2.dw', 'fp4_packed', (64, 1, 3, 3), False),
    ('shared_trunk.down_block.conv2.pw', 'fp4_packed', (64, 64, 1, 1), True),
    ('shared_trunk.up.1.dw', 'fp4_packed', (64, 1, 3, 3), False),
    ('shared_trunk.up.1.pw', 'fp4_packed', (56, 64, 1, 1), True),
    ('shared_trunk.fuse.dw', 'fp4_packed', (112, 1, 3, 3), False),
    ('shared_trunk.fuse.pw', 'fp4_packed', (56, 112, 1, 1), True),
    ('shared_trunk.fuse_block.conv1.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('shared_trunk.fuse_block.conv1.pw', 'fp4_packed', (56, 56, 1, 1), True),
    ('shared_trunk.fuse_block.conv2.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('shared_trunk.fuse_block.conv2.pw', 'fp4_packed', (56, 56, 1, 1), True),
    ('frame1_head.block1.conv1.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('frame1_head.block1.conv1.pw', 'fp4_packed', (56, 56, 1, 1), True),
    ('frame1_head.block1.conv2.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('frame1_head.block1.conv2.pw', 'fp4_packed', (56, 56, 1, 1), True),
    ('frame1_head.block2.conv1.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('frame1_head.block2.conv1.pw', 'fp4_packed', (56, 56, 1, 1), True),
    ('frame1_head.block2.conv2.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('frame1_head.block2.conv2.pw', 'fp4_packed', (56, 56, 1, 1), True),
    ('frame1_head.pre.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('frame1_head.pre.pw', 'fp4_packed', (52, 56, 1, 1), True),
    ('frame1_head.head', 'fp16', (3, 52, 1, 1), True),
    ('frame2_head.block1.conv1.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('frame2_head.block1.conv1.pw', 'fp4_packed', (56, 56, 1, 1), True),
    ('frame2_head.block1.conv2.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('frame2_head.block1.conv2.pw', 'fp4_packed', (56, 56, 1, 1), True),
    ('frame2_head.block2.conv1.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('frame2_head.block2.conv1.pw', 'fp4_packed', (56, 56, 1, 1), True),
    ('frame2_head.block2.conv2.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('frame2_head.block2.conv2.pw', 'fp4_packed', (56, 56, 1, 1), True),
    ('frame2_head.pre.dw', 'fp4_packed', (56, 1, 3, 3), False),
    ('frame2_head.pre.pw', 'fp4_packed', (52, 56, 1, 1), True),
    ('frame2_head.head', 'fp16', (3, 52, 1, 1), True),
]

DENSE_SCHEMA = [
    ('shared_trunk.stem_conv.norm.weight', (56,)),
    ('shared_trunk.stem_conv.norm.bias', (56,)),
    ('shared_trunk.stem_block.conv1.norm.weight', (56,)),
    ('shared_trunk.stem_block.conv1.norm.bias', (56,)),
    ('shared_trunk.stem_block.norm2.weight', (56,)),
    ('shared_trunk.stem_block.norm2.bias', (56,)),
    ('shared_trunk.down_conv.norm.weight', (64,)),
    ('shared_trunk.down_conv.norm.bias', (64,)),
    ('shared_trunk.down_block.conv1.norm.weight', (64,)),
    ('shared_trunk.down_block.conv1.norm.bias', (64,)),
    ('shared_trunk.down_block.norm2.weight', (64,)),
    ('shared_trunk.down_block.norm2.bias', (64,)),
    ('shared_trunk.up.1.norm.weight', (56,)),
    ('shared_trunk.up.1.norm.bias', (56,)),
    ('shared_trunk.fuse.norm.weight', (56,)),
    ('shared_trunk.fuse.norm.bias', (56,)),
    ('shared_trunk.fuse_block.conv1.norm.weight', (56,)),
    ('shared_trunk.fuse_block.conv1.norm.bias', (56,)),
    ('shared_trunk.fuse_block.norm2.weight', (56,)),
    ('shared_trunk.fuse_block.norm2.bias', (56,)),
    ('pose_mlp.0.weight', (48, 6)),
    ('pose_mlp.0.bias', (48,)),
    ('pose_mlp.2.weight', (48, 48)),
    ('pose_mlp.2.bias', (48,)),
    ('frame1_head.block1.conv1.norm.weight', (56,)),
    ('frame1_head.block1.conv1.norm.bias', (56,)),
    ('frame1_head.block1.norm2.weight', (56,)),
    ('frame1_head.block1.norm2.bias', (56,)),
    ('frame1_head.block1.film_proj.weight', (112, 48)),
    ('frame1_head.block1.film_proj.bias', (112,)),
    ('frame1_head.block2.conv1.norm.weight', (56,)),
    ('frame1_head.block2.conv1.norm.bias', (56,)),
    ('frame1_head.block2.norm2.weight', (56,)),
    ('frame1_head.block2.norm2.bias', (56,)),
    ('frame1_head.pre.norm.weight', (52,)),
    ('frame1_head.pre.norm.bias', (52,)),
    ('frame2_head.block1.conv1.norm.weight', (56,)),
    ('frame2_head.block1.conv1.norm.bias', (56,)),
    ('frame2_head.block1.norm2.weight', (56,)),
    ('frame2_head.block1.norm2.bias', (56,)),
    ('frame2_head.block2.conv1.norm.weight', (56,)),
    ('frame2_head.block2.conv1.norm.bias', (56,)),
    ('frame2_head.block2.norm2.weight', (56,)),
    ('frame2_head.block2.norm2.bias', (56,)),
    ('frame2_head.pre.norm.weight', (52,)),
    ('frame2_head.pre.norm.bias', (52,)),
]

BLOCK_SIZE = 32

ORDER_INDEX = [
    ("mask", "model", "pose"),
    ("mask", "pose", "model"),
    ("model", "mask", "pose"),
    ("model", "pose", "mask"),
    ("pose", "mask", "model"),
    ("pose", "model", "mask"),
]


# -----------------------------
# FP4 Dequantization (verbatim from qpose14)
# -----------------------------
class FP4Codebook:
    pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

    @staticmethod
    def dequantize_from_nibbles(nibbles: torch.Tensor, scales: torch.Tensor, orig_shape):
        flat_n = int(torch.tensor(orig_shape).prod().item())
        block_size = nibbles.numel() // scales.numel()

        nibbles = nibbles.view(-1, block_size)
        signs = (nibbles >> 3).to(torch.int64)
        mag_idx = (nibbles & 0x7).to(torch.int64)

        levels = FP4Codebook.pos_levels.to(scales.device, torch.float32)
        q = levels[mag_idx]
        q = torch.where(signs.bool(), -q, q)
        dq = q * scales[:, None].to(torch.float32)
        return dq.view(-1)[:flat_n].reshape(orig_shape)


def unpack_nibbles(packed: torch.Tensor, count: int) -> torch.Tensor:
    flat = packed.reshape(-1)
    hi = (flat >> 4) & 0x0F
    lo = flat & 0x0F
    out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = hi
    out[1::2] = lo
    return out[:count]


def decode_flat_model(blob: bytes, device: torch.device):
    """Walk QUANTIZED_SCHEMA + DENSE_SCHEMA, slice tensors out of `blob`."""
    state_dict = {}
    off = 0

    for name, kind, weight_shape, has_bias in QUANTIZED_SCHEMA:
        n = prod(weight_shape)

        if kind == "fp4_packed":
            padded = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
            packed_count = padded // 2
            scales_count = padded // BLOCK_SIZE

            packed = torch.frombuffer(bytearray(blob[off:off + packed_count]), dtype=torch.uint8)
            off += packed_count
            scales = torch.frombuffer(bytearray(blob[off:off + scales_count * 2]), dtype=torch.float16)
            off += scales_count * 2

            nibbles = unpack_nibbles(packed.to(device), padded)
            w = FP4Codebook.dequantize_from_nibbles(nibbles, scales.to(device), weight_shape)
            state_dict[f"{name}.weight"] = w.float()
        elif kind == "fp16":
            w = torch.frombuffer(bytearray(blob[off:off + n * 2]), dtype=torch.float16)
            off += n * 2
            state_dict[f"{name}.weight"] = w.reshape(weight_shape).to(device).float()
        else:
            raise RuntimeError(f"unknown kind {kind}")

        if has_bias:
            bias_n = weight_shape[0]
            b = torch.frombuffer(bytearray(blob[off:off + bias_n * 2]), dtype=torch.float16)
            off += bias_n * 2
            state_dict[f"{name}.bias"] = b.to(device).float()

    for name, shape in DENSE_SCHEMA:
        n = prod(shape)
        t = torch.frombuffer(bytearray(blob[off:off + n * 2]), dtype=torch.float16)
        off += n * 2
        state_dict[name] = t.reshape(shape).to(device).float()

    if off != len(blob):
        raise RuntimeError(f"flat-model decode underrun: read {off}, have {len(blob)}")
    return state_dict


# -----------------------------
# Architecture (verbatim from qpose14)
# -----------------------------
class QConv2d(nn.Conv2d):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)


class QEmbedding(nn.Embedding):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)


class SepConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        pad = k // 2
        mid_ch = in_ch * depth_mult
        self.dw = QConv2d(in_ch, mid_ch, k, stride=stride, padding=pad, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid_ch, out_ch, 1, padding=0, bias=True, quantize_weight=quantize_weight)
        self.norm = nn.GroupNorm(2, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class SepConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        pad = k // 2
        mid_ch = in_ch * depth_mult
        self.dw = QConv2d(in_ch, mid_ch, k, stride=stride, padding=pad, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid_ch, out_ch, 1, padding=0, bias=True, quantize_weight=quantize_weight)

    def forward(self, x):
        return self.pw(self.dw(x))


class SepResBlock(nn.Module):
    def __init__(self, ch, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.norm2(self.conv2(self.conv1(x))))


class FiLMSepResBlock(nn.Module):
    def __init__(self, ch, cond_dim, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch)
        self.film_proj = nn.Linear(cond_dim, ch * 2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, cond_emb):
        residual = x
        x = self.norm2(self.conv2(self.conv1(x)))
        film = self.film_proj(cond_emb).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = film.chunk(2, dim=1)
        x = x * (1.0 + gamma) + beta
        return self.act(residual + x)


class SharedMaskDecoder(nn.Module):
    def __init__(self, num_classes=5, emb_dim=6, c1=40, c2=44, depth_mult=4):
        super().__init__()
        self.embedding = QEmbedding(num_classes, emb_dim, quantize_weight=False)
        self.stem_conv = SepConvGNAct(emb_dim + 2, c1, depth_mult=depth_mult)
        self.stem_block = SepResBlock(c1, depth_mult=depth_mult)
        self.down_conv = SepConvGNAct(c1, c2, stride=2, depth_mult=depth_mult)
        self.down_block = SepResBlock(c2, depth_mult=depth_mult)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            SepConvGNAct(c2, c1, depth_mult=depth_mult),
        )
        self.fuse = SepConvGNAct(c1 + c1, c1, depth_mult=depth_mult)
        self.fuse_block = SepResBlock(c1, depth_mult=depth_mult)

    def forward(self, mask2, coords):
        e2 = self.embedding(mask2.long()).permute(0, 3, 1, 2)
        e2_up = F.interpolate(e2, size=coords.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([e2_up, coords], dim=1)
        s = self.stem_block(self.stem_conv(x))
        z = self.down_block(self.down_conv(s))
        z = self.up(z)
        return self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))


class Frame2StaticHead(nn.Module):
    def __init__(self, in_ch, hidden=36, depth_mult=4):
        super().__init__()
        self.block1 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)

    def forward(self, feat):
        x = self.block2(self.block1(feat))
        x = self.pre(x)
        return torch.sigmoid(self.head(x)) * 255.0


class FrameHead(nn.Module):
    def __init__(self, in_ch, cond_dim=32, hidden=36, depth_mult=4):
        super().__init__()
        self.block1 = FiLMSepResBlock(in_ch, cond_dim, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)

    def forward(self, feat, cond_emb):
        x = self.block1(feat, cond_emb)
        x = self.block2(x)
        x = self.pre(x)
        return torch.sigmoid(self.head(x)) * 255.0


class JointFrameGenerator(nn.Module):
    def __init__(self, num_classes=5, pose_dim=6, cond_dim=48, depth_mult=1):
        super().__init__()
        self.shared_trunk = SharedMaskDecoder(num_classes=num_classes, emb_dim=6, c1=56, c2=64, depth_mult=depth_mult)
        self.pose_mlp = nn.Sequential(nn.Linear(pose_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.frame1_head = FrameHead(in_ch=56, cond_dim=cond_dim, hidden=52, depth_mult=depth_mult)
        self.frame2_head = Frame2StaticHead(in_ch=56, hidden=52, depth_mult=depth_mult)

    def forward(self, mask2, pose6):
        b = mask2.shape[0]
        coords = make_coord_grid(b, 384, 512, mask2.device, torch.float32)
        shared_feat = self.shared_trunk(mask2, coords)
        pred_frame2 = self.frame2_head(shared_feat)
        cond_emb = self.pose_mlp(pose6)
        pred_frame1 = self.frame1_head(shared_feat, cond_emb)
        return pred_frame1, pred_frame2


def make_coord_grid(batch, height, width, device, dtype):
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


# -----------------------------
# Mask decoding (verbatim from qpose14)
# -----------------------------
def load_encoded_mask_video(path):
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="gray")
        cls_img = np.round(img / 63.0).astype(np.uint8)
        cls_img = np.clip(cls_img, 0, 4)
        frames.append(cls_img)
    container.close()
    return torch.from_numpy(np.stack(frames)).contiguous()


# -----------------------------
# Main
# -----------------------------
def main():
    if len(sys.argv) < 4:
        print("Usage: python inflate.py <data_dir> <output_dir> <file_list_txt>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = [line.strip() for line in file_list_path.read_text().splitlines() if line.strip()]

    # ---- read + decompress single payload
    plain = brotli.decompress((data_dir / "p").read_bytes())
    order_byte = plain[0]
    n_mask, n_model, n_pose = struct.unpack("<III", plain[1:13])
    body = plain[13:]
    order = ORDER_INDEX[order_byte]
    sizes = {"mask": n_mask, "model": n_model, "pose": n_pose}

    streams = {}
    o = 0
    for k in order:
        sz = sizes[k]
        streams[k] = body[o:o + sz]
        o += sz
    raw_mask  = streams["mask"]
    raw_model = streams["model"]
    raw_pose  = streams["pose"]

    # ---- load model from flat blob
    generator = JointFrameGenerator().to(device)
    generator.load_state_dict(decode_flat_model(raw_model, device), strict=True)
    generator.eval()

    # ---- decode AV1 mask via tempfile (pyav needs a path for .obu)
    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
        tmp.write(raw_mask)
        tmp_path = tmp.name
    mask_frames_all = load_encoded_mask_video(tmp_path)
    os.remove(tmp_path)

    # ---- decode pose: same uint16 format as qpose14
    q = np.frombuffer(raw_pose, dtype=np.uint16).reshape(-1, 6)
    pose_np = np.empty(q.shape, dtype=np.float32)
    pose_np[:, 0] = q[:, 0].astype(np.float32) / 512.0 + 20.0
    pose_np[:, 1:] = q[:, 1:].view(np.int16).astype(np.float32) / 2048.0
    pose_frames_all = torch.from_numpy(pose_np).float()

    out_h, out_w = 874, 1164
    cursor = 0
    batch_size = 4
    pairs_per_file = 600

    with torch.inference_mode():
        for file_name in files:
            base_name = os.path.splitext(file_name)[0]
            raw_out_path = out_dir / f"{base_name}.raw"

            file_masks = mask_frames_all[cursor:cursor + pairs_per_file]
            file_poses = pose_frames_all[cursor:cursor + pairs_per_file]
            cursor += pairs_per_file

            with open(raw_out_path, "wb") as f_out:
                pbar = tqdm(range(0, file_masks.shape[0], batch_size), desc=f"Decoding {file_name}")
                for i in pbar:
                    in_mask2 = file_masks[i:i + batch_size].to(device).long()
                    in_pose6 = file_poses[i:i + batch_size].to(device).float()
                    fake1, fake2 = generator(in_mask2, in_pose6)
                    fake1_up = F.interpolate(fake1, size=(out_h, out_w), mode="bilinear", align_corners=False)
                    fake2_up = F.interpolate(fake2, size=(out_h, out_w), mode="bilinear", align_corners=False)
                    batch_comp = torch.stack([fake1_up, fake2_up], dim=1)
                    batch_comp = einops.rearrange(batch_comp, "b t c h w -> (b t) h w c")
                    out_bytes = batch_comp.clamp(0, 255).round().to(torch.uint8)
                    f_out.write(out_bytes.cpu().numpy().tobytes())


if __name__ == "__main__":
    main()
