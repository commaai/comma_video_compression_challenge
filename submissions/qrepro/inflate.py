#!/usr/bin/env python
import io
import bz2
import lzma
import os
import struct
import sys
import tempfile
from pathlib import Path

import av
import brotli
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

GEO_MAGIC = b"QGEO1\0"
PACK_MAGIC = b"QPK1\0"
RGB_CONTROL_MAGIC = b"QRGB1\0"
RGB_BAND_CONTROL_MAGIC = b"QRGB2\0"
RGB_SPARSE_CONTROL_MAGIC = b"QRGB3\0"
RGB_HBAND_CONTROL_MAGIC = b"QRGB4\0"
RGB_HBAND_PLANE_ORDER = (5, 2, 3, 0, 11, 16, 22, 18, 6, 19, 13, 20, 7, 10, 21, 12, 1, 9, 23, 17, 24, 4, 25, 8, 28, 15, 14, 27, 26, 29)
RGB_QUAD_CONTROL_MAGIC = b"QRGB5\0"
RGB_QUAD_PLANE_ORDER = (5, 2, 3, 0, 22, 11, 18, 19, 16, 6, 20, 13, 23, 24, 7, 10, 21, 25, 12, 1, 28, 9, 17, 4, 27, 8, 15, 32, 26, 30, 14, 29, 31, 35, 33, 34)
RGB_VDETAIL_CONTROL_MAGIC = b"QRGB6\0"
RGB_VDETAIL_PLANE_ORDER = (5, 2, 3, 0, 22, 11, 18, 40, 19, 16, 6, 20, 13, 37, 23, 24, 7, 10, 21, 25, 12, 1, 36, 28, 9, 17, 4, 46, 43, 27, 8, 39, 41, 15, 32, 42, 26, 30, 14, 38, 29, 31, 35, 44, 33, 47, 45, 34)
RGB_COMPACT_CONTROL_BYTES = 5_058
RGB_COMPACT_PLANE_ORDER = (34, 33, 29, 45, 47, 44, 35, 31, 26, 14, 38, 27, 30, 15, 42, 25, 32, 8, 43, 28, 4, 17, 46, 24, 1, 41, 21, 12, 36, 39, 13, 9, 10, 7, 16, 20, 19, 37, 6, 3, 18, 23, 11, 22, 40, 5, 0, 2)
RGB_HDETAIL_CONTROL_MAGIC = b"QRGB7\0"
RGB_HDETAIL_PLANE_ORDER = RGB_VDETAIL_PLANE_ORDER + tuple(range(48, 60))
COMPACT_MASK_BODY_BYTES = 152_431
COMPACT_MODEL_BODY_BYTES = 56_385
SEM_M11_BR_MAGIC = b"SM11BR\0"
SEM_M5_BR_MAGIC = b"SM5BR\0"
SEM_M5_SHIFT_BR_MAGIC = b"SM5SBR\0"
SEM_M5_SHIFT_BIG_BR_MAGIC = b"SM5S7BR\0"
SEM_M5_SHIFT_BIG3_BR_MAGIC = b"SM5S8BR\0"
SEM_M5_SHIFT_BIG5_BR_MAGIC = b"SM5SABR\0"
SEM_TOPBAND_BR_MAGIC = b"STBM1BR\0"
FLAT_MODEL_BR_MAGIC = b"QFBR\0"
PAYLOAD_ONLY_MODEL_BR_MAGIC = b"QFPL\0"
QROW_MODEL_BR_MAGIC = b"QFQ1\0"
QROW_GROUPED_MODEL_BR_MAGIC = b"QFQ2\0"
QROW_GROUPED3_MODEL_BR_MAGIC = b"QFQ3\0"
QROW_GROUPED4_MODEL_BR_MAGIC = b"QFQ4\0"
NET_H, NET_W = 384, 512
PAIRS_PER_FILE = 600


# -----------------------------
# FP4 Dequantization Tools
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

def get_decoded_state_dict(payload_data, device: torch.device):
    data = torch.load(io.BytesIO(payload_data), map_location=device)
    state_dict = {}

    for name, rec in data["quantized"].items():
        if rec["weight_kind"] == "fp4_packed":
            padded_count = rec["packed_weight"].numel() * 2
            nibbles = unpack_nibbles(rec["packed_weight"].to(device), padded_count)
            w = FP4Codebook.dequantize_from_nibbles(
                nibbles, rec["scales_fp16"].to(device), rec["weight_shape"]
            )
        else:
            w = rec["weight_fp16"].to(device).float()

        state_dict[f"{name}.weight"] = w.float()
        if rec.get("bias_fp16") is not None:
            state_dict[f"{name}.bias"] = rec["bias_fp16"].to(device).float()

    for name, tensor in data["dense_fp16"].items():
        state_dict[name] = tensor.to(device).float() if torch.is_floating_point(tensor) else tensor.to(device)

    return state_dict

# -----------------------------
# Architecture (Inference Only)
# -----------------------------

class QConv2d(nn.Conv2d):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)

class QEmbedding(nn.Embedding):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)

class SepConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, depth_mult: int = 4, quantize_weight: bool = True):
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
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, depth_mult: int = 4, quantize_weight: bool = True):
        super().__init__()
        pad = k // 2
        mid_ch = in_ch * depth_mult

        self.dw = QConv2d(in_ch, mid_ch, k, stride=stride, padding=pad, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid_ch, out_ch, 1, padding=0, bias=True, quantize_weight=quantize_weight)

    def forward(self, x):
        return self.pw(self.dw(x))

class SepResBlock(nn.Module):
    def __init__(self, ch: int, depth_mult: int = 4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.norm2(self.conv2(self.conv1(x))))

class FiLMSepResBlock(nn.Module):
    def __init__(self, ch: int, cond_dim: int, depth_mult: int = 4, quantize_weight=True):
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

    def forward(self, mask2: torch.Tensor, coords: torch.Tensor):
        if torch.is_floating_point(mask2):
            m = mask2.clamp(0.0, 4.0)
            lo = torch.floor(m).long()
            hi = torch.clamp(lo + 1, max=4)
            alpha = (m - lo.float()).unsqueeze(-1)
            e2 = (1.0 - alpha) * F.embedding(lo, self.embedding.weight) + alpha * F.embedding(hi, self.embedding.weight)
        else:
            e2 = self.embedding(mask2.long())
        e2 = e2.permute(0, 3, 1, 2)
        e2_up = F.interpolate(e2, size=coords.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([e2_up, coords], dim=1)
        s = self.stem_block(self.stem_conv(x))
        z = self.down_block(self.down_conv(s))
        z = self.up(z)
        f = self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))
        return f

class Frame2StaticHead(nn.Module):
    def __init__(self, in_ch: int, cond_dim: int = 48, hidden: int = 36, depth_mult: int = 4):
        super().__init__()
        self.block1 = FiLMSepResBlock(in_ch, cond_dim, depth_mult=depth_mult)
        nn.init.zeros_(self.block1.film_proj.weight)
        nn.init.zeros_(self.block1.film_proj.bias)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)

    def forward(self, feat: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        x = self.block1(feat, cond_emb)
        x = self.block2(x)
        x = self.pre(x)
        return torch.sigmoid(self.head(x)) * 255.0

class FrameHead(nn.Module):
    def __init__(self, in_ch: int, cond_dim: int = 32, hidden: int = 36, depth_mult: int = 4):
        super().__init__()
        self.block1 = FiLMSepResBlock(in_ch, cond_dim, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)

    def forward(self, feat: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        x = self.block1(feat, cond_emb)
        x = self.block2(x)
        x = self.pre(x)
        return torch.sigmoid(self.head(x)) * 255.0

class JointFrameGenerator(nn.Module):
    def __init__(self, num_classes=5, pose_dim=6, cond_dim=48, depth_mult=1):
        super().__init__()
        self.shared_trunk = SharedMaskDecoder(
            num_classes=num_classes, emb_dim=6, c1=56, c2=64, depth_mult=depth_mult)

        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        self.frame1_head = FrameHead(
            in_ch=56, cond_dim=cond_dim, hidden=52, depth_mult=depth_mult)

        self.frame2_head = Frame2StaticHead(
            in_ch=56, cond_dim=cond_dim, hidden=52, depth_mult=depth_mult)

    def forward(self, mask2: torch.Tensor, pose6: torch.Tensor):
        b = mask2.shape[0]
        coords = make_coord_grid(b, 384, 512, mask2.device, torch.float32)

        cond_emb = self.pose_mlp(pose6)
        shared_feat = self.shared_trunk(mask2, coords)
        pred_frame2 = self.frame2_head(shared_feat, cond_emb)
        pred_frame1 = self.frame1_head(shared_feat, cond_emb)

        return pred_frame1, pred_frame2

def make_coord_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


# -----------------------------
# Inference Helpers & Main
# -----------------------------
def load_encoded_mask_video(path: str, soft: bool = False) -> torch.Tensor:
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="gray")
        scaled = img.astype(np.float32) / 63.0
        if soft:
            cls_img = np.clip(scaled, 0.0, 4.0).astype(np.float32)
        else:
            cls_img = np.clip(np.round(scaled), 0, 4).astype(np.uint8)
        frames.append(cls_img)
    container.close()
    return torch.from_numpy(np.stack(frames)).contiguous()

def decode_geo_topbot(payload: bytes) -> np.ndarray:
    encoded = np.frombuffer(bz2.decompress(payload), dtype=np.uint16).reshape(PAIRS_PER_FILE, 2, NET_W)
    dx = ((encoded >> 1).astype(np.int32) ^ -((encoded & 1).astype(np.int32)))
    bounds = np.cumsum(dx, axis=2).astype(np.int16)
    tops = bounds[:, 0].clip(0, NET_H)
    bots = bounds[:, 1].clip(0, NET_H)

    masks = np.zeros((PAIRS_PER_FILE, NET_H, NET_W), dtype=np.uint8)
    for t in range(PAIRS_PER_FILE):
        for x in range(NET_W):
            top = int(tops[t, x])
            bot = int(bots[t, x])
            masks[t, :top, x] = 2
            masks[t, top:bot, x] = 0
            masks[t, bot:, x] = 4
    return masks

def render_geo_components(masks: np.ndarray, payload: bytes, cls: int) -> None:
    raw = lzma.decompress(payload)
    offset = 0
    for t in range(PAIRS_PER_FILE):
        (num_components,) = struct.unpack_from("<H", raw, offset)
        offset += 2
        for _ in range(num_components):
            (num_samples,) = struct.unpack_from("<H", raw, offset)
            offset += 2
            samples = []
            y = left = right = 0
            for _ in range(num_samples):
                dy, dl, dr = struct.unpack_from("<hhh", raw, offset)
                offset += 6
                y += dy
                left += dl
                right += dr
                samples.append((y, left, right))
            if not samples:
                continue
            for i in range(len(samples) - 1):
                y0, l0, r0 = samples[i]
                y1, l1, r1 = samples[i + 1]
                dy = max(1, y1 - y0)
                for yy in range(y0, y1):
                    a = (yy - y0) / dy
                    lft = round(l0 + (l1 - l0) * a)
                    rgt = round(r0 + (r1 - r0) * a)
                    masks[t, yy, max(0, lft):min(NET_W, rgt)] = cls
            yy, lft, rgt = samples[-1]
            masks[t, yy, max(0, lft):min(NET_W, rgt)] = cls

def decode_geo_masks(top_payload: bytes, vehicle_payload: bytes, lane_payload: bytes) -> torch.Tensor:
    masks = decode_geo_topbot(top_payload)
    render_geo_components(masks, vehicle_payload, 3)
    render_geo_components(masks, lane_payload, 1)
    return torch.from_numpy(masks).contiguous()

def split_geo_payload(payload: bytes):
    offset = len(GEO_MAGIC)
    model_len, pose_len, top_len, vehicle_len, lane_len = struct.unpack_from("<IIIII", payload, offset)
    offset += 20
    model_data = payload[offset:offset + model_len]
    offset += model_len
    pose_data = payload[offset:offset + pose_len]
    offset += pose_len
    top_data = payload[offset:offset + top_len]
    offset += top_len
    vehicle_data = payload[offset:offset + vehicle_len]
    offset += vehicle_len
    lane_data = payload[offset:offset + lane_len]
    return model_data, pose_data, top_data, vehicle_data, lane_data

def split_qpack_payload(payload: bytes):
    offset = len(PACK_MAGIC)
    mask_len, model_len, pose_len = struct.unpack_from("<III", payload, offset)
    offset += 12
    mask_data = payload[offset:offset + mask_len]
    offset += mask_len
    model_data = payload[offset:offset + model_len]
    offset += model_len
    pose_data = payload[offset:offset + pose_len]
    if len(mask_data) != mask_len or len(model_data) != model_len or len(pose_data) != pose_len:
        raise ValueError("short QPK1 payload")
    return mask_data, model_data, pose_data

def split_compact_payload(payload: bytes):
    mask_end = COMPACT_MASK_BODY_BYTES
    model_end = mask_end + COMPACT_MODEL_BODY_BYTES
    if len(payload) <= model_end:
        raise ValueError("short compact payload")
    mask_data = SEM_TOPBAND_BR_MAGIC + payload[:mask_end]
    model_data = QROW_GROUPED4_MODEL_BR_MAGIC + payload[mask_end:model_end]
    pose_data = payload[model_end:]
    return mask_data, model_data, pose_data

def _decode_qpose_raw(q: np.ndarray) -> torch.Tensor:
    poses = np.empty(q.shape, dtype=np.float32)
    poses[:, 0] = q[:, 0].astype(np.float32) / 512.0 + 20.0
    poses[:, 1:] = q[:, 1:].view(np.int16).astype(np.float32) / 2048.0
    return torch.from_numpy(poses).float()


def decode_control_payload(payload: bytes) -> tuple[torch.Tensor, torch.Tensor | None]:
    if payload.startswith(RGB_HDETAIL_CONTROL_MAGIC):
        offset = len(RGB_HDETAIL_CONTROL_MAGIC)
        (control_len,) = struct.unpack_from("<H", payload, offset)
        offset += 2
        poses, biases = decode_control_payload(payload[offset:offset + control_len])
        offset += control_len
        raw = brotli.decompress(payload[offset:])
        (n_nonzero,) = struct.unpack_from("<H", raw, 0)
        gaps = np.frombuffer(raw, dtype=np.uint8, count=n_nonzero, offset=2).astype(np.int32)
        values = np.frombuffer(raw, dtype=np.int8, count=n_nonzero, offset=2 + n_nonzero)
        flat = np.zeros(36_000, dtype=np.int8)
        positions = np.cumsum(gaps + 1, dtype=np.int32) - 1
        flat[positions] = values
        ordered_planes = flat.reshape(60, PAIRS_PER_FILE)
        planes = np.empty_like(ordered_planes)
        for src, dst in enumerate(RGB_HDETAIL_PLANE_ORDER):
            planes[dst] = ordered_planes[src]
        rgb_residual = planes[:6].reshape(2, 3, PAIRS_PER_FILE).transpose(2, 0, 1).astype(np.float32)
        band_residual = planes[6:18].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
        hband_residual = planes[18:30].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
        quad_residual = planes[30:36].reshape(2, 3, PAIRS_PER_FILE).transpose(2, 0, 1).astype(np.float32)
        vdetail_residual = planes[36:48].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
        hdetail_residual = planes[48:].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
        if biases is None:
            base_bias = np.zeros((rgb_residual.shape[0], 2), dtype=np.float32)
        else:
            base_bias = biases.numpy().astype(np.float32, copy=False)[:, :2]
        full_bias = np.concatenate(
            [
                base_bias,
                rgb_residual[:, 0, :],
                rgb_residual[:, 1, :],
                band_residual.reshape(PAIRS_PER_FILE, -1),
                hband_residual.reshape(PAIRS_PER_FILE, -1),
                quad_residual.reshape(PAIRS_PER_FILE, -1),
                vdetail_residual.reshape(PAIRS_PER_FILE, -1),
                hdetail_residual.reshape(PAIRS_PER_FILE, -1),
            ],
            axis=1,
        )
        return poses, torch.from_numpy(full_bias)

    if payload.startswith(RGB_VDETAIL_CONTROL_MAGIC):
        offset = len(RGB_VDETAIL_CONTROL_MAGIC)
        (control_len,) = struct.unpack_from("<H", payload, offset)
        offset += 2
        poses, biases = decode_control_payload(payload[offset:offset + control_len])
        offset += control_len
        raw = brotli.decompress(payload[offset:])
        (n_nonzero,) = struct.unpack_from("<H", raw, 0)
        gaps = np.frombuffer(raw, dtype=np.uint8, count=n_nonzero, offset=2).astype(np.int32)
        values = np.frombuffer(raw, dtype=np.int8, count=n_nonzero, offset=2 + n_nonzero)
        flat = np.zeros(28_800, dtype=np.int8)
        positions = np.cumsum(gaps + 1, dtype=np.int32) - 1
        flat[positions] = values
        ordered_planes = flat.reshape(48, PAIRS_PER_FILE)
        planes = np.empty_like(ordered_planes)
        for src, dst in enumerate(RGB_VDETAIL_PLANE_ORDER):
            planes[dst] = ordered_planes[src]
        rgb_residual = planes[:6].reshape(2, 3, PAIRS_PER_FILE).transpose(2, 0, 1).astype(np.float32)
        band_residual = planes[6:18].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
        hband_residual = planes[18:30].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
        quad_residual = planes[30:36].reshape(2, 3, PAIRS_PER_FILE).transpose(2, 0, 1).astype(np.float32)
        vdetail_residual = planes[36:].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
        if biases is None:
            base_bias = np.zeros((rgb_residual.shape[0], 2), dtype=np.float32)
        else:
            base_bias = biases.numpy().astype(np.float32, copy=False)[:, :2]
        full_bias = np.concatenate(
            [
                base_bias,
                rgb_residual[:, 0, :],
                rgb_residual[:, 1, :],
                band_residual.reshape(PAIRS_PER_FILE, -1),
                hband_residual.reshape(PAIRS_PER_FILE, -1),
                quad_residual.reshape(PAIRS_PER_FILE, -1),
                vdetail_residual.reshape(PAIRS_PER_FILE, -1),
            ],
            axis=1,
        )
        return poses, torch.from_numpy(full_bias)

    if payload.startswith(RGB_QUAD_CONTROL_MAGIC):
        offset = len(RGB_QUAD_CONTROL_MAGIC)
        (control_len,) = struct.unpack_from("<H", payload, offset)
        offset += 2
        poses, biases = decode_control_payload(payload[offset:offset + control_len])
        offset += control_len
        raw = brotli.decompress(payload[offset:])
        (n_nonzero,) = struct.unpack_from("<H", raw, 0)
        gaps = np.frombuffer(raw, dtype=np.uint8, count=n_nonzero, offset=2).astype(np.int32)
        values = np.frombuffer(raw, dtype=np.int8, count=n_nonzero, offset=2 + n_nonzero)
        flat = np.zeros(21_600, dtype=np.int8)
        positions = np.cumsum(gaps + 1, dtype=np.int32) - 1
        flat[positions] = values
        ordered_planes = flat.reshape(36, PAIRS_PER_FILE)
        planes = np.empty_like(ordered_planes)
        for src, dst in enumerate(RGB_QUAD_PLANE_ORDER):
            planes[dst] = ordered_planes[src]
        rgb_residual = planes[:6].reshape(2, 3, PAIRS_PER_FILE).transpose(2, 0, 1).astype(np.float32)
        band_residual = planes[6:18].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
        hband_residual = planes[18:30].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
        quad_residual = planes[30:].reshape(2, 3, PAIRS_PER_FILE).transpose(2, 0, 1).astype(np.float32)
        if biases is None:
            base_bias = np.zeros((rgb_residual.shape[0], 2), dtype=np.float32)
        else:
            base_bias = biases.numpy().astype(np.float32, copy=False)[:, :2]
        full_bias = np.concatenate(
            [
                base_bias,
                rgb_residual[:, 0, :],
                rgb_residual[:, 1, :],
                band_residual.reshape(PAIRS_PER_FILE, -1),
                hband_residual.reshape(PAIRS_PER_FILE, -1),
                quad_residual.reshape(PAIRS_PER_FILE, -1),
            ],
            axis=1,
        )
        return poses, torch.from_numpy(full_bias)

    if payload.startswith(RGB_HBAND_CONTROL_MAGIC):
        offset = len(RGB_HBAND_CONTROL_MAGIC)
        (control_len,) = struct.unpack_from("<H", payload, offset)
        offset += 2
        poses, biases = decode_control_payload(payload[offset:offset + control_len])
        offset += control_len
        raw = brotli.decompress(payload[offset:])
        (n_nonzero,) = struct.unpack_from("<H", raw, 0)
        gaps = np.frombuffer(raw, dtype=np.uint8, count=n_nonzero, offset=2).astype(np.int32)
        values = np.frombuffer(raw, dtype=np.int8, count=n_nonzero, offset=2 + n_nonzero)
        flat = np.zeros(18_000, dtype=np.int8)
        positions = np.cumsum(gaps + 1, dtype=np.int32) - 1
        flat[positions] = values
        ordered_planes = flat.reshape(30, PAIRS_PER_FILE)
        planes = np.empty_like(ordered_planes)
        for src, dst in enumerate(RGB_HBAND_PLANE_ORDER):
            planes[dst] = ordered_planes[src]
        rgb_residual = planes[:6].reshape(2, 3, PAIRS_PER_FILE).transpose(2, 0, 1).astype(np.float32)
        band_residual = planes[6:18].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
        hband_residual = planes[18:].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
        if biases is None:
            base_bias = np.zeros((rgb_residual.shape[0], 2), dtype=np.float32)
        else:
            base_bias = biases.numpy().astype(np.float32, copy=False)[:, :2]
        full_bias = np.concatenate(
            [
                base_bias,
                rgb_residual[:, 0, :],
                rgb_residual[:, 1, :],
                band_residual.reshape(PAIRS_PER_FILE, -1),
                hband_residual.reshape(PAIRS_PER_FILE, -1),
            ],
            axis=1,
        )
        return poses, torch.from_numpy(full_bias)

    if payload.startswith(RGB_SPARSE_CONTROL_MAGIC):
        offset = len(RGB_SPARSE_CONTROL_MAGIC)
        (control_len,) = struct.unpack_from("<H", payload, offset)
        offset += 2
        poses, biases = decode_control_payload(payload[offset:offset + control_len])
        offset += control_len
        raw = brotli.decompress(payload[offset:])
        (n_nonzero,) = struct.unpack_from("<H", raw, 0)
        gaps = np.frombuffer(raw, dtype=np.uint8, count=n_nonzero, offset=2).astype(np.int32)
        values = np.frombuffer(raw, dtype=np.int8, count=n_nonzero, offset=2 + n_nonzero)
        flat = np.zeros(10_800, dtype=np.int8)
        positions = np.cumsum(gaps + 1, dtype=np.int32) - 1
        flat[positions] = values
        rgb_residual = flat[:3_600].reshape(2, 3, PAIRS_PER_FILE).transpose(2, 0, 1).astype(np.float32)
        band_residual = flat[3_600:].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
        if biases is None:
            base_bias = np.zeros((rgb_residual.shape[0], 2), dtype=np.float32)
        else:
            base_bias = biases.numpy().astype(np.float32, copy=False)[:, :2]
        full_bias = np.concatenate([base_bias, rgb_residual[:, 0, :], rgb_residual[:, 1, :], band_residual.reshape(PAIRS_PER_FILE, -1)], axis=1)
        return poses, torch.from_numpy(full_bias)

    if payload.startswith(RGB_BAND_CONTROL_MAGIC):
        offset = len(RGB_BAND_CONTROL_MAGIC)
        control_len, rgb_len = struct.unpack_from("<HH", payload, offset)
        offset += 4
        poses, biases = decode_control_payload(payload[offset:offset + control_len])
        offset += control_len
        rgb_residual = np.frombuffer(brotli.decompress(payload[offset:offset + rgb_len]), dtype=np.int8).reshape(PAIRS_PER_FILE, 2, 3).astype(np.float32)
        offset += rgb_len
        band_residual = np.frombuffer(brotli.decompress(payload[offset:]), dtype=np.int8).reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
        if biases is None:
            base_bias = np.zeros((rgb_residual.shape[0], 2), dtype=np.float32)
        else:
            base_bias = biases.numpy().astype(np.float32, copy=False)[:, :2]
        full_bias = np.concatenate([base_bias, rgb_residual[:, 0, :], rgb_residual[:, 1, :], band_residual.reshape(PAIRS_PER_FILE, -1)], axis=1)
        return poses, torch.from_numpy(full_bias)

    if payload.startswith(RGB_CONTROL_MAGIC):
        offset = len(RGB_CONTROL_MAGIC)
        (control_len,) = struct.unpack_from("<H", payload, offset)
        offset += 2
        poses, biases = decode_control_payload(payload[offset:offset + control_len])
        offset += control_len
        rgb_residual = np.frombuffer(brotli.decompress(payload[offset:]), dtype=np.int8).reshape(PAIRS_PER_FILE, 2, 3).astype(np.float32)
        if biases is None:
            base_bias = np.zeros((rgb_residual.shape[0], 2), dtype=np.float32)
        else:
            base_bias = biases.numpy().astype(np.float32, copy=False)[:, :2]
        full_bias = np.concatenate([base_bias, rgb_residual[:, 0, :], rgb_residual[:, 1, :]], axis=1)
        return poses, torch.from_numpy(full_bias)

    if len(payload) > RGB_COMPACT_CONTROL_BYTES:
        try:
            raw_residual = brotli.decompress(payload[RGB_COMPACT_CONTROL_BYTES:])
        except brotli.error:
            raw_residual = None
        if raw_residual is not None and len(raw_residual) % 2 == 0:
            poses, biases = decode_control_payload(payload[:RGB_COMPACT_CONTROL_BYTES])
            n_nonzero = len(raw_residual) // 2
            gaps = np.frombuffer(raw_residual, dtype=np.uint8, count=n_nonzero, offset=0).astype(np.int32)
            values = np.frombuffer(raw_residual, dtype=np.int8, count=n_nonzero, offset=n_nonzero)
            flat = np.zeros(28_800, dtype=np.int8)
            positions = np.cumsum(gaps + 1, dtype=np.int32) - 1
            flat[positions] = values
            ordered_planes = flat.reshape(48, PAIRS_PER_FILE)
            planes = np.empty_like(ordered_planes)
            for src, dst in enumerate(RGB_COMPACT_PLANE_ORDER):
                planes[dst] = ordered_planes[src]
            rgb_residual = planes[:6].reshape(2, 3, PAIRS_PER_FILE).transpose(2, 0, 1).astype(np.float32)
            band_residual = planes[6:18].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
            hband_residual = planes[18:30].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
            quad_residual = planes[30:36].reshape(2, 3, PAIRS_PER_FILE).transpose(2, 0, 1).astype(np.float32)
            vdetail_residual = planes[36:].reshape(2, 3, 2, PAIRS_PER_FILE).transpose(3, 0, 1, 2).astype(np.float32)
            if biases is None:
                base_bias = np.zeros((rgb_residual.shape[0], 2), dtype=np.float32)
            else:
                base_bias = biases.numpy().astype(np.float32, copy=False)[:, :2]
            full_bias = np.concatenate(
                [
                    base_bias,
                    rgb_residual[:, 0, :],
                    rgb_residual[:, 1, :],
                    band_residual.reshape(PAIRS_PER_FILE, -1),
                    hband_residual.reshape(PAIRS_PER_FILE, -1),
                    quad_residual.reshape(PAIRS_PER_FILE, -1),
                    vdetail_residual.reshape(PAIRS_PER_FILE, -1),
                ],
                axis=1,
            )
            return poses, torch.from_numpy(full_bias)

    raw = brotli.decompress(payload)
    stripped_qbias3_len = PAIRS_PER_FILE * (2 + 5 * 2 + 2)
    if len(raw) == stripped_qbias3_len:
        offset = 0
        n = PAIRS_PER_FILE
        # Compact submissions store QBIAS3 values as byte planes. Plane order is
        # tuned for Brotli: bias1,bias0, then low bytes for q1,q2,q3,q5,q4,q0delta,
        # then high bytes in the same order.
        plane_order = (1, 2, 3, 5, 4, 0)
        biases_i8 = np.empty((n, 2), dtype=np.int8)
        biases_i8[:, 1] = np.frombuffer(raw, dtype=np.int8, count=n, offset=offset)
        offset += n
        biases_i8[:, 0] = np.frombuffer(raw, dtype=np.int8, count=n, offset=offset)
        offset += n
        planes = {dim: np.empty((n, 2), dtype=np.uint8) for dim in plane_order}
        for dim in plane_order:
            planes[dim][:, 0] = np.frombuffer(raw, dtype=np.uint8, count=n, offset=offset)
            offset += n
        for dim in plane_order:
            planes[dim][:, 1] = np.frombuffer(raw, dtype=np.uint8, count=n, offset=offset)
            offset += n
        q0_delta = planes[0].view(np.int16).reshape(n).astype(np.int32)
        q_signed = np.empty((n, 6), dtype=np.int16)
        q0 = np.cumsum(q0_delta, dtype=np.int32).clip(0, 65535).astype(np.uint16)
        q_signed[:, 0] = q0.astype(np.int32).astype(np.int16)
        for dim in range(1, 6):
            q_signed[:, dim] = planes[dim].view(np.int16).reshape(n)
        biases = biases_i8.astype(np.float32)
        return _decode_qpose_raw(q_signed.view(np.uint16)), torch.from_numpy(biases)
    if raw.startswith(b"QBIAS1\0"):
        offset = 7
        n, n_candidates = struct.unpack_from("<HH", raw, offset)
        offset += 4
        q = np.frombuffer(raw, dtype=np.uint16, count=n * 6, offset=offset).reshape(n, 6)
        offset += n * 6 * 2
        candidates = np.frombuffer(raw, dtype=np.int8, count=n_candidates * 2, offset=offset).reshape(n_candidates, 2).copy()
        offset += n_candidates * 2
        selected = np.frombuffer(raw, dtype=np.uint8, count=n, offset=offset).astype(np.int64)
        poses = _decode_qpose_raw(q)
        biases = torch.from_numpy(candidates[selected].astype(np.float32, copy=False))
        return poses, biases
    if raw.startswith(b"QBIAS2\0"):
        offset = 7
        (n,) = struct.unpack_from("<H", raw, offset)
        offset += 2
        q = np.frombuffer(raw, dtype=np.uint16, count=n * 6, offset=offset).reshape(n, 6)
        offset += n * 6 * 2
        biases = np.frombuffer(raw, dtype=np.int8, count=n * 2, offset=offset).reshape(n, 2).astype(np.float32)
        return _decode_qpose_raw(q), torch.from_numpy(biases)
    if raw.startswith(b"QBIAS5\0"):
        offset = 7
        n, n_values = struct.unpack_from("<HB", raw, offset)
        offset += 3
        values = np.frombuffer(raw, dtype=np.int8, count=n_values, offset=offset).astype(np.float32)
        offset += n_values
        q0_delta = np.frombuffer(raw, dtype=np.int16, count=n, offset=offset).astype(np.int32)
        offset += n * 2
        q_signed = np.empty((n, 6), dtype=np.int16)
        q0 = np.cumsum(q0_delta, dtype=np.int32).clip(0, 65535).astype(np.uint16)
        q_signed[:, 0] = q0.astype(np.int32).astype(np.int16)
        q_signed[:, 1:] = np.frombuffer(raw, dtype=np.int16, count=n * 5, offset=offset).reshape(n, 5)
        offset += n * 5 * 2
        packed = np.frombuffer(raw, dtype=np.uint8, count=n, offset=offset)
        codes = np.empty(n * 2, dtype=np.uint8)
        codes[0::2] = (packed >> 4) & 15
        codes[1::2] = packed & 15
        biases = values[codes.reshape(n, 2)]
        return _decode_qpose_raw(q_signed.view(np.uint16)), torch.from_numpy(biases.astype(np.float32, copy=False))
    if raw.startswith(b"QBIAS3\0"):
        offset = 7
        (n,) = struct.unpack_from("<H", raw, offset)
        offset += 2
        q0_delta = np.frombuffer(raw, dtype=np.int16, count=n, offset=offset).astype(np.int32)
        offset += n * 2
        q_signed = np.empty((n, 6), dtype=np.int16)
        q0 = np.cumsum(q0_delta, dtype=np.int32).clip(0, 65535).astype(np.uint16)
        q_signed[:, 0] = q0.astype(np.int32).astype(np.int16)
        q_signed[:, 1:] = np.frombuffer(raw, dtype=np.int16, count=n * 5, offset=offset).reshape(n, 5)
        offset += n * 5 * 2
        biases = np.frombuffer(raw, dtype=np.int8, count=n * 2, offset=offset).reshape(n, 2).astype(np.float32)
        return _decode_qpose_raw(q_signed.view(np.uint16)), torch.from_numpy(biases)
    if raw.startswith(b"QGAIN2\0"):
        offset = 7
        n, n_gains = struct.unpack_from("<HB", raw, offset)
        offset += 3
        gains = np.frombuffer(raw, dtype=np.float16, count=n_gains, offset=offset).astype(np.float32)
        offset += n_gains * 2
        q0_delta = np.frombuffer(raw, dtype=np.int16, count=n, offset=offset).astype(np.int32)
        offset += n * 2
        q_signed = np.empty((n, 6), dtype=np.int16)
        q0 = np.cumsum(q0_delta, dtype=np.int32).clip(0, 65535).astype(np.uint16)
        q_signed[:, 0] = q0.astype(np.int32).astype(np.int16)
        for i in range(1, 6):
            q_signed[:, i] = np.frombuffer(raw, dtype=np.int16, count=n, offset=offset)
            offset += n * 2
        n_bias_values = raw[offset]
        offset += 1
        bias_values = np.frombuffer(raw, dtype=np.int8, count=n_bias_values, offset=offset).astype(np.float32)
        offset += n_bias_values
        bias_codes = np.frombuffer(raw, dtype=np.uint8, count=n, offset=offset)
        offset += n
        bias_nibbles = np.empty(n * 2, dtype=np.uint8)
        bias_nibbles[0::2] = (bias_codes >> 4) & 15
        bias_nibbles[1::2] = bias_codes & 15
        biases = bias_values[bias_nibbles.reshape(n, 2)]
        gain_codes = np.frombuffer(raw, dtype=np.uint8, count=n, offset=offset)
        gain_nibbles = np.empty(n * 2, dtype=np.uint8)
        gain_nibbles[0::2] = (gain_codes >> 4) & 15
        gain_nibbles[1::2] = gain_codes & 15
        frame_gains = gains[gain_nibbles.reshape(n, 2)]
        q = q_signed.view(np.uint16)
        return _decode_qpose_raw(q), torch.from_numpy(np.concatenate([biases, frame_gains], axis=1).astype(np.float32))
    if raw.startswith(b"QCTRL1\0"):
        offset = 7
        n, k = struct.unpack_from("<HH", raw, offset)
        offset += 4
        scales = np.frombuffer(raw, dtype=np.float32, count=6, offset=offset).copy()
        offset += 24
        base_q = np.frombuffer(raw, dtype=np.uint16, count=n * 6, offset=offset).reshape(n, 6)
        offset += n * 6 * 2
        coeff_q = np.frombuffer(raw, dtype=np.int16, count=k * 6, offset=offset).reshape(k, 6).astype(np.float32)

        base = _decode_qpose_raw(base_q).numpy()
        coeff = coeff_q * scales[None, :]

        t = np.arange(n, dtype=np.float32)[:, None]
        freq = np.arange(k, dtype=np.float32)[None, :]
        basis = np.cos(np.pi * (t + 0.5) * freq / float(n)).astype(np.float32)
        if k:
            basis[:, 0] *= 1.0 / np.sqrt(float(n))
        if k > 1:
            basis[:, 1:] *= np.sqrt(2.0 / float(n))
        poses = base + basis @ coeff
        return torch.from_numpy(poses.astype(np.float32, copy=False)), None
    if raw.startswith(b"QLAT16\0"):
        offset = 7
        mins = np.frombuffer(raw, dtype=np.float32, count=6, offset=offset).copy()
        offset += 24
        maxs = np.frombuffer(raw, dtype=np.float32, count=6, offset=offset).copy()
        offset += 24
        q = np.frombuffer(raw, dtype=np.uint16, offset=offset).reshape(-1, 6).astype(np.float32)
        scale = np.maximum(maxs - mins, 1e-8) / 65535.0
        return torch.from_numpy(q * scale[None, :] + mins[None, :]).float(), None
    q = np.frombuffer(raw, dtype=np.uint16).reshape(-1, 6)
    return _decode_qpose_raw(q), None


def decode_poseq_payload(payload: bytes) -> torch.Tensor:
    poses, _biases = decode_control_payload(payload)
    return poses

def load_pose_frames(data_dir: Path) -> torch.Tensor:
    poseq_path = data_dir / "poseq.bin.br"
    short_poseq_path = data_dir / "c"
    q8_path = data_dir / "pose.q8.br"
    q12_path = data_dir / "pose.q12.br"
    legacy_path = data_dir / "pose.npy.br"

    if poseq_path.exists():
        return decode_poseq_payload(poseq_path.read_bytes())

    if short_poseq_path.exists():
        return decode_poseq_payload(short_poseq_path.read_bytes())

    if q8_path.exists():
        raw = brotli.decompress(q8_path.read_bytes())
        magic = b"QPOSE8\0"
        if not raw.startswith(magic):
            raise ValueError("invalid pose.q8.br header")
        offset = len(magic)
        mins = np.frombuffer(raw, dtype=np.float32, count=6, offset=offset).copy()
        offset += 6 * 4
        scales = np.frombuffer(raw, dtype=np.float32, count=6, offset=offset).copy()
        offset += 6 * 4
        q = np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape(-1, 6)
        poses = q.astype(np.float32) * scales[None, :] + mins[None, :]
        return torch.from_numpy(poses).float()

    if q12_path.exists():
        raw = brotli.decompress(q12_path.read_bytes())
        magic = b"QPOSE12\0"
        if not raw.startswith(magic):
            raise ValueError("invalid pose.q12.br header")
        offset = len(magic)
        mins = np.frombuffer(raw, dtype=np.float32, count=6, offset=offset).copy()
        offset += 6 * 4
        scales = np.frombuffer(raw, dtype=np.float32, count=6, offset=offset).copy()
        offset += 6 * 4
        packed = np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape(-1, 3)
        a = packed[:, 0].astype(np.uint16) | ((packed[:, 1].astype(np.uint16) & 0x0F) << 8)
        b = ((packed[:, 1].astype(np.uint16) >> 4) & 0x0F) | (packed[:, 2].astype(np.uint16) << 4)
        q = np.empty(packed.shape[0] * 2, dtype=np.uint16)
        q[0::2] = a
        q[1::2] = b
        q = q.reshape(-1, 6)
        poses = q.astype(np.float32) * scales[None, :] + mins[None, :]
        return torch.from_numpy(poses).float()

    with open(legacy_path, "rb") as f:
        pose_bytes = brotli.decompress(f.read())
    return torch.from_numpy(np.load(io.BytesIO(pose_bytes))).float()

def main():
    if len(sys.argv) < 4:
        print("Usage: python inflate.py <data_dir> <output_dir> <file_list_txt>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    files = [line.strip() for line in file_list_path.read_text().splitlines() if line.strip()]

    model_br = data_dir / "model.pt.br"
    mask_br = data_dir / "mask.obu.br"
    short_model_br = data_dir / "b"
    short_mask_br = data_dir / "a"
 
    packed_payload = data_dir / "p"
    packed_poseq = None
    geo_mask_payloads = None
    if packed_payload.exists():
        payload = packed_payload.read_bytes()
        if payload.startswith(GEO_MAGIC):
            model_data, packed_poseq, top_data, vehicle_data, lane_data = split_geo_payload(payload)
            geo_mask_payloads = (top_data, vehicle_data, lane_data)
            mask_data = None
        elif payload.startswith(PACK_MAGIC):
            mask_data, model_data, packed_poseq = split_qpack_payload(payload)
        elif len(payload) < 260_000:
            mask_data, model_data, packed_poseq = split_compact_payload(payload)
        else:
            mask_data = payload[:219472]
            model_data = payload[219472:219472 + 66841]
            packed_poseq = payload[219472 + 66841:]
    else:
        if not model_br.exists():
            model_br = short_model_br
        if not mask_br.exists():
            mask_br = short_mask_br
        model_data = model_br.read_bytes()
        mask_data = mask_br.read_bytes()

    generator = JointFrameGenerator().to(device)

    # 1. Load Weights
    if model_data.startswith(FLAT_MODEL_BR_MAGIC):
        from flat_fp4_codec import decode_flat

        state_dict = decode_flat(model_data[len(FLAT_MODEL_BR_MAGIC):])
    elif model_data.startswith(PAYLOAD_ONLY_MODEL_BR_MAGIC):
        from flat_fp4_codec import decode_payload_only_for_model

        state_dict = decode_payload_only_for_model(model_data[len(PAYLOAD_ONLY_MODEL_BR_MAGIC):], generator)
    elif model_data.startswith(QROW_MODEL_BR_MAGIC):
        from flat_fp4_codec import decode_qrow_payload_for_model

        state_dict = decode_qrow_payload_for_model(model_data[len(QROW_MODEL_BR_MAGIC):], generator)
    elif model_data.startswith(QROW_GROUPED_MODEL_BR_MAGIC):
        from flat_fp4_codec import decode_qrow_grouped_payload_for_model

        state_dict = decode_qrow_grouped_payload_for_model(model_data[len(QROW_GROUPED_MODEL_BR_MAGIC):], generator)
    elif model_data.startswith(QROW_GROUPED3_MODEL_BR_MAGIC):
        from flat_fp4_codec import decode_qrow_grouped3_payload_for_model

        state_dict = decode_qrow_grouped3_payload_for_model(model_data[len(QROW_GROUPED3_MODEL_BR_MAGIC):], generator)
    elif model_data.startswith(QROW_GROUPED4_MODEL_BR_MAGIC):
        from flat_fp4_codec import decode_qrow_grouped4_payload_for_model

        state_dict = decode_qrow_grouped4_payload_for_model(model_data[len(QROW_GROUPED4_MODEL_BR_MAGIC):], generator)
    else:
        weights_data = brotli.decompress(model_data)
        state_dict = get_decoded_state_dict(weights_data, device)

    generator.load_state_dict(state_dict, strict=False)
    generator.eval()

    # 2. Load masks.
    if geo_mask_payloads is not None:
        mask_frames_all = decode_geo_masks(*geo_mask_payloads)
    elif mask_data.startswith(SEM_M11_BR_MAGIC):
        from seg_sparse_m11_codec import decode_seg_split_m11

        with tempfile.NamedTemporaryFile(suffix=".m11", delete=False) as tmp_mask:
            tmp_mask.write(brotli.decompress(mask_data[len(SEM_M11_BR_MAGIC):]))
            tmp_mask_path = tmp_mask.name
        try:
            mask_frames_all = torch.from_numpy(decode_seg_split_m11(tmp_mask_path)).contiguous()
        finally:
            os.remove(tmp_mask_path)
    elif (
        mask_data.startswith(SEM_M5_BR_MAGIC)
        or mask_data.startswith(SEM_M5_SHIFT_BR_MAGIC)
        or mask_data.startswith(SEM_M5_SHIFT_BIG_BR_MAGIC)
        or mask_data.startswith(SEM_M5_SHIFT_BIG3_BR_MAGIC)
        or mask_data.startswith(SEM_M5_SHIFT_BIG5_BR_MAGIC)
    ):
        from seg_sparse_m5_codec import decode_seg_split_m5

        if mask_data.startswith(SEM_M5_SHIFT_BIG5_BR_MAGIC):
            magic_len = len(SEM_M5_SHIFT_BIG5_BR_MAGIC)
        elif mask_data.startswith(SEM_M5_SHIFT_BIG3_BR_MAGIC):
            magic_len = len(SEM_M5_SHIFT_BIG3_BR_MAGIC)
        elif mask_data.startswith(SEM_M5_SHIFT_BIG_BR_MAGIC):
            magic_len = len(SEM_M5_SHIFT_BIG_BR_MAGIC)
        elif mask_data.startswith(SEM_M5_SHIFT_BR_MAGIC):
            magic_len = len(SEM_M5_SHIFT_BR_MAGIC)
        else:
            magic_len = len(SEM_M5_BR_MAGIC)
        with tempfile.NamedTemporaryFile(suffix=".sm5", delete=False) as tmp_mask:
            tmp_mask.write(brotli.decompress(mask_data[magic_len:]))
            tmp_mask_path = tmp_mask.name
        try:
            mask_frames_all = torch.from_numpy(decode_seg_split_m5(tmp_mask_path)).contiguous()
        finally:
            os.remove(tmp_mask_path)
    elif mask_data.startswith(SEM_TOPBAND_BR_MAGIC):
        from seg_sparse_m5_codec import decode_seg_topband

        with tempfile.NamedTemporaryFile(suffix=".stbm", delete=False) as tmp_mask:
            tmp_mask.write(brotli.decompress(mask_data[len(SEM_TOPBAND_BR_MAGIC):]))
            tmp_mask_path = tmp_mask.name
        try:
            mask_frames_all = torch.from_numpy(decode_seg_topband(tmp_mask_path)).contiguous()
        finally:
            os.remove(tmp_mask_path)
    else:
        with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp_obu:
            tmp_obu.write(brotli.decompress(mask_data))
            tmp_obu_path = tmp_obu.name

        mask_frames_all = load_encoded_mask_video(tmp_obu_path)
        os.remove(tmp_obu_path)

    # 3. Load Pose Vectors
    if packed_poseq is not None:
        pose_frames_all, frame_bias_all = decode_control_payload(packed_poseq)
    else:
        pose_frames_all, frame_bias_all = load_pose_frames(data_dir), None

    out_h, out_w = 874, 1164
    cursor = 0
    batch_size = 4 
    
    # 1 mask per generated pair, assume 600 pairs per standard 1200 frame chunk.
    pairs_per_file = PAIRS_PER_FILE

    with torch.inference_mode():
        for file_name in files:
            base_name = os.path.splitext(file_name)[0]
            raw_out_path = out_dir / f"{base_name}.raw"
            
            # Retrieve exactly the pairs mapping to this file
            file_masks = mask_frames_all[cursor : cursor + pairs_per_file]
            file_poses = pose_frames_all[cursor : cursor + pairs_per_file]
            file_biases = frame_bias_all[cursor : cursor + pairs_per_file] if frame_bias_all is not None else None
            cursor += pairs_per_file
            
            with open(raw_out_path, "wb") as f_out:
                pbar = tqdm(range(0, file_masks.shape[0], batch_size), desc=f"Decoding {file_name}")
                
                for i in pbar:
                    in_mask2 = file_masks[i : i + batch_size].to(device)
                    if not torch.is_floating_point(in_mask2):
                        in_mask2 = in_mask2.long()
                    in_pose6 = file_poses[i : i + batch_size].to(device).float()

                    fake1, fake2 = generator(in_mask2, in_pose6)
                    if file_biases is not None:
                        bias = file_biases[i : i + batch_size].to(device).float()
                        if bias.shape[1] == 62:
                            fake1 = fake1 + bias[:, 0, None, None, None] + bias[:, 2:5, None, None]
                            fake2 = fake2 + bias[:, 1, None, None, None] + bias[:, 5:8, None, None]
                            band = bias[:, 8:20].reshape(-1, 2, 3, 2)
                            hband = bias[:, 20:32].reshape(-1, 2, 3, 2)
                            quad = bias[:, 32:38].reshape(-1, 2, 3)
                            vdetail = bias[:, 38:50].reshape(-1, 2, 3, 2)
                            hdetail = bias[:, 50:].reshape(-1, 2, 3, 2)
                            fake1 = fake1.clone()
                            fake2 = fake2.clone()
                            fake1[:, :, :192, :] = fake1[:, :, :192, :] + band[:, 0, :, 0, None, None]
                            fake1[:, :, 192:, :] = fake1[:, :, 192:, :] + band[:, 0, :, 1, None, None]
                            fake2[:, :, :192, :] = fake2[:, :, :192, :] + band[:, 1, :, 0, None, None]
                            fake2[:, :, 192:, :] = fake2[:, :, 192:, :] + band[:, 1, :, 1, None, None]
                            fake1[:, :, :, :256] = fake1[:, :, :, :256] + hband[:, 0, :, 0, None, None]
                            fake1[:, :, :, 256:] = fake1[:, :, :, 256:] + hband[:, 0, :, 1, None, None]
                            fake2[:, :, :, :256] = fake2[:, :, :, :256] + hband[:, 1, :, 0, None, None]
                            fake2[:, :, :, 256:] = fake2[:, :, :, 256:] + hband[:, 1, :, 1, None, None]
                            fake1[:, :, :192, :256] = fake1[:, :, :192, :256] + quad[:, 0, :, None, None]
                            fake1[:, :, :192, 256:] = fake1[:, :, :192, 256:] - quad[:, 0, :, None, None]
                            fake1[:, :, 192:, :256] = fake1[:, :, 192:, :256] - quad[:, 0, :, None, None]
                            fake1[:, :, 192:, 256:] = fake1[:, :, 192:, 256:] + quad[:, 0, :, None, None]
                            fake2[:, :, :192, :256] = fake2[:, :, :192, :256] + quad[:, 1, :, None, None]
                            fake2[:, :, :192, 256:] = fake2[:, :, :192, 256:] - quad[:, 1, :, None, None]
                            fake2[:, :, 192:, :256] = fake2[:, :, 192:, :256] - quad[:, 1, :, None, None]
                            fake2[:, :, 192:, 256:] = fake2[:, :, 192:, 256:] + quad[:, 1, :, None, None]
                            fake1[:, :, :96, :] = fake1[:, :, :96, :] + vdetail[:, 0, :, 0, None, None]
                            fake1[:, :, 96:192, :] = fake1[:, :, 96:192, :] - vdetail[:, 0, :, 0, None, None]
                            fake1[:, :, 192:288, :] = fake1[:, :, 192:288, :] + vdetail[:, 0, :, 1, None, None]
                            fake1[:, :, 288:, :] = fake1[:, :, 288:, :] - vdetail[:, 0, :, 1, None, None]
                            fake2[:, :, :96, :] = fake2[:, :, :96, :] + vdetail[:, 1, :, 0, None, None]
                            fake2[:, :, 96:192, :] = fake2[:, :, 96:192, :] - vdetail[:, 1, :, 0, None, None]
                            fake2[:, :, 192:288, :] = fake2[:, :, 192:288, :] + vdetail[:, 1, :, 1, None, None]
                            fake2[:, :, 288:, :] = fake2[:, :, 288:, :] - vdetail[:, 1, :, 1, None, None]
                            fake1[:, :, :, :128] = fake1[:, :, :, :128] + hdetail[:, 0, :, 0, None, None]
                            fake1[:, :, :, 128:256] = fake1[:, :, :, 128:256] - hdetail[:, 0, :, 0, None, None]
                            fake1[:, :, :, 256:384] = fake1[:, :, :, 256:384] + hdetail[:, 0, :, 1, None, None]
                            fake1[:, :, :, 384:] = fake1[:, :, :, 384:] - hdetail[:, 0, :, 1, None, None]
                            fake2[:, :, :, :128] = fake2[:, :, :, :128] + hdetail[:, 1, :, 0, None, None]
                            fake2[:, :, :, 128:256] = fake2[:, :, :, 128:256] - hdetail[:, 1, :, 0, None, None]
                            fake2[:, :, :, 256:384] = fake2[:, :, :, 256:384] + hdetail[:, 1, :, 1, None, None]
                            fake2[:, :, :, 384:] = fake2[:, :, :, 384:] - hdetail[:, 1, :, 1, None, None]
                        elif bias.shape[1] == 50:
                            fake1 = fake1 + bias[:, 0, None, None, None] + bias[:, 2:5, None, None]
                            fake2 = fake2 + bias[:, 1, None, None, None] + bias[:, 5:8, None, None]
                            band = bias[:, 8:20].reshape(-1, 2, 3, 2)
                            hband = bias[:, 20:32].reshape(-1, 2, 3, 2)
                            quad = bias[:, 32:38].reshape(-1, 2, 3)
                            vdetail = bias[:, 38:].reshape(-1, 2, 3, 2)
                            fake1 = fake1.clone()
                            fake2 = fake2.clone()
                            fake1[:, :, :192, :] = fake1[:, :, :192, :] + band[:, 0, :, 0, None, None]
                            fake1[:, :, 192:, :] = fake1[:, :, 192:, :] + band[:, 0, :, 1, None, None]
                            fake2[:, :, :192, :] = fake2[:, :, :192, :] + band[:, 1, :, 0, None, None]
                            fake2[:, :, 192:, :] = fake2[:, :, 192:, :] + band[:, 1, :, 1, None, None]
                            fake1[:, :, :, :256] = fake1[:, :, :, :256] + hband[:, 0, :, 0, None, None]
                            fake1[:, :, :, 256:] = fake1[:, :, :, 256:] + hband[:, 0, :, 1, None, None]
                            fake2[:, :, :, :256] = fake2[:, :, :, :256] + hband[:, 1, :, 0, None, None]
                            fake2[:, :, :, 256:] = fake2[:, :, :, 256:] + hband[:, 1, :, 1, None, None]
                            fake1[:, :, :192, :256] = fake1[:, :, :192, :256] + quad[:, 0, :, None, None]
                            fake1[:, :, :192, 256:] = fake1[:, :, :192, 256:] - quad[:, 0, :, None, None]
                            fake1[:, :, 192:, :256] = fake1[:, :, 192:, :256] - quad[:, 0, :, None, None]
                            fake1[:, :, 192:, 256:] = fake1[:, :, 192:, 256:] + quad[:, 0, :, None, None]
                            fake2[:, :, :192, :256] = fake2[:, :, :192, :256] + quad[:, 1, :, None, None]
                            fake2[:, :, :192, 256:] = fake2[:, :, :192, 256:] - quad[:, 1, :, None, None]
                            fake2[:, :, 192:, :256] = fake2[:, :, 192:, :256] - quad[:, 1, :, None, None]
                            fake2[:, :, 192:, 256:] = fake2[:, :, 192:, 256:] + quad[:, 1, :, None, None]
                            fake1[:, :, :96, :] = fake1[:, :, :96, :] + vdetail[:, 0, :, 0, None, None]
                            fake1[:, :, 96:192, :] = fake1[:, :, 96:192, :] - vdetail[:, 0, :, 0, None, None]
                            fake1[:, :, 192:288, :] = fake1[:, :, 192:288, :] + vdetail[:, 0, :, 1, None, None]
                            fake1[:, :, 288:, :] = fake1[:, :, 288:, :] - vdetail[:, 0, :, 1, None, None]
                            fake2[:, :, :96, :] = fake2[:, :, :96, :] + vdetail[:, 1, :, 0, None, None]
                            fake2[:, :, 96:192, :] = fake2[:, :, 96:192, :] - vdetail[:, 1, :, 0, None, None]
                            fake2[:, :, 192:288, :] = fake2[:, :, 192:288, :] + vdetail[:, 1, :, 1, None, None]
                            fake2[:, :, 288:, :] = fake2[:, :, 288:, :] - vdetail[:, 1, :, 1, None, None]
                        elif bias.shape[1] == 38:
                            fake1 = fake1 + bias[:, 0, None, None, None] + bias[:, 2:5, None, None]
                            fake2 = fake2 + bias[:, 1, None, None, None] + bias[:, 5:8, None, None]
                            band = bias[:, 8:20].reshape(-1, 2, 3, 2)
                            hband = bias[:, 20:32].reshape(-1, 2, 3, 2)
                            quad = bias[:, 32:].reshape(-1, 2, 3)
                            fake1 = fake1.clone()
                            fake2 = fake2.clone()
                            fake1[:, :, :192, :] = fake1[:, :, :192, :] + band[:, 0, :, 0, None, None]
                            fake1[:, :, 192:, :] = fake1[:, :, 192:, :] + band[:, 0, :, 1, None, None]
                            fake2[:, :, :192, :] = fake2[:, :, :192, :] + band[:, 1, :, 0, None, None]
                            fake2[:, :, 192:, :] = fake2[:, :, 192:, :] + band[:, 1, :, 1, None, None]
                            fake1[:, :, :, :256] = fake1[:, :, :, :256] + hband[:, 0, :, 0, None, None]
                            fake1[:, :, :, 256:] = fake1[:, :, :, 256:] + hband[:, 0, :, 1, None, None]
                            fake2[:, :, :, :256] = fake2[:, :, :, :256] + hband[:, 1, :, 0, None, None]
                            fake2[:, :, :, 256:] = fake2[:, :, :, 256:] + hband[:, 1, :, 1, None, None]
                            fake1[:, :, :192, :256] = fake1[:, :, :192, :256] + quad[:, 0, :, None, None]
                            fake1[:, :, :192, 256:] = fake1[:, :, :192, 256:] - quad[:, 0, :, None, None]
                            fake1[:, :, 192:, :256] = fake1[:, :, 192:, :256] - quad[:, 0, :, None, None]
                            fake1[:, :, 192:, 256:] = fake1[:, :, 192:, 256:] + quad[:, 0, :, None, None]
                            fake2[:, :, :192, :256] = fake2[:, :, :192, :256] + quad[:, 1, :, None, None]
                            fake2[:, :, :192, 256:] = fake2[:, :, :192, 256:] - quad[:, 1, :, None, None]
                            fake2[:, :, 192:, :256] = fake2[:, :, 192:, :256] - quad[:, 1, :, None, None]
                            fake2[:, :, 192:, 256:] = fake2[:, :, 192:, 256:] + quad[:, 1, :, None, None]
                        elif bias.shape[1] == 32:
                            fake1 = fake1 + bias[:, 0, None, None, None] + bias[:, 2:5, None, None]
                            fake2 = fake2 + bias[:, 1, None, None, None] + bias[:, 5:8, None, None]
                            band = bias[:, 8:20].reshape(-1, 2, 3, 2)
                            hband = bias[:, 20:].reshape(-1, 2, 3, 2)
                            fake1 = fake1.clone()
                            fake2 = fake2.clone()
                            fake1[:, :, :192, :] = fake1[:, :, :192, :] + band[:, 0, :, 0, None, None]
                            fake1[:, :, 192:, :] = fake1[:, :, 192:, :] + band[:, 0, :, 1, None, None]
                            fake2[:, :, :192, :] = fake2[:, :, :192, :] + band[:, 1, :, 0, None, None]
                            fake2[:, :, 192:, :] = fake2[:, :, 192:, :] + band[:, 1, :, 1, None, None]
                            fake1[:, :, :, :256] = fake1[:, :, :, :256] + hband[:, 0, :, 0, None, None]
                            fake1[:, :, :, 256:] = fake1[:, :, :, 256:] + hband[:, 0, :, 1, None, None]
                            fake2[:, :, :, :256] = fake2[:, :, :, :256] + hband[:, 1, :, 0, None, None]
                            fake2[:, :, :, 256:] = fake2[:, :, :, 256:] + hband[:, 1, :, 1, None, None]
                        elif bias.shape[1] == 20:
                            fake1 = fake1 + bias[:, 0, None, None, None] + bias[:, 2:5, None, None]
                            fake2 = fake2 + bias[:, 1, None, None, None] + bias[:, 5:8, None, None]
                            band = bias[:, 8:].reshape(-1, 2, 3, 2)
                            fake1 = fake1.clone()
                            fake2 = fake2.clone()
                            fake1[:, :, :192, :] = fake1[:, :, :192, :] + band[:, 0, :, 0, None, None]
                            fake1[:, :, 192:, :] = fake1[:, :, 192:, :] + band[:, 0, :, 1, None, None]
                            fake2[:, :, :192, :] = fake2[:, :, :192, :] + band[:, 1, :, 0, None, None]
                            fake2[:, :, 192:, :] = fake2[:, :, 192:, :] + band[:, 1, :, 1, None, None]
                        elif bias.shape[1] == 8:
                            fake1 = fake1 + bias[:, 0, None, None, None] + bias[:, 2:5, None, None]
                            fake2 = fake2 + bias[:, 1, None, None, None] + bias[:, 5:8, None, None]
                        else:
                            fake1 = fake1 + bias[:, 0, None, None, None]
                            fake2 = fake2 + bias[:, 1, None, None, None]

                    fake1_up = F.interpolate(fake1, size=(out_h, out_w), mode="bilinear", align_corners=False)
                    fake2_up = F.interpolate(fake2, size=(out_h, out_w), mode="bilinear", align_corners=False)
                    if file_biases is not None and file_biases.shape[1] == 4:
                        gain = file_biases[i : i + batch_size, 2:4].to(device).float()
                        fake1_up = fake1_up.clamp(0, 255).round() * gain[:, 0, None, None, None]
                        fake2_up = fake2_up.clamp(0, 255).round() * gain[:, 1, None, None, None]

                    batch_comp = torch.stack([fake1_up, fake2_up], dim=1)
                    batch_comp = einops.rearrange(batch_comp, "b t c h w -> (b t) h w c")

                    output_bytes = batch_comp.clamp(0, 255).round().to(torch.uint8)
                    f_out.write(output_bytes.cpu().numpy().tobytes())

if __name__ == "__main__":
    main()
