#!/usr/bin/env python
import io
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
    if payload_data.startswith(b"QZS3"):
        return get_grouped_qv_state_dict(payload_data, device)
    if payload_data.startswith(b"QZS2"):
        return get_grouped_q10_state_dict(payload_data, device)
    if payload_data.startswith(b"QZS1"):
        return get_grouped_compact_state_dict(payload_data, device)
    if payload_data.startswith(b"QZC1") or payload_data.startswith(b"QZC2") or payload_data.startswith(b"QZC3"):
        return get_compact_state_dict(payload_data, device)

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

def get_compact_state_dict(payload_data, device: torch.device):
    quant_all_side = payload_data.startswith(b"QZC2")
    quant_dense_side = payload_data.startswith(b"QZC3")
    block_size = int.from_bytes(payload_data[4:6], "little")
    view = memoryview(payload_data)
    offset = 6
    template = JointFrameGenerator()
    state_dict = {}
    covered_keys = set()

    def take(count):
        nonlocal offset
        out = view[offset:offset + count]
        offset += count
        return out

    def take_fp(shape, quantized=False):
        count = int(np.prod(shape))
        if not quantized or count < 16:
            return torch.frombuffer(take(count * 2), dtype=torch.float16).clone().reshape(shape).to(device).float()
        mn = torch.frombuffer(take(2), dtype=torch.float16).clone().float()
        step = torch.frombuffer(take(2), dtype=torch.float16).clone().float()
        q = torch.frombuffer(take(count), dtype=torch.uint8).clone().reshape(shape).float()
        return (mn + q * step).to(device)

    for name, module in template.named_modules():
        if isinstance(module, (QConv2d, QEmbedding)):
            weight_shape = tuple(module.weight.shape)
            covered_keys.add(f"{name}.weight")
            if getattr(module, "quantize_weight", False):
                weight_numel = int(module.weight.numel())
                scale_count = (weight_numel + block_size - 1) // block_size
                packed_count = (scale_count * block_size + 1) // 2
                packed = torch.frombuffer(take(packed_count), dtype=torch.uint8).clone().to(device)
                scales = take_fp((scale_count,), quantized=quant_all_side)
                nibbles = unpack_nibbles(packed, packed.numel() * 2)
                weight = FP4Codebook.dequantize_from_nibbles(nibbles, scales, weight_shape)
            else:
                weight = take_fp(weight_shape, quantized=quant_all_side)
            state_dict[f"{name}.weight"] = weight.float()

            if isinstance(module, QConv2d) and module.bias is not None:
                covered_keys.add(f"{name}.bias")
                state_dict[f"{name}.bias"] = take_fp(tuple(module.bias.shape), quantized=quant_all_side)

    for key, tensor in template.state_dict().items():
        if key in covered_keys:
            continue
        if torch.is_floating_point(tensor):
            state_dict[key] = take_fp(tuple(tensor.shape), quantized=(quant_all_side or quant_dense_side))
        else:
            state_dict[key] = torch.frombuffer(take(tensor.numel() * tensor.element_size()), dtype=tensor.dtype).clone().reshape(tuple(tensor.shape)).to(device)
    return state_dict

def get_grouped_compact_state_dict(payload_data, device: torch.device):
    block_size = int.from_bytes(payload_data[4:6], "little")
    template = JointFrameGenerator()
    specs = []
    covered_keys = set()
    sizes = {
        "packed": 0,
        "scales": 0,
        "bias": 0,
        "dense_fp": 0,
        "fp_weight": 0,
        "dense_other": 0,
    }

    for name, module in template.named_modules():
        if not isinstance(module, (QConv2d, QEmbedding)):
            continue
        weight_shape = tuple(module.weight.shape)
        covered_keys.add(f"{name}.weight")
        if getattr(module, "quantize_weight", False):
            weight_numel = int(module.weight.numel())
            scale_count = (weight_numel + block_size - 1) // block_size
            packed_count = (scale_count * block_size + 1) // 2
            specs.append((name, "q", weight_shape, packed_count, scale_count))
            sizes["packed"] += packed_count
            sizes["scales"] += scale_count * 2
        else:
            specs.append((name, "fp", weight_shape, int(module.weight.numel())))
            sizes["fp_weight"] += int(module.weight.numel()) * 2
        if isinstance(module, QConv2d) and module.bias is not None:
            covered_keys.add(f"{name}.bias")
            specs.append((name, "bias", tuple(module.bias.shape), int(module.bias.numel())))
            sizes["bias"] += int(module.bias.numel()) * 2

    dense_specs = []
    for key, tensor in template.state_dict().items():
        if key in covered_keys:
            continue
        if torch.is_floating_point(tensor):
            dense_specs.append((key, "dense_fp", tuple(tensor.shape), int(tensor.numel()), tensor.dtype))
            sizes["dense_fp"] += int(tensor.numel()) * 2
        else:
            dense_specs.append((key, "dense_other", tuple(tensor.shape), int(tensor.numel()), tensor.dtype))
            sizes["dense_other"] += int(tensor.numel()) * tensor.element_size()

    view = memoryview(payload_data)
    offset = 6
    segments = {}
    for key in ("packed", "scales", "bias", "dense_fp", "fp_weight", "dense_other"):
        segments[key] = [view[offset:offset + sizes[key]], 0]
        offset += sizes[key]

    def take_from(key, count):
        segment, pos = segments[key]
        out = segment[pos:pos + count]
        segments[key][1] = pos + count
        return out

    state_dict = {}
    for spec in specs:
        name, kind = spec[0], spec[1]
        if kind == "q":
            _, _, weight_shape, packed_count, scale_count = spec
            packed = torch.frombuffer(take_from("packed", packed_count), dtype=torch.uint8).clone().to(device)
            scales = torch.frombuffer(take_from("scales", scale_count * 2), dtype=torch.float16).clone().to(device).float()
            nibbles = unpack_nibbles(packed, packed.numel() * 2)
            weight = FP4Codebook.dequantize_from_nibbles(nibbles, scales, weight_shape)
            state_dict[f"{name}.weight"] = weight.float()
        elif kind == "fp":
            _, _, weight_shape, count = spec
            state_dict[f"{name}.weight"] = torch.frombuffer(take_from("fp_weight", count * 2), dtype=torch.float16).clone().reshape(weight_shape).to(device).float()
        else:
            _, _, bias_shape, count = spec
            state_dict[f"{name}.bias"] = torch.frombuffer(take_from("bias", count * 2), dtype=torch.float16).clone().reshape(bias_shape).to(device).float()

    for key, kind, shape, count, dtype in dense_specs:
        if kind == "dense_fp":
            state_dict[key] = torch.frombuffer(take_from("dense_fp", count * 2), dtype=torch.float16).clone().reshape(shape).to(device).float()
        else:
            state_dict[key] = torch.frombuffer(take_from("dense_other", count * torch.empty((), dtype=dtype).element_size()), dtype=dtype).clone().reshape(shape).to(device)
    return state_dict

def unpack_q10(data: memoryview, count: int) -> torch.Tensor:
    raw = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(count, dtype=np.uint16)
    acc = 0
    bits = 0
    j = 0
    for byte in raw:
        acc |= int(byte) << bits
        bits += 8
        while bits >= 10 and j < count:
            out[j] = acc & 0x3FF
            acc >>= 10
            bits -= 10
            j += 1
    return torch.from_numpy(out.astype(np.float32, copy=False))

def unpack_qbits(data: memoryview, count: int, width: int) -> torch.Tensor:
    raw = np.frombuffer(data, dtype=np.uint8)
    mask = (1 << width) - 1
    out = np.empty(count, dtype=np.uint16)
    acc = 0
    bits = 0
    j = 0
    for byte in raw:
        acc |= int(byte) << bits
        bits += 8
        while bits >= width and j < count:
            out[j] = acc & mask
            acc >>= width
            bits -= width
            j += 1
    return torch.from_numpy(out.astype(np.float32, copy=False))

def get_qv_specs():
    specs = {
        "frame1_head.block1.film_proj.weight": (9, False),
        "pose_mlp.2.weight": (10, True),
    }
    for key in [
        "frame1_head.block1.conv1.norm.weight",
        "frame1_head.block1.conv1.norm.bias",
        "frame1_head.block1.norm2.weight",
        "frame1_head.block1.norm2.bias",
        "frame1_head.block1.film_proj.bias",
        "frame1_head.block2.conv1.norm.weight",
        "frame1_head.block2.conv1.norm.bias",
        "frame1_head.block2.norm2.weight",
        "frame1_head.block2.norm2.bias",
        "frame1_head.pre.norm.weight",
        "frame1_head.pre.norm.bias",
    ]:
        specs[key] = (8, False)
    for key in [
        "frame2_head.block1.conv1.norm.weight",
        "frame2_head.block1.conv1.norm.bias",
        "frame2_head.block1.norm2.weight",
        "frame2_head.block1.norm2.bias",
        "frame2_head.block2.conv1.norm.weight",
        "frame2_head.block2.conv1.norm.bias",
        "frame2_head.block2.norm2.weight",
        "frame2_head.block2.norm2.bias",
        "frame2_head.pre.norm.weight",
        "frame2_head.pre.norm.bias",
    ]:
        specs[key] = (8, False)
    return specs

def get_grouped_qv_state_dict(payload_data, device: torch.device):
    block_size = int.from_bytes(payload_data[4:6], "little")
    qv_specs = get_qv_specs()
    template = JointFrameGenerator()
    specs = []
    dense_specs = []
    covered_keys = set()
    sizes = {"packed": 0, "scales": 0, "bias": 0, "dense_fp": 0, "fp_weight": 0, "dense_other": 0, "qv": 0}

    for name, module in template.named_modules():
        if not isinstance(module, (QConv2d, QEmbedding)):
            continue
        weight_shape = tuple(module.weight.shape)
        covered_keys.add(f"{name}.weight")
        if getattr(module, "quantize_weight", False):
            weight_numel = int(module.weight.numel())
            scale_count = (weight_numel + block_size - 1) // block_size
            packed_count = (scale_count * block_size + 1) // 2
            specs.append((name, "q", weight_shape, packed_count, scale_count))
            sizes["packed"] += packed_count
            sizes["scales"] += scale_count * 2
        else:
            specs.append((name, "fp", weight_shape, int(module.weight.numel())))
            sizes["fp_weight"] += int(module.weight.numel()) * 2
        if isinstance(module, QConv2d) and module.bias is not None:
            covered_keys.add(f"{name}.bias")
            specs.append((name, "bias", tuple(module.bias.shape), int(module.bias.numel())))
            sizes["bias"] += int(module.bias.numel()) * 2

    for key, tensor in template.state_dict().items():
        if key in covered_keys:
            continue
        shape = tuple(tensor.shape)
        count = int(tensor.numel())
        if key in qv_specs:
            bits, per_row = qv_specs[key]
            rows = shape[0] if per_row and len(shape) >= 2 else 1
            sizes["qv"] += rows * 4 + (count * bits + 7) // 8
            dense_specs.append((key, "qv", shape, count, tensor.dtype, bits, rows))
        elif torch.is_floating_point(tensor):
            sizes["dense_fp"] += count * 2
            dense_specs.append((key, "dense_fp", shape, count, tensor.dtype, 0, 0))
        else:
            sizes["dense_other"] += count * tensor.element_size()
            dense_specs.append((key, "dense_other", shape, count, tensor.dtype, 0, 0))

    view = memoryview(payload_data)
    offset = 6
    segments = {}
    for key in ("packed", "scales", "bias", "dense_fp", "fp_weight", "dense_other", "qv"):
        segments[key] = [view[offset:offset + sizes[key]], 0]
        offset += sizes[key]

    def take_from(key, count):
        segment, pos = segments[key]
        out = segment[pos:pos + count]
        segments[key][1] = pos + count
        return out

    state_dict = {}
    for spec in specs:
        name, kind = spec[0], spec[1]
        if kind == "q":
            _, _, weight_shape, packed_count, scale_count = spec
            packed = torch.frombuffer(take_from("packed", packed_count), dtype=torch.uint8).clone().to(device)
            scales = torch.frombuffer(take_from("scales", scale_count * 2), dtype=torch.float16).clone().to(device).float()
            nibbles = unpack_nibbles(packed, packed.numel() * 2)
            state_dict[f"{name}.weight"] = FP4Codebook.dequantize_from_nibbles(nibbles, scales, weight_shape).float()
        elif kind == "fp":
            _, _, weight_shape, count = spec
            state_dict[f"{name}.weight"] = torch.frombuffer(take_from("fp_weight", count * 2), dtype=torch.float16).clone().reshape(weight_shape).to(device).float()
        else:
            _, _, bias_shape, count = spec
            state_dict[f"{name}.bias"] = torch.frombuffer(take_from("bias", count * 2), dtype=torch.float16).clone().reshape(bias_shape).to(device).float()

    for key, kind, shape, count, dtype, bits, rows in dense_specs:
        if kind == "dense_fp":
            state_dict[key] = torch.frombuffer(take_from("dense_fp", count * 2), dtype=torch.float16).clone().reshape(shape).to(device).float()
        elif kind == "dense_other":
            state_dict[key] = torch.frombuffer(take_from("dense_other", count * torch.empty((), dtype=dtype).element_size()), dtype=dtype).clone().reshape(shape).to(device)
        else:
            meta = take_from("qv", rows * 4)
            mn_step = torch.frombuffer(meta, dtype=torch.float16).clone().reshape(rows, 2).float()
            packed_count = (count * bits + 7) // 8
            q = unpack_qbits(take_from("qv", packed_count), count, bits).reshape(rows, -1)
            value = mn_step[:, :1] + q * mn_step[:, 1:].clamp_min(1e-8)
            state_dict[key] = value.reshape(shape).to(device).float()
    return state_dict

def get_grouped_q10_state_dict(payload_data, device: torch.device):
    block_size = int.from_bytes(payload_data[4:6], "little")
    template = JointFrameGenerator()
    q10_keys = {
        "frame1_head.block1.film_proj.weight",
        "pose_mlp.2.weight",
    }
    specs = []
    dense_specs = []
    covered_keys = set()
    sizes = {"packed": 0, "scales": 0, "bias": 0, "dense_fp": 0, "fp_weight": 0, "dense_other": 0, "q10": 0}

    for name, module in template.named_modules():
        if not isinstance(module, (QConv2d, QEmbedding)):
            continue
        weight_shape = tuple(module.weight.shape)
        covered_keys.add(f"{name}.weight")
        if getattr(module, "quantize_weight", False):
            weight_numel = int(module.weight.numel())
            scale_count = (weight_numel + block_size - 1) // block_size
            packed_count = (scale_count * block_size + 1) // 2
            specs.append((name, "q", weight_shape, packed_count, scale_count))
            sizes["packed"] += packed_count
            sizes["scales"] += scale_count * 2
        else:
            specs.append((name, "fp", weight_shape, int(module.weight.numel())))
            sizes["fp_weight"] += int(module.weight.numel()) * 2
        if isinstance(module, QConv2d) and module.bias is not None:
            covered_keys.add(f"{name}.bias")
            specs.append((name, "bias", tuple(module.bias.shape), int(module.bias.numel())))
            sizes["bias"] += int(module.bias.numel()) * 2

    for key, tensor in template.state_dict().items():
        if key in covered_keys:
            continue
        shape = tuple(tensor.shape)
        count = int(tensor.numel())
        if key in q10_keys:
            rows = shape[0]
            sizes["q10"] += rows * 4 + (count * 10 + 7) // 8
            dense_specs.append((key, "q10", shape, count, tensor.dtype))
        elif torch.is_floating_point(tensor):
            sizes["dense_fp"] += count * 2
            dense_specs.append((key, "dense_fp", shape, count, tensor.dtype))
        else:
            sizes["dense_other"] += count * tensor.element_size()
            dense_specs.append((key, "dense_other", shape, count, tensor.dtype))

    view = memoryview(payload_data)
    offset = 6
    segments = {}
    for key in ("packed", "scales", "bias", "dense_fp", "fp_weight", "dense_other", "q10"):
        segments[key] = [view[offset:offset + sizes[key]], 0]
        offset += sizes[key]

    def take_from(key, count):
        segment, pos = segments[key]
        out = segment[pos:pos + count]
        segments[key][1] = pos + count
        return out

    state_dict = {}
    for spec in specs:
        name, kind = spec[0], spec[1]
        if kind == "q":
            _, _, weight_shape, packed_count, scale_count = spec
            packed = torch.frombuffer(take_from("packed", packed_count), dtype=torch.uint8).clone().to(device)
            scales = torch.frombuffer(take_from("scales", scale_count * 2), dtype=torch.float16).clone().to(device).float()
            nibbles = unpack_nibbles(packed, packed.numel() * 2)
            state_dict[f"{name}.weight"] = FP4Codebook.dequantize_from_nibbles(nibbles, scales, weight_shape).float()
        elif kind == "fp":
            _, _, weight_shape, count = spec
            state_dict[f"{name}.weight"] = torch.frombuffer(take_from("fp_weight", count * 2), dtype=torch.float16).clone().reshape(weight_shape).to(device).float()
        else:
            _, _, bias_shape, count = spec
            state_dict[f"{name}.bias"] = torch.frombuffer(take_from("bias", count * 2), dtype=torch.float16).clone().reshape(bias_shape).to(device).float()

    for key, kind, shape, count, dtype in dense_specs:
        if kind == "dense_fp":
            state_dict[key] = torch.frombuffer(take_from("dense_fp", count * 2), dtype=torch.float16).clone().reshape(shape).to(device).float()
        elif kind == "dense_other":
            state_dict[key] = torch.frombuffer(take_from("dense_other", count * torch.empty((), dtype=dtype).element_size()), dtype=dtype).clone().reshape(shape).to(device)
        else:
            rows = shape[0]
            meta = take_from("q10", rows * 4)
            mn_step = torch.frombuffer(meta, dtype=torch.float16).clone().reshape(rows, 2).float()
            packed_count = (count * 10 + 7) // 8
            q = unpack_q10(take_from("q10", packed_count), count).reshape(rows, -1)
            value = mn_step[:, :1] + q * mn_step[:, 1:].clamp_min(1e-8)
            state_dict[key] = value.reshape(shape).to(device).float()
    return state_dict

# -----------------------------
# Architecture (Inference Only)
# -----------------------------

class QConv2d(nn.Conv2d):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.quantize_weight = quantize_weight

class QEmbedding(nn.Embedding):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.quantize_weight = quantize_weight

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
        e2 = self.embedding(mask2.long()).permute(0, 3, 1, 2)
        e2_up = F.interpolate(e2, size=coords.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([e2_up, coords], dim=1)
        s = self.stem_block(self.stem_conv(x))
        z = self.down_block(self.down_conv(s))
        z = self.up(z)
        f = self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))
        return f

class Frame2StaticHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 36, depth_mult: int = 4):
        super().__init__()
        self.block1 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.block1(feat)
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
            in_ch=56, hidden=52, depth_mult=depth_mult)

    def forward(self, mask2: torch.Tensor, pose6: torch.Tensor):
        b = mask2.shape[0]
        coords = make_coord_grid(b, 384, 512, mask2.device, torch.float32)

        shared_feat = self.shared_trunk(mask2, coords)
        pred_frame2 = self.frame2_head(shared_feat)

        cond_emb = self.pose_mlp(pose6)
        pred_frame1 = self.frame1_head(shared_feat, cond_emb)

        return pred_frame1, pred_frame2

def make_coord_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)

def make_dct_basis(k: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    freqs = []
    max_freq = 16
    for fy in range(max_freq):
        for fx in range(max_freq):
            if fx == 0 and fy == 0:
                continue
            freqs.append((fx, fy, fx * fx + fy * fy))
    freqs.sort(key=lambda item: item[2])

    patterns = []
    for channel in range(3):
        for fx, fy, _ in freqs:
            pat = torch.cos(np.pi * fx * xx) * torch.cos(np.pi * fy * yy)
            chans = torch.zeros(3, h, w, device=device)
            chans[channel] = pat
            patterns.append(chans)
            if len(patterns) >= k:
                basis = torch.stack(patterns, dim=0)
                return basis / basis.flatten(1).std(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    raise ValueError(f"not enough DCT basis patterns for k={k}")

def load_actuator(path: Path, device: torch.device):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        payload = np.load(io.BytesIO(brotli.decompress(f.read())))
    q = payload["q"].astype(np.float32)
    scale = payload["scale"].astype(np.float32)
    basis_k = int(payload["basis_k"][0])
    basis_kind = str(payload["basis_kind"][0])
    base = str(payload["base"][0]) if "base" in payload else "frame1"
    payload.close()
    if basis_kind != "dct":
        raise ValueError(f"unsupported actuator basis: {basis_kind}")
    return {
        "alpha": torch.from_numpy(q * scale).to(device=device, dtype=torch.float32),
        "basis": make_dct_basis(basis_k, 384, 512, device),
        "base": base,
    }

def seg_tile_action_specs(device: torch.device):
    specs = []
    directions = [
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (-0.35, 0.15, 0.45),
        (0.25, 0.15, -0.20),
    ]
    for vec in directions:
        v = torch.tensor(vec, dtype=torch.float32, device=device).view(3, 1, 1)
        v = v / v.abs().max().clamp_min(1e-6)
        for amp in (2.0, 4.0, 6.0, 8.0, 12.0, 16.0):
            specs.append(v * amp)
            specs.append(-v * amp)
    return torch.stack(specs, dim=0)


def load_seg_tile_actions_data(data: bytes, device: torch.device):
    raw = brotli.decompress(data)
    records = []
    if raw.startswith(b"TA4"):
        raw = raw[3:]
        record_width = 4
    elif raw.startswith(b"TA5"):
        raw = raw[3:]
        record_width = 5
    elif len(raw) % 4 == 0 and len(raw) % 5 != 0:
        record_width = 4
    elif len(raw) % 5 == 0 and len(raw) % 4 != 0:
        record_width = 5
    elif not raw:
        record_width = 4
    else:
        raise ValueError(f"ambiguous seg tile action payload length without TA4/TA5 header: {len(raw)}")

    if record_width == 4:
        for i in range(0, len(raw), 4):
            frame = int.from_bytes(raw[i:i + 2], "little")
            tile = raw[i + 2]
            action = raw[i + 3]
            records.append((frame, tile, action))
    elif record_width == 5:
        for i in range(0, len(raw), 5):
            frame = int.from_bytes(raw[i:i + 2], "little")
            tile = int.from_bytes(raw[i + 2:i + 4], "little")
            action = raw[i + 4]
            records.append((frame, tile, action))

    action_count = seg_tile_action_specs(device).shape[0]
    for frame, tile, action in records:
        if not (0 <= frame < 600):
            raise ValueError(f"seg tile action frame out of range: {frame}")
        if action >= action_count:
            raise ValueError(f"seg tile action index out of range: {action}")

    by_frame = {}
    for frame, tile, action in records:
        by_frame.setdefault(frame, []).append((tile, action))
    return {"by_frame": by_frame, "deltas": seg_tile_action_specs(device)}


def load_seg_tile_actions(path: Path, device: torch.device):
    if not path.exists():
        return None
    return load_seg_tile_actions_data(path.read_bytes(), device)


def load_smooth_pose(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        payload = np.load(io.BytesIO(brotli.decompress(f.read())))
    coeff = payload["coeff"].astype(np.float32)
    basis_kind = str(payload["basis_kind"][0]) if "basis_kind" in payload else "poly_fourier"
    scale = payload["scale"].astype(np.float32) if "scale" in payload else None
    payload.close()
    return {"coeff": coeff, "basis_kind": basis_kind, "scale": scale}


def make_smooth_pose_basis(num_pairs: int, basis_kind: str) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, num_pairs, dtype=np.float32)
    cols = [np.ones_like(t), t, t * t, t * t * t]
    if basis_kind == "poly_fourier":
        u = (t + 1.0) * 0.5
        for f in (1.0, 2.0, 3.0, 4.0):
            cols.append(np.sin(np.float32(2.0 * np.pi * f) * u))
            cols.append(np.cos(np.float32(2.0 * np.pi * f) * u))
    elif basis_kind != "poly":
        raise ValueError(f"unsupported smooth pose basis: {basis_kind}")
    return np.stack(cols, axis=1).astype(np.float32)


# -----------------------------
# Inference Helpers & Main
# -----------------------------
def load_encoded_mask_video(path: str) -> torch.Tensor:
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="gray")
        cls_img = np.round(img / 63.0).astype(np.uint8)
        cls_img = np.clip(cls_img, 0, 4)
        frames.append(cls_img)
    container.close()
    return torch.from_numpy(np.stack(frames)).contiguous()

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
    pose_br = data_dir / "pose.npy.br"
    pose_q_br = data_dir / "pose_q.br"
    packed_payload = data_dir / "p"
    color_lut_br = data_dir / "color_lut.npy.br"
    actuator_br = data_dir / "actuator.npz.br"
    seg_tile_actions_br = data_dir / "seg_tile_actions.br"
    smooth_pose_br = data_dir / "smooth_pose.npz.br"
    embedded_seg_tile_actions = None

    if packed_payload.exists():
        payload = packed_payload.read_bytes()
        if payload.startswith(b"P3"):
            mask_len, model_br_len, actions_len = struct.unpack_from("<IHH", payload, 2)
            cursor = 10
            mask_br_data = payload[cursor : cursor + mask_len]
            cursor += mask_len
            model_br_data = payload[cursor : cursor + model_br_len]
            cursor += model_br_len
            embedded_seg_tile_actions = payload[cursor : cursor + actions_len]
            pose_q_br_data = payload[cursor + actions_len:]
        elif payload.startswith(b"P2"):
            mask_len, model_br_len = struct.unpack_from("<IH", payload, 2)
            cursor = 8
            mask_br_data = payload[cursor : cursor + mask_len]
            cursor += mask_len
            model_br_data = payload[cursor : cursor + model_br_len]
            pose_q_br_data = payload[cursor + model_br_len:]
        elif len(payload) in (276641, 276520):
            mask_br_len = 219472
            model_br_len = 56034 if len(payload) == 276641 else 55914
            actions_len = 236
            cursor = 0
            mask_br_data = payload[cursor : cursor + mask_br_len]
            cursor += mask_br_len
            model_br_data = payload[cursor : cursor + model_br_len]
            cursor += model_br_len
            embedded_seg_tile_actions = payload[cursor : cursor + actions_len]
            pose_q_br_data = payload[cursor + actions_len:]
        else:
            mask_br_data = payload[:219472]
            if 276430 <= len(payload) <= 276470:
                model_br_len = 56093
            elif 276550 <= len(payload) <= 276610:
                model_br_len = 56221
            elif 278100 <= len(payload) <= 278130:
                model_br_len = 57757
            elif 277400 <= len(payload) <= 277430:
                model_br_len = 57053
            elif 277350 <= len(payload) <= 277399:
                model_br_len = 57031
            elif len(payload) == 281240:
                model_br_len = 60880
            else:
                model_br_len = 61147
            model_br_data = payload[219472:219472 + model_br_len]
            pose_q_br_data = payload[219472 + model_br_len:]
    else:
        mask_br_data = mask_br.read_bytes()
        model_br_data = model_br.read_bytes()
        pose_q_br_data = pose_q_br.read_bytes() if pose_q_br.exists() else None
 
    generator = JointFrameGenerator().to(device)

    # 1. Load Weights
    weights_data = brotli.decompress(model_br_data)
    
    generator.load_state_dict(get_decoded_state_dict(weights_data, device), strict=True)
    generator.eval()

    # 2. Load Mask Video (.obu)
    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp_obu:
        tmp_obu.write(brotli.decompress(mask_br_data))
        tmp_obu_path = tmp_obu.name

    mask_frames_all = load_encoded_mask_video(tmp_obu_path)
    if mask_frames_all.shape[0] < 600:
        pad_count = 600 - mask_frames_all.shape[0]
        tail = mask_frames_all[-1:].expand(pad_count, -1, -1, -1)
        mask_frames_all = torch.cat([mask_frames_all, tail], dim=0).contiguous()
    os.remove(tmp_obu_path)

    # 3. Load Pose Vectors
    if pose_q_br_data is not None:
        pose_raw = brotli.decompress(pose_q_br_data)
        if pose_raw.startswith(b"QP1"):
            first = np.frombuffer(pose_raw[3:5], dtype=np.uint16, count=1)[0]
            vals = [int(first)]
            cursor = 5
            while cursor < len(pose_raw):
                shift = 0
                acc = 0
                while True:
                    byte = pose_raw[cursor]
                    cursor += 1
                    acc |= (byte & 0x7F) << shift
                    if byte < 0x80:
                        break
                    shift += 7
                delta = (acc >> 1) ^ -(acc & 1)
                vals.append(vals[-1] + delta)
            q_pose = np.zeros((len(vals), 6), dtype=np.uint16)
            q_pose[:, 0] = np.asarray(vals, dtype=np.uint16)
        else:
            q_pose = np.frombuffer(pose_raw, dtype=np.uint16).reshape(-1, 6)
        pose_np = np.empty(q_pose.shape, dtype=np.float32)
        pose_np[:, 0] = q_pose[:, 0].astype(np.float32) / 512.0 + 20.0
        pose_np[:, 1:] = q_pose[:, 1:].view(np.int16).astype(np.float32) / 2048.0
    else:
        with open(pose_br, "rb") as f:
            pose_bytes = brotli.decompress(f.read())
        pose_payload = np.load(io.BytesIO(pose_bytes))
        if isinstance(pose_payload, np.lib.npyio.NpzFile):
            q_pose = pose_payload["q"].astype(np.float32)
            pose_min = pose_payload["min"].astype(np.float32)
            pose_scale = pose_payload["scale"].astype(np.float32)
            pose_np = q_pose * pose_scale + pose_min
            pose_payload.close()
        else:
            pose_np = pose_payload
    pose_frames_all = torch.from_numpy(pose_np).float()
    smooth_pose = load_smooth_pose(smooth_pose_br)
    if smooth_pose is not None:
        basis = make_smooth_pose_basis(pose_np.shape[0], smooth_pose["basis_kind"])
        corr = basis @ smooth_pose["coeff"].astype(np.float32)
        if smooth_pose["scale"] is not None:
            corr = corr * smooth_pose["scale"].reshape(1, -1)
        pose_frames_all = torch.from_numpy(pose_np + corr.astype(np.float32)).float()
    actuator = load_actuator(actuator_br, device)
    if embedded_seg_tile_actions is not None:
        seg_tile_actions = load_seg_tile_actions_data(embedded_seg_tile_actions, device)
    else:
        seg_tile_actions = load_seg_tile_actions(seg_tile_actions_br, device)

    color_bias = None
    color_scale = None
    if color_lut_br.exists():
        with open(color_lut_br, "rb") as f:
            lut_payload = np.load(io.BytesIO(brotli.decompress(f.read())))
        if isinstance(lut_payload, np.lib.npyio.NpzFile):
            color_bias = torch.from_numpy(lut_payload["bias"]).to(device=device, dtype=torch.float32)
            color_scale = torch.from_numpy(lut_payload["scale"]).to(device=device, dtype=torch.float32)
            lut_payload.close()
        else:
            color_bias = torch.from_numpy(lut_payload).to(device=device, dtype=torch.float32)
            color_scale = torch.zeros_like(color_bias)

    out_h, out_w = 874, 1164
    cursor = 0
    batch_size = 4 
    
    # 1 mask per generated pair, assume 600 pairs per standard 1200 frame chunk.
    pairs_per_file = 600

    with torch.inference_mode():
        for file_name in files:
            base_name = os.path.splitext(file_name)[0]
            raw_out_path = out_dir / f"{base_name}.raw"
            
            # Retrieve exactly the pairs mapping to this file
            file_masks = mask_frames_all[cursor : cursor + pairs_per_file]
            file_poses = pose_frames_all[cursor : cursor + pairs_per_file]
            file_pair_start = cursor
            cursor += pairs_per_file
            
            with open(raw_out_path, "wb") as f_out:
                pbar = tqdm(range(0, file_masks.shape[0], batch_size), desc=f"Decoding {file_name}")
                
                for i in pbar:
                    in_mask2 = file_masks[i : i + batch_size].to(device).long()
                    in_pose6 = file_poses[i : i + batch_size].to(device).float()

                    fake1, fake2 = generator(in_mask2, in_pose6)
                    if color_bias is not None:
                        bias = F.embedding(in_mask2, color_bias).permute(0, 3, 1, 2)
                        scale = F.embedding(in_mask2, color_scale).permute(0, 3, 1, 2)
                        fake1 = fake1 * (1.0 + scale) + bias
                        fake2 = fake2 * (1.0 + scale) + bias
                    if actuator is not None:
                        alpha = actuator["alpha"][file_pair_start + i : file_pair_start + i + in_mask2.shape[0]]
                        delta = torch.einsum("bk,kchw->bchw", alpha, actuator["basis"])
                        if actuator["base"] == "frame2":
                            fake1 = fake2 + delta
                        else:
                            fake1 = fake1 + delta
                    if seg_tile_actions is not None:
                        tile_size = 32
                        grid_w = 512 // tile_size
                        for batch_j in range(in_mask2.shape[0]):
                            frame_idx = file_pair_start + i + batch_j
                            for tile_id, action_id in seg_tile_actions["by_frame"].get(frame_idx, []):
                                y0 = (tile_id // grid_w) * tile_size
                                x0 = (tile_id % grid_w) * tile_size
                                fake2[batch_j, :, y0:y0 + tile_size, x0:x0 + tile_size] = (
                                    fake2[batch_j, :, y0:y0 + tile_size, x0:x0 + tile_size]
                                    + seg_tile_actions["deltas"][action_id]
                                ).clamp(0, 255)

                    fake1_up = F.interpolate(fake1, size=(out_h, out_w), mode="bilinear", align_corners=False)
                    fake2_up = F.interpolate(fake2, size=(out_h, out_w), mode="bilinear", align_corners=False)

                    batch_comp = torch.stack([fake1_up, fake2_up], dim=1)
                    batch_comp = einops.rearrange(batch_comp, "b t c h w -> (b t) h w c")

                    output_bytes = batch_comp.clamp(0, 255).round().to(torch.uint8)
                    f_out.write(output_bytes.cpu().numpy().tobytes())

if __name__ == "__main__":
    main()
