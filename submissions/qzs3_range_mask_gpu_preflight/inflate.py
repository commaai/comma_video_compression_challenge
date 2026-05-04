#!/usr/bin/env python
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import brotli
except ImportError:
    class _BrotliCompat:
        @staticmethod
        def decompress(*args, **kwargs):
            raise RuntimeError(
                "Brotli decompression was requested, but the optional 'brotli' package "
                "is not installed in this environment."
            )

    brotli = _BrotliCompat()
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


RANGE_MASK_BYTES = 159011
SPLIT_MODEL_PACKED_REORDERED_BR_BYTES = 37086
SPLIT_MODEL_SCALES_REORDERED_BR_BYTES = 3035
SPLIT_MODEL_TAIL_REORDERED_BR_BYTES = 15604
SPLIT_MODEL_REORDERED_BYTES = (
    SPLIT_MODEL_PACKED_REORDERED_BR_BYTES
    + SPLIT_MODEL_SCALES_REORDERED_BR_BYTES
    + SPLIT_MODEL_TAIL_REORDERED_BR_BYTES
)
POSE_STREAM_BYTES = 899
ROUTER_ACTION_BYTES = 225
ROUTER_ACTION_COUNT = 600
ROUTER_ACTION_BITS = 3
PACKED_PAYLOAD_BYTES = RANGE_MASK_BYTES + SPLIT_MODEL_REORDERED_BYTES + POSE_STREAM_BYTES + ROUTER_ACTION_BYTES
OUT_H, OUT_W = 874, 1164
PAIRS_PER_FILE = 600
INFLATE_BATCH_SIZE = 4


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

def tensor_from_payload(data: memoryview, dtype: torch.dtype) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=dtype)

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
            packed = tensor_from_payload(take_from("packed", packed_count), torch.uint8).to(device)
            scales = tensor_from_payload(take_from("scales", scale_count * 2), torch.float16).to(device).float()
            nibbles = unpack_nibbles(packed, packed.numel() * 2)
            state_dict[f"{name}.weight"] = FP4Codebook.dequantize_from_nibbles(nibbles, scales, weight_shape).float()
        elif kind == "fp":
            _, _, weight_shape, count = spec
            state_dict[f"{name}.weight"] = tensor_from_payload(take_from("fp_weight", count * 2), torch.float16).reshape(weight_shape).to(device).float()
        else:
            _, _, bias_shape, count = spec
            state_dict[f"{name}.bias"] = tensor_from_payload(take_from("bias", count * 2), torch.float16).reshape(bias_shape).to(device).float()

    for key, kind, shape, count, dtype, bits, rows in dense_specs:
        if kind == "dense_fp":
            state_dict[key] = tensor_from_payload(take_from("dense_fp", count * 2), torch.float16).reshape(shape).to(device).float()
        elif kind == "dense_other":
            state_dict[key] = tensor_from_payload(take_from("dense_other", count * torch.empty((), dtype=dtype).element_size()), dtype).reshape(shape).to(device)
        else:
            meta = take_from("qv", rows * 4)
            mn_step = tensor_from_payload(meta, torch.float16).reshape(rows, 2).float()
            packed_count = (count * bits + 7) // 8
            q = unpack_qbits(take_from("qv", packed_count), count, bits).reshape(rows, -1)
            value = mn_step[:, :1] + q * mn_step[:, 1:].clamp_min(1e-8)
            state_dict[key] = value.reshape(shape).to(device).float()
    return state_dict

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

def load_range_mask(mask_payload: bytes) -> torch.Tensor:
    codec_src = Path(__file__).with_name("range_mask_codec.cpp")
    if not codec_src.exists():
        raise RuntimeError(f"missing range-mask decoder source: {codec_src}")
    packed_mask = mask_payload
    if len(packed_mask) < 20:
        raise RuntimeError("truncated range-mask payload")
    t_count = int.from_bytes(packed_mask[4:8], "little")
    mask_h = int.from_bytes(packed_mask[8:12], "little")
    mask_w = int.from_bytes(packed_mask[12:16], "little")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        exe = td_path / "range_mask_codec"
        compiler_candidates = [os.environ.get("CXX", ""), "c++", "g++", "clang++"]
        last_error = None
        for compiler in dict.fromkeys(c for c in compiler_candidates if c):
            compiler_path = shutil.which(compiler)
            if compiler_path is None:
                continue
            try:
                subprocess.run([compiler_path, "-O3", "-std=c++17", str(codec_src), "-o", str(exe)], check=True)
                break
            except subprocess.CalledProcessError as exc:
                last_error = exc
        else:
            raise RuntimeError("failed to compile range-mask decoder with c++/g++/clang++") from last_error
        packed = td_path / "mask.range"
        raw = td_path / "mask.raw"
        packed.write_bytes(packed_mask)
        subprocess.run([str(exe), "decode", str(packed), str(raw)], check=True)
        decoded = np.frombuffer(raw.read_bytes(), dtype=np.uint8)
        if (t_count, mask_h, mask_w) == (600, 512, 384):
            arr = decoded.reshape(600, 512, 384).transpose(0, 2, 1).copy()
        elif (t_count, mask_h, mask_w) == (600, 384, 512):
            arr = decoded.reshape(600, 384, 512).copy()
        else:
            raise RuntimeError(f"unexpected range-mask dimensions: {(t_count, mask_h, mask_w)}")
    return torch.from_numpy(arr).contiguous()

def get_qzs3_split_specs(block_size: int = 32):
    template = JointFrameGenerator()
    qv_specs = get_qv_specs()
    packed_chunks = []
    scales_chunks = []
    tail_chunks = {
        "bias": [],
        "dense_fp": [],
        "fp_weight": [],
        "dense_other": [],
        "qv": [],
    }
    covered_keys = set()

    for name, module in template.named_modules():
        if not isinstance(module, (QConv2d, QEmbedding)):
            continue
        covered_keys.add(f"{name}.weight")
        if getattr(module, "quantize_weight", False):
            weight_numel = int(module.weight.numel())
            scale_count = (weight_numel + block_size - 1) // block_size
            packed_count = (scale_count * block_size + 1) // 2
            packed_chunks.append((name, packed_count))
            scales_chunks.append((name, scale_count * 2))
        else:
            tail_chunks["fp_weight"].append((f"{name}.weight", int(module.weight.numel()) * 2))
        if isinstance(module, QConv2d) and module.bias is not None:
            covered_keys.add(f"{name}.bias")
            tail_chunks["bias"].append((f"{name}.bias", int(module.bias.numel()) * 2))

    for key, tensor in template.state_dict().items():
        if key in covered_keys:
            continue
        shape = tuple(tensor.shape)
        count = int(tensor.numel())
        if key in qv_specs:
            bits, per_row = qv_specs[key]
            rows = shape[0] if per_row and len(shape) >= 2 else 1
            tail_chunks["qv"].append((key, rows * 4 + (count * bits + 7) // 8))
        elif torch.is_floating_point(tensor):
            tail_chunks["dense_fp"].append((key, count * 2))
        else:
            tail_chunks["dense_other"].append((key, count * tensor.element_size()))

    return packed_chunks, scales_chunks, tail_chunks

def split_chunks(data: bytes, chunks):
    offset = 0
    out = {}
    for name, count in chunks:
        out[name] = data[offset : offset + count]
        offset += count
    if offset != len(data):
        raise RuntimeError(f"split chunk length mismatch: consumed {offset}, got {len(data)}")
    return out

def restore_chunk_order(data: bytes, raw_chunks, stored_chunks) -> bytes:
    pieces = split_chunks(data, stored_chunks)
    return b"".join(pieces[name] for name, _ in raw_chunks)

def load_reordered_split_model_payload(model_payload: bytes) -> bytes:
    if len(model_payload) != SPLIT_MODEL_REORDERED_BYTES:
        raise RuntimeError(f"unexpected reordered split model payload length: {len(model_payload)}")

    offset = 0
    packed_br = model_payload[offset : offset + SPLIT_MODEL_PACKED_REORDERED_BR_BYTES]
    offset += SPLIT_MODEL_PACKED_REORDERED_BR_BYTES
    scales_br = model_payload[offset : offset + SPLIT_MODEL_SCALES_REORDERED_BR_BYTES]
    offset += SPLIT_MODEL_SCALES_REORDERED_BR_BYTES
    tail_br = model_payload[offset : offset + SPLIT_MODEL_TAIL_REORDERED_BR_BYTES]

    packed_chunks, scales_chunks, tail_chunks = get_qzs3_split_specs()

    packed = restore_chunk_order(
        brotli.decompress(packed_br),
        packed_chunks,
        sorted(packed_chunks, key=lambda item: (item[1], item[0])),
    )
    scales = restore_chunk_order(
        brotli.decompress(scales_br),
        scales_chunks,
        sorted(scales_chunks, key=lambda item: (-item[1], item[0])),
    )

    tail_raw_order = ("bias", "dense_fp", "fp_weight", "dense_other", "qv")
    tail_stored_order = ("qv", "dense_fp", "fp_weight", "bias")
    tail_stored_chunks = {
        "qv": sorted(tail_chunks["qv"], key=lambda item: item[0], reverse=True),
        "dense_fp": sorted(tail_chunks["dense_fp"], key=lambda item: (item[1], item[0])),
        "fp_weight": list(reversed(tail_chunks["fp_weight"])),
        "bias": sorted(tail_chunks["bias"], key=lambda item: (-item[1], item[0])),
    }
    tail_data = brotli.decompress(tail_br)
    tail_offset = 0
    tail_by_type = {}
    for key in tail_stored_order:
        byte_count = sum(size for _, size in tail_stored_chunks[key])
        type_data = tail_data[tail_offset : tail_offset + byte_count]
        tail_offset += byte_count
        tail_by_type[key] = restore_chunk_order(type_data, tail_chunks[key], tail_stored_chunks[key])
    if tail_offset != len(tail_data):
        raise RuntimeError(f"tail length mismatch: consumed {tail_offset}, got {len(tail_data)}")
    tail_by_type["dense_other"] = b""
    tail = b"".join(tail_by_type[key] for key in tail_raw_order)

    return b"QZS3" + (32).to_bytes(2, "little") + packed + scales + tail

def unpack_router_actions(action_payload: bytes, count: int = ROUTER_ACTION_COUNT) -> torch.Tensor:
    if len(action_payload) != ROUTER_ACTION_BYTES:
        raise RuntimeError(f"unexpected router action payload length: {len(action_payload)}")
    vals = []
    acc = 0
    bits = 0
    for byte in action_payload:
        acc |= int(byte) << bits
        bits += 8
        while bits >= ROUTER_ACTION_BITS and len(vals) < count:
            vals.append(acc & ((1 << ROUTER_ACTION_BITS) - 1))
            acc >>= ROUTER_ACTION_BITS
            bits -= ROUTER_ACTION_BITS
    if len(vals) != count:
        raise RuntimeError(f"decoded {len(vals)} router actions, expected {count}")
    return torch.tensor(vals, dtype=torch.uint8)

def apply_router_actions(batch_comp: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if actions.numel() == 0:
        return batch_comp
    out = batch_comp.clamp(0, 255).round()
    actions = actions.to(device=out.device, dtype=torch.long)

    def select(action_id: int) -> torch.Tensor:
        return actions == action_id

    mask = select(1)
    if mask.any():
        out[mask, 1, 2:3] = (out[mask, 1, 2:3] - 3.0).clamp(0, 255).round()
    mask = select(2)
    if mask.any():
        out[mask, :, 1:2] = (out[mask, :, 1:2] - 3.0).clamp(0, 255).round()
    mask = select(3)
    if mask.any():
        out[mask] = (out[mask] - 2.0).clamp(0, 255).round()
    mask = select(4)
    if mask.any():
        out[mask, 1:2] = ((out[mask, 1:2] - 128.0) * 1.03 + 128.0).clamp(0, 255).round()
    mask = select(5)
    if mask.any():
        out[mask, 1, 0:1] = (out[mask, 1, 0:1] + 3.0).clamp(0, 255).round()
    mask = select(6)
    if mask.any():
        out[mask, 1, 1:2] = (out[mask, 1, 1:2] - 4.0).clamp(0, 255).round()
    mask = select(7)
    if mask.any():
        out[mask] = torch.pow((out[mask] / 255.0).clamp(0.0, 1.0), 1.04).mul(255.0).clamp(0, 255).round()
    return out


def split_submission_payload(data_dir: Path) -> tuple[bytes, bytes, bytes, bytes]:
    payload_path = data_dir / "p"
    if not payload_path.exists():
        raise FileNotFoundError(f"missing packed payload: {payload_path}")
    payload = payload_path.read_bytes()
    if len(payload) != PACKED_PAYLOAD_BYTES:
        raise RuntimeError(f"unexpected packed payload length: {len(payload)}")
    mask_end = RANGE_MASK_BYTES
    model_end = mask_end + SPLIT_MODEL_REORDERED_BYTES
    pose_end = model_end + POSE_STREAM_BYTES
    return payload[:mask_end], payload[mask_end:model_end], payload[model_end:pose_end], payload[pose_end:]


def decode_pose_payload(pose_payload: bytes) -> torch.Tensor:
    pose_raw = brotli.decompress(pose_payload)
    if not pose_raw.startswith(b"QP1"):
        raise RuntimeError("unexpected pose stream")
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
        vals.append(vals[-1] + ((acc >> 1) ^ -(acc & 1)))
    q_pose = np.zeros((len(vals), 6), dtype=np.uint16)
    q_pose[:, 0] = np.asarray(vals, dtype=np.uint16)
    pose_np = np.empty(q_pose.shape, dtype=np.float32)
    pose_np[:, 0] = q_pose[:, 0].astype(np.float32) / 512.0 + 20.0
    pose_np[:, 1:] = q_pose[:, 1:].view(np.int16).astype(np.float32) / 2048.0
    return torch.from_numpy(pose_np).float()


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

    mask_br_data, model_br_data, pose_q_br_data, router_actions_data = split_submission_payload(data_dir)
    generator = JointFrameGenerator().to(device)
    weights_data = load_reordered_split_model_payload(model_br_data)
    
    generator.load_state_dict(get_grouped_qv_state_dict(weights_data, device), strict=True)
    generator.eval()

    mask_frames_all = load_range_mask(mask_br_data)
    pose_frames_all = decode_pose_payload(pose_q_br_data)
    router_actions_all = unpack_router_actions(router_actions_data)
    cursor = 0

    with torch.inference_mode():
        for file_name in files:
            base_name = os.path.splitext(file_name)[0]
            raw_out_path = out_dir / f"{base_name}.raw"
            
            file_masks = mask_frames_all[cursor : cursor + PAIRS_PER_FILE]
            file_poses = pose_frames_all[cursor : cursor + PAIRS_PER_FILE]
            file_pair_start = cursor
            cursor += PAIRS_PER_FILE
            
            with open(raw_out_path, "wb") as f_out:
                pbar = tqdm(range(0, file_masks.shape[0], INFLATE_BATCH_SIZE), desc=f"Decoding {file_name}")
                
                for i in pbar:
                    in_mask2 = file_masks[i : i + INFLATE_BATCH_SIZE].to(device).long()
                    in_pose6 = file_poses[i : i + INFLATE_BATCH_SIZE].to(device).float()

                    fake1, fake2 = generator(in_mask2, in_pose6)
                    fake1_up = F.interpolate(fake1, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
                    fake2_up = F.interpolate(fake2, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)

                    batch_comp = torch.stack([fake1_up, fake2_up], dim=1)
                    if router_actions_all is not None:
                        actions = router_actions_all[file_pair_start + i : file_pair_start + i + batch_comp.shape[0]]
                        batch_comp = apply_router_actions(batch_comp, actions)
                    batch_comp = einops.rearrange(batch_comp, "b t c h w -> (b t) h w c")

                    output_bytes = batch_comp.clamp(0, 255).round().to(torch.uint8)
                    f_out.write(output_bytes.cpu().numpy().tobytes())

if __name__ == "__main__":
    main()
