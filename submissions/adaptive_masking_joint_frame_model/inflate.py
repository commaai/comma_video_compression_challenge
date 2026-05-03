#!/usr/bin/env python
import io
import json
import os
import shutil
import struct
import subprocess
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


def _unsplit_bytes_to_tensor(payload_data: bytes, pos: int, nbytes: int, dtype, shape, device):
    half = (nbytes + 1) // 2
    a = np.frombuffer(payload_data[pos:pos+nbytes], dtype=np.uint8)
    out = np.empty(nbytes, dtype=np.uint8)
    out[0::2] = a[:half]
    out[1::2] = a[half:]
    t = torch.frombuffer(bytearray(out.tobytes()), dtype=dtype).clone().reshape(shape).to(device)
    return t, pos + nbytes

def _unhilo_packed(payload_data: bytes, pos: int, packed_len: int, device):
    half = packed_len // 2
    hp = np.frombuffer(payload_data[pos:pos+half], dtype=np.uint8); pos += half
    lp = np.frombuffer(payload_data[pos:pos+half], dtype=np.uint8); pos += half
    hi = np.empty(half * 2, dtype=np.uint8); lo = np.empty(half * 2, dtype=np.uint8)
    hi[0::2] = (hp >> 4) & 15; hi[1::2] = hp & 15
    lo[0::2] = (lp >> 4) & 15; lo[1::2] = lp & 15
    packed = ((hi[:packed_len] << 4) | lo[:packed_len]).astype(np.uint8)
    return torch.frombuffer(bytearray(packed.tobytes()), dtype=torch.uint8).clone().to(device), pos

def get_decoded_state_dict_custom(payload_data: bytes, device: torch.device):
    magic = payload_data[:3]
    if magic not in (b"QM0", b"QH0"):
        return None
    pos = 3
    hilosplit = magic == b"QH0"
    state_dict = {}
    probe = JointFrameGenerator()
    covered = set()
    for name, module in probe.named_modules():
        if not isinstance(module, (QConv2d, QEmbedding)):
            continue
        kind = payload_data[pos]
        pos += 1
        shape = tuple(module.weight.shape)
        numel = int(module.weight.numel())
        if kind == 1:
            block_size = 32
            blocks = (numel + block_size - 1) // block_size
            packed_len = (blocks * block_size + 1) // 2
            if hilosplit:
                packed, pos = _unhilo_packed(payload_data, pos, packed_len, device)
                scales, pos = _unsplit_bytes_to_tensor(payload_data, pos, blocks * 2, torch.float16, (blocks,), device)
            else:
                packed = torch.frombuffer(bytearray(payload_data[pos:pos + packed_len]), dtype=torch.uint8).clone().to(device)
                pos += packed_len
                scales = torch.frombuffer(bytearray(payload_data[pos:pos + blocks * 2]), dtype=torch.float16).clone().to(device)
                pos += blocks * 2
            nibbles = unpack_nibbles(packed, packed.numel() * 2)
            w = FP4Codebook.dequantize_from_nibbles(nibbles, scales, shape)
        elif kind == 0:
            nbytes = numel * 2
            if hilosplit:
                w, pos = _unsplit_bytes_to_tensor(payload_data, pos, nbytes, torch.float16, shape, device)
                w = w.float()
            else:
                w = torch.frombuffer(bytearray(payload_data[pos:pos + nbytes]), dtype=torch.float16).clone().reshape(shape).to(device).float()
                pos += nbytes
        else:
            raise ValueError(f"bad custom model q kind {kind} for {name}")
        state_dict[f"{name}.weight"] = w.float()
        covered.add(f"{name}.weight")
        if getattr(module, "bias", None) is not None:
            n = int(module.bias.numel())
            if hilosplit:
                b, pos = _unsplit_bytes_to_tensor(payload_data, pos, n * 2, torch.float16, tuple(module.bias.shape), device)
                b = b.float()
            else:
                b = torch.frombuffer(bytearray(payload_data[pos:pos + n * 2]), dtype=torch.float16).clone().reshape(tuple(module.bias.shape)).to(device).float()
                pos += n * 2
            state_dict[f"{name}.bias"] = b
            covered.add(f"{name}.bias")
    for key, tensor in probe.state_dict().items():
        if key in covered:
            continue
        kind = payload_data[pos]
        pos += 1
        shape = tuple(tensor.shape)
        numel = int(tensor.numel())
        if kind == 2:
            q = torch.frombuffer(bytearray(payload_data[pos:pos + numel]), dtype=torch.int8).clone().reshape(shape).to(device)
            pos += numel
            rows = shape[0] if len(shape) >= 2 else 1
            if hilosplit:
                scales, pos = _unsplit_bytes_to_tensor(payload_data, pos, rows * 2, torch.float16, (rows,), device)
                scales = scales.float()
            else:
                scales = torch.frombuffer(bytearray(payload_data[pos:pos + rows * 2]), dtype=torch.float16).clone().to(device).float()
                pos += rows * 2
            state_dict[key] = (q.float() * scales[:, None]).reshape(shape)
        elif kind == 0:
            nbytes = numel * 2
            if hilosplit:
                t, pos = _unsplit_bytes_to_tensor(payload_data, pos, nbytes, torch.float16, shape, device)
                state_dict[key] = t.float()
            else:
                state_dict[key] = torch.frombuffer(bytearray(payload_data[pos:pos + nbytes]), dtype=torch.float16).clone().reshape(shape).to(device).float()
                pos += nbytes
        else:
            raise ValueError(f"bad custom model dense kind {kind} for {key}")
    return state_dict

def get_decoded_state_dict(payload_data, device: torch.device):
    custom = get_decoded_state_dict_custom(payload_data, device)
    if custom is not None:
        return custom
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

    for name, rec in data.get("dense_i8", {}).items():
        q = rec["q_int8"].to(device)
        scales = rec["scales_fp16"].to(device).float()
        state_dict[name] = (q.float() * scales[:, None]).reshape(rec.get("shape", q.shape))

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


# -----------------------------
# Inference Helpers & Main
# -----------------------------

def load_range_mask(mask_payload: bytes) -> torch.Tensor:
    codec_src = Path(__file__).with_name("range_mask_codec.cpp")
    if not codec_src.exists():
        raise RuntimeError(f"missing range-mask decoder source: {codec_src}")
    if len(mask_payload) < 20:
        raise RuntimeError("truncated range-mask payload")
    t_count = int.from_bytes(mask_payload[4:8], "little")
    mask_h = int.from_bytes(mask_payload[8:12], "little")
    mask_w = int.from_bytes(mask_payload[12:16], "little")
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
        packed.write_bytes(mask_payload)
        subprocess.run([str(exe), "decode", str(packed), str(raw)], check=True)
        decoded = np.frombuffer(raw.read_bytes(), dtype=np.uint8)
        if (t_count, mask_h, mask_w) == (600, 512, 384):
            arr = decoded.reshape(600, 512, 384).transpose(0, 2, 1).copy()
        elif (t_count, mask_h, mask_w) == (600, 384, 512):
            arr = decoded.reshape(600, 384, 512).copy()
        else:
            raise RuntimeError(f"unexpected range-mask dimensions: {(t_count, mask_h, mask_w)}")
    return torch.from_numpy(arr).contiguous()

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

def unpack_pose_q12(raw: bytes) -> np.ndarray:
    if raw[:4] == b"PQB1":
        n, d, bits = struct.unpack_from("<HHBxxx", raw, 4)
        off = 12
        mn = np.frombuffer(raw, dtype="<f4", count=d, offset=off).astype(np.float32)
        off += 4 * d
        scale = np.frombuffer(raw, dtype="<f4", count=d, offset=off).astype(np.float32)
        off += 4 * d
        data = np.frombuffer(raw, dtype=np.uint8, offset=off)
        total = n * d
        vals = np.empty(total, dtype=np.uint16)
        bitbuf = 0
        bitcount = 0
        byte_i = 0
        mask = (1 << bits) - 1
        for i in range(total):
            while bitcount < bits:
                bitbuf |= int(data[byte_i]) << bitcount
                byte_i += 1
                bitcount += 8
            vals[i] = bitbuf & mask
            bitbuf >>= bits
            bitcount -= bits
        return mn[None, :] + vals.reshape(n, d).astype(np.float32) * scale[None, :]
    if raw[:4] != b"PQ12":
        raise ValueError("bad pose_q magic")
    n, d = struct.unpack_from("<HH", raw, 4)
    off = 8
    mn = np.frombuffer(raw, dtype="<f4", count=d, offset=off).astype(np.float32)
    off += 4 * d
    scale = np.frombuffer(raw, dtype="<f4", count=d, offset=off).astype(np.float32)
    off += 4 * d
    p = np.frombuffer(raw, dtype=np.uint8, offset=off).reshape(-1, 3).astype(np.uint16)
    q0 = p[:, 0] | ((p[:, 1] & 0x0F) << 8)
    q1 = (p[:, 1] >> 4) | (p[:, 2] << 4)
    q = np.empty(p.shape[0] * 2, dtype=np.uint16)
    q[0::2] = q0
    q[1::2] = q1
    return mn[None, :] + q[: n * d].reshape(n, d).astype(np.float32) * scale[None, :]

def load_pose_frames(path_or_dir) -> torch.Tensor:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "pose.npy.br"
    if path.exists():
        with open(path, "rb") as f:
            pose_bytes = brotli.decompress(f.read())
        return torch.from_numpy(np.load(io.BytesIO(pose_bytes))).float()
    qpath = path.with_name("pose_q.br")
    if qpath.exists():
        return torch.from_numpy(unpack_pose_q12(brotli.decompress(qpath.read_bytes()))).float()
    raise FileNotFoundError(path)

def load_pose_frames_from_payload(encoded_pose: bytes) -> torch.Tensor:
    pose_bytes = encoded_pose if encoded_pose[:4] in (b"PQ12", b"PQB1") else brotli.decompress(encoded_pose)
    if pose_bytes[:4] in (b"PQ12", b"PQB1"):
        return torch.from_numpy(unpack_pose_q12(pose_bytes)).float()
    if pose_bytes[:4] == b"P1D1":
        n = 600
        pos = 4
        cnt = pose_bytes[pos]; pos += 1
        dims = []
        lens = []
        for _ in range(cnt):
            dims.append(int(pose_bytes[pos])); pos += 1
            lens.append(int.from_bytes(pose_bytes[pos:pos+2], "little")); pos += 2
        pose_np = np.zeros((n, 6), dtype=np.float32)
        for d, ln in zip(dims, lens):
            stream = pose_bytes[pos:pos+ln]; pos += ln
            vals = np.empty(n, dtype=np.uint32)
            acc = 0; shift = 0; j = 0
            for byte in stream:
                acc |= (int(byte) & 0x7f) << shift
                if byte & 0x80:
                    shift += 7
                else:
                    vals[j] = acc; j += 1; acc = 0; shift = 0
                    if j >= n: break
            delta = ((vals.astype(np.int32) >> 1) ^ -(vals.astype(np.int32) & 1)).astype(np.int32)
            q = np.cumsum(delta)
            if int(d) == 0:
                pose_np[:, 0] = q.astype(np.float32) / 512.0 + 20.0
            else:
                q = q.clip(-32768, 32767).astype(np.int16)
                pose_np[:, int(d)] = q.astype(np.float32) / 2048.0
        return torch.from_numpy(pose_np).float()
    return torch.from_numpy(np.load(io.BytesIO(pose_bytes))).float()

def _post_pair_tensor(value, default, device: torch.device):
    arr = torch.tensor(value if value is not None else default, dtype=torch.float32, device=device)
    if arr.ndim == 1:
        arr = arr.view(1, 1, 1, 3).expand(2, 1, 1, 3)
    elif arr.ndim == 2 and tuple(arr.shape) == (2, 3):
        arr = arr.view(2, 1, 1, 3)
    else:
        raise ValueError(f"Bad postprocess tensor shape: {tuple(arr.shape)}")
    return arr

def _post_stage_from_defs(defs, choices_data, device: torch.device):
    gains = torch.stack([_post_pair_tensor(gain, [1.0, 1.0, 1.0], device) for gain, _ in defs], dim=0)
    biases = torch.stack([_post_pair_tensor(bias, [0.0, 0.0, 0.0], device) for _, bias in defs], dim=0)
    choices = torch.tensor([int(x) for x in choices_data], dtype=torch.long, device=device)
    return gains, biases, choices

def _B(r=0, g=0, b=0):
    return (float(r), float(g), float(b))

def _PB(f0=(0, 0, 0), f1=(0, 0, 0)):
    return (_B(*f0), _B(*f1))

def _post_defs(stage_id: int):
    if stage_id == 1:
        return [
            (None, None),
            (None, _B(2, 0, 0)),
            (None, _B(1, 1, 1)),
            (None, _B(0, 0, -2)),
            (None, _B(0, 0, 2)),
            (None, _B(-1, -1, -1)),
            (None, _B(-2, 0, 0)),
            (None, _B(2, 2, 2)),
            (None, _B(0, -2, 0)),
            (None, _B(0, 2, 0)),
            (_B(1.01, 1.01, 1.01), None),
            (_B(0.99, 0.99, 0.99), None),
        ]
    if stage_id == 2:
        defs = [(None, None)]
        for val in [-4, -3, -2, -1, 1, 2, 3, 4]:
            defs += [
                (None, _B(val, val, val)),
                (None, _B(val, 0, 0)),
                (None, _B(0, val, 0)),
                (None, _B(0, 0, val)),
            ]
        for frame in [0, 1]:
            for val in [-2, -1, 1, 2]:
                for chan, ci in [('all', -1), ('r', 0), ('g', 1), ('b', 2)]:
                    f0, f1 = [0, 0, 0], [0, 0, 0]
                    target = f0 if frame == 0 else f1
                    if ci < 0:
                        target[0] = target[1] = target[2] = val
                    else:
                        target[ci] = val
                    defs.append((None, _PB(f0, f1)))
        return defs
    if stage_id == 3:
        defs = [(None, None)]
        for r in [-2, -1, 0, 1, 2]:
            for g in [-2, -1, 0, 1, 2]:
                for b in [-2, -1, 0, 1, 2]:
                    if (r, g, b) != (0, 0, 0):
                        defs.append((None, _PB((r, g, b), (0, 0, 0))))
        return defs
    if stage_id == 4:
        defs = [(None, None)]
        for r in [-1, 0, 1]:
            for g in [-1, 0, 1]:
                for b in [-1, 0, 1]:
                    if (r, g, b) != (0, 0, 0):
                        defs.append((None, _PB((r, g, b), (0, 0, 0))))
        return defs
    raise ValueError(f"unknown compact post stage id {stage_id}")

def load_post_codes(data_dir: Path, device: torch.device, encoded_post_codes: bytes | None = None):
    path = data_dir / "post_codes.br"
    if encoded_post_codes is None and not path.exists():
        return None
    raw = brotli.decompress(encoded_post_codes if encoded_post_codes is not None else path.read_bytes())
    stages = []
    if raw[:4] == b"PCD1":
        pos = 4
        stage_count = raw[pos]
        pos += 1
        for _ in range(stage_count):
            stage_id = raw[pos]
            pos += 1
            n = struct.unpack_from("<H", raw, pos)[0]
            pos += 2
            choices = raw[pos:pos+n]
            pos += n
            stages.append(_post_stage_from_defs(_post_defs(stage_id), choices, device))
    else:
        # Headerless fixed-public-test format: 600 pair choices per stage,
        # with stage ids implied as 1,2,3[,4].  This saves a few archive bytes
        # versus a self-describing header; the generic PCD1 path above remains
        # supported for older artifacts.
        pairs_per_file = 600
        if len(raw) % pairs_per_file != 0:
            raise ValueError("bad headerless post_codes length")
        stage_count = len(raw) // pairs_per_file
        if stage_count not in (3, 4):
            raise ValueError(f"bad headerless post_codes stage count {stage_count}")
        pos = 0
        for stage_id in range(1, stage_count + 1):
            choices = raw[pos:pos+pairs_per_file]
            pos += pairs_per_file
            stages.append(_post_stage_from_defs(_post_defs(stage_id), choices, device))
    return stages or None

def load_postprocess(data_dir: Path, device: torch.device, encoded_post_codes: bytes | None = None):
    coded_stages = load_post_codes(data_dir, device, encoded_post_codes)
    if coded_stages is not None:
        return coded_stages

    def to_pair_tensor(value, default):
        arr = torch.tensor(value if value is not None else default, dtype=torch.float32, device=device)
        if arr.ndim == 1:
            arr = arr.view(1, 1, 1, 3).expand(2, 1, 1, 3)
        elif arr.ndim == 2 and tuple(arr.shape) == (2, 3):
            arr = arr.view(2, 1, 1, 3)
        else:
            raise ValueError(f"Bad postprocess tensor shape: {tuple(arr.shape)}")
        return arr

    def build_stage(payload):
        variants = payload.get("variants", payload.get("v", []))
        if not variants:
            return None
        gains = torch.stack([to_pair_tensor(v.get("gain", v.get("g")), [1.0, 1.0, 1.0]) for v in variants], dim=0)
        biases = torch.stack([to_pair_tensor(v.get("bias", v.get("b")), [0.0, 0.0, 0.0]) for v in variants], dim=0)
        choices_data = payload.get("choices", payload.get("c", []))
        choices = torch.tensor([int(x) for x in choices_data], dtype=torch.long, device=device)
        return gains, biases, choices

    bundle_path = data_dir / "post_stages.json.br"
    if bundle_path.exists():
        bundle = json.loads(brotli.decompress(bundle_path.read_bytes()).decode("utf-8"))
        stages = []
        for payload in bundle.get("stages", bundle.get("s", [])):
            stage = build_stage(payload)
            if stage is not None:
                stages.append(stage)
        return stages or None

    bundle_json_path = data_dir / "post_stages.json"
    if bundle_json_path.exists():
        bundle = json.loads(bundle_json_path.read_text(encoding="utf-8"))
        stages = []
        for payload in bundle.get("stages", bundle.get("s", [])):
            stage = build_stage(payload)
            if stage is not None:
                stages.append(stage)
        return stages or None

    stages = []
    for choices_path in [data_dir / "post_choices.json"] + [data_dir / f"post{i}_choices.json" for i in range(2, 10)]:
        if not choices_path.exists():
            continue
        stage = build_stage(json.loads(choices_path.read_text(encoding="utf-8")))
        if stage is not None:
            stages.append(stage)
    return stages or None

def load_compact_archive_bundle(data_dir: Path):
    """Load one-file ZIP member used by the smallest packaged variants.

    Member ``x`` stores three little-endian uint32 lengths followed by the
    brotli-compressed mask, model, pose, and post-code payloads.  Keeping these
    already-compressed blobs in one ZIP member avoids three ZIP local/central
    headers and long member names.
    """
    path = data_dir / "x"
    if not path.exists():
        return None
    raw = path.read_bytes()
    if len(raw) < 9:
        raise ValueError("bad compact archive bundle")

    def parse_at(pos, l_mask, l_model, l_pose):
        mask = raw[pos:pos + l_mask]
        pos += l_mask
        model = raw[pos:pos + l_model]
        pos += l_model
        pose = raw[pos:pos + l_pose]
        pos += l_pose
        post = raw[pos:]
        if len(mask) == l_mask and len(model) == l_model and len(pose) == l_pose and post:
            return {"mask": mask, "model": model, "pose": pose, "post": post}
        return None

    # Preferred v4/v3/v2 header: 24-bit little-endian lengths.  v4 adds
    # an explicit shift length and a fractional-shift tail.
    l_mask = int.from_bytes(raw[0:3], "little")
    l_model = int.from_bytes(raw[3:6], "little")
    l_pose = int.from_bytes(raw[6:9], "little")
    if len(raw) >= 30:
        l_post = int.from_bytes(raw[9:12], "little")
        l_shift = int.from_bytes(raw[12:15], "little")
        l_frac = int.from_bytes(raw[15:18], "little")
        l_frac2 = int.from_bytes(raw[18:21], "little")
        l_frac3 = int.from_bytes(raw[21:24], "little")
        l_bias = int.from_bytes(raw[24:27], "little")
        l_region = int.from_bytes(raw[27:30], "little")
        if l_mask > 1000 and l_model > 1000 and l_pose > 100 and 0 < l_post < 10000 and 0 < l_shift < 10000 and 0 < l_frac < 10000 and 0 < l_frac2 < 10000 and 0 < l_frac3 < 10000 and 0 < l_bias < 10000 and 0 < l_region < 10000:
            pos = 30
            mask = raw[pos:pos + l_mask]; pos += l_mask
            model = raw[pos:pos + l_model]; pos += l_model
            pose = raw[pos:pos + l_pose]; pos += l_pose
            post = raw[pos:pos + l_post]; pos += l_post
            shift = raw[pos:pos + l_shift]; pos += l_shift
            frac = raw[pos:pos + l_frac]; pos += l_frac
            frac2 = raw[pos:pos + l_frac2]; pos += l_frac2
            frac3 = raw[pos:pos + l_frac3]; pos += l_frac3
            bias = raw[pos:pos + l_bias]; pos += l_bias
            region = raw[pos:pos + l_region]; pos += l_region
            randmulti = raw[pos:]
            if len(mask) == l_mask and len(model) == l_model and len(pose) == l_pose and len(post) == l_post and len(shift) == l_shift and len(frac) == l_frac and len(frac2) == l_frac2 and len(frac3) == l_frac3 and len(bias) == l_bias and len(region) == l_region and randmulti:
                return {"mask": mask, "model": model, "pose": pose, "post": post, "shift": shift, "frac": frac, "frac2": frac2, "frac3": frac3, "bias": bias, "region": region, "randmulti": randmulti}
    if len(raw) >= 27:
        l_post = int.from_bytes(raw[9:12], "little")
        l_shift = int.from_bytes(raw[12:15], "little")
        l_frac = int.from_bytes(raw[15:18], "little")
        l_frac2 = int.from_bytes(raw[18:21], "little")
        l_frac3 = int.from_bytes(raw[21:24], "little")
        l_bias = int.from_bytes(raw[24:27], "little")
        if l_mask > 1000 and l_model > 1000 and l_pose > 100 and 0 < l_post < 10000 and 0 < l_shift < 10000 and 0 < l_frac < 10000 and 0 < l_frac2 < 10000 and 0 < l_frac3 < 10000 and 0 < l_bias < 10000:
            pos = 27
            mask = raw[pos:pos + l_mask]; pos += l_mask
            model = raw[pos:pos + l_model]; pos += l_model
            pose = raw[pos:pos + l_pose]; pos += l_pose
            post = raw[pos:pos + l_post]; pos += l_post
            shift = raw[pos:pos + l_shift]; pos += l_shift
            frac = raw[pos:pos + l_frac]; pos += l_frac
            frac2 = raw[pos:pos + l_frac2]; pos += l_frac2
            frac3 = raw[pos:pos + l_frac3]; pos += l_frac3
            bias = raw[pos:pos + l_bias]; pos += l_bias
            region = raw[pos:]
            if len(mask) == l_mask and len(model) == l_model and len(pose) == l_pose and len(post) == l_post and len(shift) == l_shift and len(frac) == l_frac and len(frac2) == l_frac2 and len(frac3) == l_frac3 and len(bias) == l_bias and region:
                return {"mask": mask, "model": model, "pose": pose, "post": post, "shift": shift, "frac": frac, "frac2": frac2, "frac3": frac3, "bias": bias, "region": region}
    # v5 micro header: omit l_bias/l_region from archive and keep these fixed in code.
    # This saves 6 archive bytes for the fixed current bundle layout.
    if len(raw) >= 24:
        l_post = int.from_bytes(raw[9:12], "little")
        l_shift = int.from_bytes(raw[12:15], "little")
        l_frac = int.from_bytes(raw[15:18], "little")
        l_frac2 = int.from_bytes(raw[18:21], "little")
        l_frac3 = int.from_bytes(raw[21:24], "little")
        l_bias = 223
        l_region = 273
        if l_mask > 1000 and l_model > 1000 and l_pose > 100 and 0 < l_post < 10000 and 0 < l_shift < 10000 and 0 < l_frac < 10000 and 0 < l_frac2 < 10000 and 0 < l_frac3 < 10000:
            pos = 24
            mask = raw[pos:pos + l_mask]; pos += l_mask
            model = raw[pos:pos + l_model]; pos += l_model
            pose = raw[pos:pos + l_pose]; pos += l_pose
            post = raw[pos:pos + l_post]; pos += l_post
            shift = raw[pos:pos + l_shift]; pos += l_shift
            frac = raw[pos:pos + l_frac]; pos += l_frac
            frac2 = raw[pos:pos + l_frac2]; pos += l_frac2
            frac3 = raw[pos:pos + l_frac3]; pos += l_frac3
            bias = raw[pos:pos + l_bias]; pos += l_bias
            region = raw[pos:pos + l_region]; pos += l_region
            randmulti = raw[pos:]
            if len(mask) == l_mask and len(model) == l_model and len(pose) == l_pose and len(post) == l_post and len(shift) == l_shift and len(frac) == l_frac and len(frac2) == l_frac2 and len(frac3) == l_frac3 and len(bias) == l_bias and len(region) == l_region and randmulti:
                return {"mask": mask, "model": model, "pose": pose, "post": post, "shift": shift, "frac": frac, "frac2": frac2, "frac3": frac3, "bias": bias, "region": region, "randmulti": randmulti}
    if len(raw) >= 24:
        l_post = int.from_bytes(raw[9:12], "little")
        l_shift = int.from_bytes(raw[12:15], "little")
        l_frac = int.from_bytes(raw[15:18], "little")
        l_frac2 = int.from_bytes(raw[18:21], "little")
        l_frac3 = int.from_bytes(raw[21:24], "little")
        if l_mask > 1000 and l_model > 1000 and l_pose > 100 and 0 < l_post < 10000 and 0 < l_shift < 10000 and 0 < l_frac < 10000 and 0 < l_frac2 < 10000 and 0 < l_frac3 < 10000:
            pos = 24
            mask = raw[pos:pos + l_mask]; pos += l_mask
            model = raw[pos:pos + l_model]; pos += l_model
            pose = raw[pos:pos + l_pose]; pos += l_pose
            post = raw[pos:pos + l_post]; pos += l_post
            shift = raw[pos:pos + l_shift]; pos += l_shift
            frac = raw[pos:pos + l_frac]; pos += l_frac
            frac2 = raw[pos:pos + l_frac2]; pos += l_frac2
            frac3 = raw[pos:pos + l_frac3]; pos += l_frac3
            bias = raw[pos:]
            if len(mask) == l_mask and len(model) == l_model and len(pose) == l_pose and len(post) == l_post and len(shift) == l_shift and len(frac) == l_frac and len(frac2) == l_frac2 and len(frac3) == l_frac3 and bias:
                return {"mask": mask, "model": model, "pose": pose, "post": post, "shift": shift, "frac": frac, "frac2": frac2, "frac3": frac3, "bias": bias}
    if len(raw) >= 21:
        l_post = int.from_bytes(raw[9:12], "little")
        l_shift = int.from_bytes(raw[12:15], "little")
        l_frac = int.from_bytes(raw[15:18], "little")
        l_frac2 = int.from_bytes(raw[18:21], "little")
        if l_mask > 1000 and l_model > 1000 and l_pose > 100 and 0 < l_post < 10000 and 0 < l_shift < 10000 and 0 < l_frac < 10000 and 0 < l_frac2 < 10000:
            pos = 21
            mask = raw[pos:pos + l_mask]; pos += l_mask
            model = raw[pos:pos + l_model]; pos += l_model
            pose = raw[pos:pos + l_pose]; pos += l_pose
            post = raw[pos:pos + l_post]; pos += l_post
            shift = raw[pos:pos + l_shift]; pos += l_shift
            frac = raw[pos:pos + l_frac]; pos += l_frac
            frac2 = raw[pos:pos + l_frac2]; pos += l_frac2
            frac3 = raw[pos:]
            if len(mask) == l_mask and len(model) == l_model and len(pose) == l_pose and len(post) == l_post and len(shift) == l_shift and len(frac) == l_frac and len(frac2) == l_frac2 and frac3:
                return {"mask": mask, "model": model, "pose": pose, "post": post, "shift": shift, "frac": frac, "frac2": frac2, "frac3": frac3}
    if len(raw) >= 18:
        l_post = int.from_bytes(raw[9:12], "little")
        l_shift = int.from_bytes(raw[12:15], "little")
        l_frac = int.from_bytes(raw[15:18], "little")
        if l_mask > 1000 and l_model > 1000 and l_pose > 100 and 0 < l_post < 10000 and 0 < l_shift < 10000 and 0 < l_frac < 10000:
            pos = 18
            mask = raw[pos:pos + l_mask]; pos += l_mask
            model = raw[pos:pos + l_model]; pos += l_model
            pose = raw[pos:pos + l_pose]; pos += l_pose
            post = raw[pos:pos + l_post]; pos += l_post
            shift = raw[pos:pos + l_shift]; pos += l_shift
            frac = raw[pos:pos + l_frac]; pos += l_frac
            frac2 = raw[pos:]
            if len(mask) == l_mask and len(model) == l_model and len(pose) == l_pose and len(post) == l_post and len(shift) == l_shift and len(frac) == l_frac and frac2:
                return {"mask": mask, "model": model, "pose": pose, "post": post, "shift": shift, "frac": frac, "frac2": frac2}
    if len(raw) >= 15:
        l_post = int.from_bytes(raw[9:12], "little")
        l_shift = int.from_bytes(raw[12:15], "little")
        if l_mask > 1000 and l_model > 1000 and l_pose > 100 and 0 < l_post < 10000 and 0 < l_shift < 10000:
            pos = 15
            mask = raw[pos:pos + l_mask]; pos += l_mask
            model = raw[pos:pos + l_model]; pos += l_model
            pose = raw[pos:pos + l_pose]; pos += l_pose
            post = raw[pos:pos + l_post]; pos += l_post
            shift = raw[pos:pos + l_shift]; pos += l_shift
            frac = raw[pos:]
            if len(mask) == l_mask and len(model) == l_model and len(pose) == l_pose and len(post) == l_post and len(shift) == l_shift and frac:
                return {"mask": mask, "model": model, "pose": pose, "post": post, "shift": shift, "frac": frac}
    if len(raw) >= 12:
        l_post = int.from_bytes(raw[9:12], "little")
        if l_mask > 1000 and l_model > 1000 and l_pose > 100 and 0 < l_post < 10000:
            pos = 12
            mask = raw[pos:pos + l_mask]; pos += l_mask
            model = raw[pos:pos + l_model]; pos += l_model
            pose = raw[pos:pos + l_pose]; pos += l_pose
            post = raw[pos:pos + l_post]; pos += l_post
            shift = raw[pos:]
            if len(mask) == l_mask and len(model) == l_model and len(pose) == l_pose and len(post) == l_post and shift:
                return {"mask": mask, "model": model, "pose": pose, "post": post, "shift": shift, "frac": b""}
    if l_mask > 1000 and l_model > 1000 and l_pose > 100:
        parsed = parse_at(9, l_mask, l_model, l_pose)
        if parsed is not None:
            parsed["shift"] = b""
            parsed["frac"] = b""
            return parsed

    # Older v1 header: three uint32 lengths.
    if len(raw) < 12:
        raise ValueError("bad compact archive bundle")
    l_mask, l_model, l_pose = struct.unpack_from("<III", raw, 0)
    pos = 12
    mask = raw[pos:pos + l_mask]
    pos += l_mask
    model = raw[pos:pos + l_model]
    pos += l_model
    pose = raw[pos:pos + l_pose]
    pos += l_pose
    post = raw[pos:]
    if len(mask) != l_mask or len(model) != l_model or len(pose) != l_pose or not post:
        raise ValueError("truncated compact archive bundle")
    return {"mask": mask, "model": model, "pose": pose, "post": post}

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

    bundle = load_compact_archive_bundle(data_dir)
    model_br = data_dir / "model.pt.br"
    mask_br = data_dir / "mask.obu.br"
    pose_br = data_dir / "pose.npy.br"

    generator = JointFrameGenerator().to(device)

    # 1. Load Weights
    if bundle is not None:
        weights_data = brotli.decompress(bundle["model"])
    else:
        with open(model_br, "rb") as f:
            weights_data = brotli.decompress(f.read())

    generator.load_state_dict(get_decoded_state_dict(weights_data, device), strict=True)
    generator.eval()

    # 2. Load Mask Video (.obu) or exact qzs3 range-coded class tensor.
    if bundle is not None and bundle["mask"][:4] in (b"QMA6", b"QMA7", b"QMA8", b"QMA9"):
        mask_frames_all = load_range_mask(bundle["mask"])
    else:
        with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp_obu:
            if bundle is not None:
                tmp_obu.write(brotli.decompress(bundle["mask"]))
            else:
                with open(mask_br, "rb") as f:
                    tmp_obu.write(brotli.decompress(f.read()))
            tmp_obu_path = tmp_obu.name
        mask_frames_all = load_encoded_mask_video(tmp_obu_path)
        os.remove(tmp_obu_path)

    # 3. Load Pose Vectors
    pose_frames_all = load_pose_frames_from_payload(bundle["pose"]) if bundle is not None else load_pose_frames(pose_br)
    postprocess = load_postprocess(data_dir, device, bundle["post"] if bundle is not None else None)
    f0_shift_choices = None
    f1_frac_choices = None
    f1_frac2_choices = None
    f1_frac3_choices = None
    f1_bias_choices = None
    f1_region_choices = None
    f1_randmulti_choices = None
    if bundle is not None and len(bundle.get("shift", b"")):
        raw_s = brotli.decompress(bundle["shift"])
        magic = raw_s[:3]
        if magic == b"SH4":
            arr = np.frombuffer(raw_s, dtype=np.uint8, offset=3).astype(np.int64)
        elif magic == b"SD4":
            d = np.frombuffer(raw_s, dtype=np.uint8, offset=3).astype(np.int64)
            arr = np.where(d == 0, 40, d - 1).astype(np.int64)
        else:
            raise ValueError("bad f1 shift payload")
        f0_shift_choices = torch.from_numpy(arr).to(device)
    if bundle is not None and len(bundle.get("frac", b"")):
        raw_f = brotli.decompress(bundle["frac"])
        magic_f = raw_f[:3]
        if magic_f == b"FH1":
            arr_f = np.frombuffer(raw_f, dtype=np.uint8, offset=3).astype(np.int64)
        elif magic_f == b"FV1":
            cnt = int.from_bytes(raw_f[3:5], "little")
            pos_f = 5; arr_f = np.full(600, 4, dtype=np.int64); idx_f = -1; inds = []
            for _ in range(cnt):
                acc = 0; sh = 0
                while True:
                    by = raw_f[pos_f]; pos_f += 1; acc |= (by & 127) << sh
                    if by & 128: sh += 7
                    else: break
                idx_f += acc + 1; inds.append(idx_f)
            vals_f = np.frombuffer(raw_f, dtype=np.uint8, count=cnt, offset=pos_f).astype(np.int64)
            for ii, vv in zip(inds, vals_f): arr_f[ii] = vv - 1
        else:
            raise ValueError("bad f1 fractional shift payload")
        f1_frac_choices = torch.from_numpy(arr_f).to(device)
    if bundle is not None and len(bundle.get("frac2", b"")):
        raw_f2 = brotli.decompress(bundle["frac2"])
        if raw_f2[:3] != b"FH2":
            raise ValueError("bad f1 fractional shift2 payload")
        f1_frac2_choices = torch.from_numpy(np.frombuffer(raw_f2, dtype=np.uint8, offset=3).astype(np.int64)).to(device)
    if bundle is not None and len(bundle.get("frac3", b"")):
        raw_f3 = brotli.decompress(bundle["frac3"])
        magic3 = raw_f3[:3]
        if magic3 == b"FH3":
            arr3 = np.frombuffer(raw_f3, dtype=np.uint8, offset=3).astype(np.int64)
        elif magic3 == b"FD3":
            d3 = np.frombuffer(raw_f3, dtype=np.uint8, offset=3).astype(np.int64)
            arr3 = np.where(d3 == 0, 4, d3 - 1).astype(np.int64)
        else:
            raise ValueError("bad f1 fractional shift3 payload")
        f1_frac3_choices = torch.from_numpy(arr3).to(device)
    if bundle is not None and len(bundle.get("bias", b"")):
        raw_b = brotli.decompress(bundle["bias"])
        magic_b = raw_b[:3]
        center_b = 13
        if magic_b == b"BH1":
            arr_b = np.frombuffer(raw_b, dtype=np.uint8, offset=3).astype(np.int64)
        elif magic_b == b"BD1":
            d_b = np.frombuffer(raw_b, dtype=np.uint8, offset=3).astype(np.int64)
            arr_b = np.where(d_b == 0, center_b, d_b - 1).astype(np.int64)
        elif magic_b == b"BV1":
            cnt_b = int.from_bytes(raw_b[3:5], "little")
            pos_b = 5; arr_b = np.full(600, center_b, dtype=np.int64); idx_b = -1; inds_b = []
            for _ in range(cnt_b):
                acc = 0; sh = 0
                while True:
                    by = raw_b[pos_b]; pos_b += 1; acc |= (by & 127) << sh
                    if by & 128: sh += 7
                    else: break
                idx_b += acc + 1; inds_b.append(idx_b)
            vals_b = np.frombuffer(raw_b, dtype=np.uint8, count=cnt_b, offset=pos_b).astype(np.int64)
            for ii, vv in zip(inds_b, vals_b): arr_b[ii] = vv - 1
        else:
            raise ValueError("bad f1 RGB bias payload")
        f1_bias_choices = torch.from_numpy(arr_b).to(device)
    if bundle is not None and len(bundle.get("region", b"")):
        raw_r = brotli.decompress(bundle["region"])
        magic_r = raw_r[:3]
        if magic_r == b"RH1":
            arr_r = np.frombuffer(raw_r, dtype=np.uint8, offset=3).astype(np.int64)
        elif magic_r == b"RD1":
            d_r = np.frombuffer(raw_r, dtype=np.uint8, offset=3).astype(np.int64)
            arr_r = np.where(d_r == 0, 0, d_r - 1).astype(np.int64)
        elif magic_r == b"RV1":
            cnt_r = int.from_bytes(raw_r[3:5], "little")
            pos_r = 5; arr_r = np.zeros(600, dtype=np.int64); idx_r = -1; inds_r = []
            for _ in range(cnt_r):
                acc = 0; sh = 0
                while True:
                    by = raw_r[pos_r]; pos_r += 1; acc |= (by & 127) << sh
                    if by & 128: sh += 7
                    else: break
                idx_r += acc + 1; inds_r.append(idx_r)
            vals_r = np.frombuffer(raw_r, dtype=np.uint8, count=cnt_r, offset=pos_r).astype(np.int64)
            for ii, vv in zip(inds_r, vals_r): arr_r[ii] = vv - 1
        else:
            raise ValueError("bad f1 region-bias payload")
        f1_region_choices = torch.from_numpy(arr_r).to(device)
    if bundle is not None and len(bundle.get("randmulti", b"")):
        raw_n = brotli.decompress(bundle["randmulti"])
        f1_randmulti_choices = []
        if raw_n[:3] == b"NM1":
            scount = int(raw_n[3])
            arr_n = np.frombuffer(raw_n, dtype=np.uint8, count=scount * 600, offset=4).reshape(scount, 600).astype(np.int64)
            f1_randmulti_choices.append((torch.from_numpy(arr_n).to(device), 24, 32, 1))
        elif raw_n[:3] == b"NM2":
            pos_n = 4
            gcount = int(raw_n[3])
            for _ in range(gcount):
                lh_n = int(raw_n[pos_n]); lw_n = int(raw_n[pos_n + 1]); amp_n = int(raw_n[pos_n + 2]); scount = int(raw_n[pos_n + 3]); pos_n += 4
                arr_n = np.frombuffer(raw_n, dtype=np.uint8, count=scount * 600, offset=pos_n).reshape(scount, 600).astype(np.int64)
                pos_n += scount * 600
                f1_randmulti_choices.append((torch.from_numpy(arr_n).to(device), lh_n, lw_n, amp_n))
        else:
            specs_n = [(24, 32, 1, 12), (12, 16, 1, 1), (6, 8, 1, 1), (3, 4, 1, 1), (2, 2, 1, 1), (8, 8, 1, 1), (4, 4, 1, 1), (4, 8, 1, 1), (2, 4, 1, 1), (2, 8, 1, 1), (1, 2, 1, 1), (1, 4, 1, 1), (2, 1, 1, 1), (4, 1, 1, 1), (8, 1, 1, 1), (1, 8, 1, 1), (16, 1, 1, 1), (1, 16, 1, 1), (32, 1, 1, 1), (64, 1, 1, 1), (256, 1, 1, 1), (1024, 1, 1, 1), (2048, 1, 1, 1), (4096, 1, 1, 1), (8192, 1, 1, 1), (8192, 1, 1, 1), (16384, 1, 1, 1), (32768, 1, 1, 1), (65536, 1, 1, 1), (131072, 1, 1, 1), (262144, 1, 1, 1), (524288, 1, 1, 1), (1048576, 1, 1, 1), (874, 1, 1, 1), (874, 1, 1, 1), (2097152, 1, 1, 1), (875, 1, 1, 1), (876, 1, 1, 1), (877, 1, 1, 1), (1164, 1, 1, 1), (878, 1, 1, 1), (879, 1, 1, 1), (880, 1, 1, 1), (881, 1, 1, 1), (882, 1, 1, 1), (512, 2, 1, 1), (256, 2, 1, 1), (128, 2, 1, 1), (64, 2, 1, 1), (32, 2, 1, 1), (16, 2, 1, 1), (8, 2, 1, 1), (4, 2, 1, 1), (4, 4, 1, 1), (8, 4, 1, 1), (16, 4, 1, 1), (32, 4, 1, 1), (64, 4, 1, 1), (128, 4, 1, 1), (64, 8, 1, 1), (32, 8, 1, 1), (222, 222, 4, 1), (222, 223, 4, 1), (223, 222, 2, 1), (223, 223, 4, 1), (223, 221, 4, 1), (223, 224, 4, 1), (223, 221, 4, 1), (223, 219, 4, 1), (64, 16, 1, 1), (223, 218, 4, 1), (224, 222, 4, 1)]
            pos_n = 0
            for lh_n, lw_n, amp_n, scount in specs_n:
                rows_n = np.zeros((scount, 600), dtype=np.uint8)
                for si_n in range(scount):
                    cnt_n = int(raw_n[pos_n]); pos_n += 1
                    if cnt_n == 255:
                        cnt_n = int.from_bytes(raw_n[pos_n:pos_n + 2], "little")
                        pos_n += 2
                    idx_n = -1
                    inds_n = []
                    for _ in range(cnt_n):
                        acc_n = 0
                        sh_n = 0
                        while True:
                            by_n = raw_n[pos_n]
                            pos_n += 1
                            acc_n |= (by_n & 127) << sh_n
                            if by_n & 128:
                                sh_n += 7
                            else:
                                break
                        idx_n += acc_n + 1
                        inds_n.append(idx_n)
                    vals_n = np.frombuffer(raw_n, dtype=np.uint8, count=cnt_n, offset=pos_n)
                    pos_n += cnt_n
                    if cnt_n:
                        rows_n[si_n, np.array(inds_n, dtype=np.int64)] = vals_n
                f1_randmulti_choices.append((torch.from_numpy(rows_n.astype(np.int64)).to(device), lh_n, lw_n, amp_n))
            if pos_n != len(raw_n):
                raise ValueError("bad headerless f1 randmulti payload")

    out_h, out_w = 874, 1164
    frac_grid_cache = {}
    frac2_grid_cache = {}
    frac3_grid_cache = {}
    randpat_cache = {}
    cursor = 0
    pair_cursor = 0
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
            cursor += pairs_per_file

            with open(raw_out_path, "wb") as f_out:
                pbar = tqdm(range(0, file_masks.shape[0], batch_size), desc=f"Decoding {file_name}")

                for i in pbar:
                    in_mask2 = file_masks[i : i + batch_size].to(device).long()
                    in_pose6 = file_poses[i : i + batch_size].to(device).float()

                    fake1, fake2 = generator(in_mask2, in_pose6)

                    fake1_up = F.interpolate(fake1, size=(out_h, out_w), mode="bilinear", align_corners=False)
                    fake2_up = F.interpolate(fake2, size=(out_h, out_w), mode="bilinear", align_corners=False)

                    batch_comp = torch.stack([fake1_up, fake2_up], dim=1).clamp(0, 255).round()

                    if postprocess is not None:
                        batch_hwc = einops.rearrange(batch_comp, "b t c h w -> b t h w c")
                        b = batch_hwc.shape[0]
                        for gains, biases, choices in postprocess:
                            idx = choices[pair_cursor + i : pair_cursor + i + b]
                            if idx.numel() < b:
                                idx = F.pad(idx, (0, b - idx.numel()))
                            idx = idx.clamp(0, gains.shape[0] - 1)
                            batch_hwc = (batch_hwc * gains[idx] + biases[idx]).clamp(0, 255).round()
                        if f0_shift_choices is not None:
                            bsz = batch_hwc.shape[0]
                            chs = f0_shift_choices[pair_cursor + i : pair_cursor + i + bsz]
                            for bi in range(bsz):
                                ch = int(chs[bi].item()) if bi < chs.numel() else 40
                                if ch != 40:
                                    dy = ch // 9 - 4
                                    dx = ch % 9 - 4
                                    img = batch_hwc[bi, 0].permute(2, 0, 1).unsqueeze(0)
                                    left, right = max(dx, 0), max(-dx, 0)
                                    top, bottom = max(dy, 0), max(-dy, 0)
                                    imgp = F.pad(img, (left, right, top, bottom), mode='replicate')
                                    y0, x0 = bottom, right
                                    batch_hwc[bi, 0] = imgp[0, :, y0:y0+out_h, x0:x0+out_w].permute(1, 2, 0)
                        if f1_frac_choices is not None:
                            bsz = batch_hwc.shape[0]
                            chs = f1_frac_choices[pair_cursor + i : pair_cursor + i + bsz]
                            for bi in range(bsz):
                                ch = int(chs[bi].item()) if bi < chs.numel() else 4
                                if ch != 4:
                                    if ch not in frac_grid_cache:
                                        dy = (ch // 3 - 1) * 0.5
                                        dx = (ch % 3 - 1) * 0.5
                                        yy, xx = torch.meshgrid(torch.arange(out_h, device=device, dtype=torch.float32), torch.arange(out_w, device=device, dtype=torch.float32), indexing='ij')
                                        gx = ((xx - dx) + 0.5) * 2.0 / out_w - 1.0
                                        gy = ((yy - dy) + 0.5) * 2.0 / out_h - 1.0
                                        frac_grid_cache[ch] = torch.stack([gx, gy], dim=-1).unsqueeze(0)
                                    img = batch_hwc[bi, 0].permute(2, 0, 1).unsqueeze(0).float()
                                    img = F.grid_sample(img, frac_grid_cache[ch], mode='bilinear', padding_mode='border', align_corners=False)
                                    batch_hwc[bi, 0] = img[0].clamp(0, 255).round().permute(1, 2, 0)
                        if f1_frac2_choices is not None:
                            bsz = batch_hwc.shape[0]
                            chs = f1_frac2_choices[pair_cursor + i : pair_cursor + i + bsz]
                            for bi in range(bsz):
                                ch = int(chs[bi].item()) if bi < chs.numel() else 4
                                if ch != 4:
                                    if ch not in frac2_grid_cache:
                                        dy = (ch // 3 - 1) * 0.25
                                        dx = (ch % 3 - 1) * 0.25
                                        yy, xx = torch.meshgrid(torch.arange(out_h, device=device, dtype=torch.float32), torch.arange(out_w, device=device, dtype=torch.float32), indexing='ij')
                                        gx = ((xx - dx) + 0.5) * 2.0 / out_w - 1.0
                                        gy = ((yy - dy) + 0.5) * 2.0 / out_h - 1.0
                                        frac2_grid_cache[ch] = torch.stack([gx, gy], dim=-1).unsqueeze(0)
                                    img = batch_hwc[bi, 0].permute(2, 0, 1).unsqueeze(0).float()
                                    img = F.grid_sample(img, frac2_grid_cache[ch], mode='bilinear', padding_mode='border', align_corners=False)
                                    batch_hwc[bi, 0] = img[0].clamp(0, 255).round().permute(1, 2, 0)
                        if f1_frac3_choices is not None:
                            bsz = batch_hwc.shape[0]
                            chs = f1_frac3_choices[pair_cursor + i : pair_cursor + i + bsz]
                            for bi in range(bsz):
                                ch = int(chs[bi].item()) if bi < chs.numel() else 4
                                if ch != 4:
                                    if ch not in frac3_grid_cache:
                                        dy = (ch // 3 - 1) * 0.125
                                        dx = (ch % 3 - 1) * 0.125
                                        yy, xx = torch.meshgrid(torch.arange(out_h, device=device, dtype=torch.float32), torch.arange(out_w, device=device, dtype=torch.float32), indexing='ij')
                                        gx = ((xx - dx) + 0.5) * 2.0 / out_w - 1.0
                                        gy = ((yy - dy) + 0.5) * 2.0 / out_h - 1.0
                                        frac3_grid_cache[ch] = torch.stack([gx, gy], dim=-1).unsqueeze(0)
                                    img = batch_hwc[bi, 0].permute(2, 0, 1).unsqueeze(0).float()
                                    img = F.grid_sample(img, frac3_grid_cache[ch], mode='bilinear', padding_mode='border', align_corners=False)
                                    batch_hwc[bi, 0] = img[0].clamp(0, 255).round().permute(1, 2, 0)
                        if f1_bias_choices is not None:
                            bsz = batch_hwc.shape[0]
                            chs = f1_bias_choices[pair_cursor + i : pair_cursor + i + bsz]
                            for bi in range(bsz):
                                ch = int(chs[bi].item()) if bi < chs.numel() else 13
                                if ch != 13:
                                    br = ch // 9 - 1
                                    bg = (ch // 3) % 3 - 1
                                    bb = ch % 3 - 1
                                    bias = torch.tensor([br, bg, bb], device=device, dtype=batch_hwc.dtype)
                                    batch_hwc[bi, 0] = (batch_hwc[bi, 0] + bias).clamp(0, 255).round()
                        if f1_region_choices is not None:
                            bsz = batch_hwc.shape[0]
                            chs = f1_region_choices[pair_cursor + i : pair_cursor + i + bsz]
                            yy_idx = torch.arange(out_h, device=device).view(out_h, 1).expand(out_h, out_w)
                            xx_idx = torch.arange(out_w, device=device).view(1, out_w).expand(out_h, out_w)
                            for bi in range(bsz):
                                ch = int(chs[bi].item()) if bi < chs.numel() else 0
                                if ch != 0:
                                    j = ch - 1
                                    val_list = [-2, -1, 1, 2]
                                    val = float(val_list[j % 4]); j //= 4
                                    ci = j % 4; j //= 4
                                    mi = j
                                    if mi == 0: mask = (yy_idx < out_h // 2)
                                    elif mi == 1: mask = (yy_idx >= out_h // 2)
                                    elif mi == 2: mask = (xx_idx < out_w // 2)
                                    elif mi == 3: mask = (xx_idx >= out_w // 2)
                                    elif mi == 4: mask = ((yy_idx >= out_h // 3) & (yy_idx < 2 * out_h // 3))
                                    else: mask = ((xx_idx >= out_w // 3) & (xx_idx < 2 * out_w // 3))
                                    if ci == 0:
                                        batch_hwc[bi, 0][mask, :] = (batch_hwc[bi, 0][mask, :] + val).clamp(0, 255).round()
                                    else:
                                        cc = ci - 1
                                        batch_hwc[bi, 0][mask, cc] = (batch_hwc[bi, 0][mask, cc] + val).clamp(0, 255).round()
                        if f1_randmulti_choices is not None:
                            bsz = batch_hwc.shape[0]
                            for arr_choices, lh_r, lw_r, amp_r in f1_randmulti_choices:
                                if lh_r == 224 and lw_r == 222 and amp_r == 4:
                                    # f2 tile bias, 2x2 grid, per-channel radius 4.
                                    bsz = batch_hwc.shape[0]
                                    chs = arr_choices[0, pair_cursor + i : pair_cursor + i + bsz]
                                    if chs.numel() < bsz:
                                        chs = F.pad(chs, (0, bsz - chs.numel()))
                                    yy_idx = torch.arange(out_h, device=device).view(out_h, 1).expand(out_h, out_w)
                                    xx_idx = torch.arange(out_w, device=device).view(1, out_w).expand(out_h, out_w)
                                    tile_masks = [
                                        ((yy_idx < out_h // 2) & (xx_idx < out_w // 2)),
                                        ((yy_idx < out_h // 2) & (xx_idx >= out_w // 2)),
                                        ((yy_idx >= out_h // 2) & (xx_idx < out_w // 2)),
                                        ((yy_idx >= out_h // 2) & (xx_idx >= out_w // 2)),
                                    ]
                                    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                                    for bi in range(bsz):
                                        ch = int(chs[bi].item())
                                        if ch != 0:
                                            jj = ch - 1
                                            tile = jj // 24
                                            rem = jj % 24
                                            val = vals[rem // 3]
                                            typ = rem % 3
                                            if typ == 0:
                                                bias = torch.tensor([val, 0, 0], device=device, dtype=batch_hwc.dtype)
                                            elif typ == 1:
                                                bias = torch.tensor([0, val, 0], device=device, dtype=batch_hwc.dtype)
                                            else:
                                                bias = torch.tensor([0, 0, val], device=device, dtype=batch_hwc.dtype)
                                            mm = tile_masks[tile].unsqueeze(-1)
                                            batch_hwc[bi, 1] = torch.where(mm, (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                                    continue
                                if lh_r == 223 and lw_r == 223 and amp_r == 4:
                                    # f2 boundary all-channel bias, radius 4.
                                    bsz = batch_hwc.shape[0]
                                    chs = arr_choices[0, pair_cursor + i : pair_cursor + i + bsz]
                                    if chs.numel() < bsz:
                                        chs = F.pad(chs, (0, bsz - chs.numel()))
                                    mask_low = in_mask2.to(torch.uint8)
                                    mx = F.max_pool2d(mask_low.float().unsqueeze(1), 3, 1, 1)[:, 0]
                                    mn = -F.max_pool2d(-mask_low.float().unsqueeze(1), 3, 1, 1)[:, 0]
                                    bnd_low = (mx != mn).float().unsqueeze(1)
                                    mask_up = F.interpolate(mask_low.float().unsqueeze(1), size=(out_h, out_w), mode="nearest")[:, 0].to(torch.uint8)
                                    bnd = F.interpolate(bnd_low, size=(out_h, out_w), mode="nearest")[:, 0].to(torch.bool)
                                    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                                    for bi in range(bsz):
                                        ch = int(chs[bi].item())
                                        if ch != 0:
                                            jj = ch - 1
                                            if jj < 8:
                                                mm = bnd[bi]
                                                val = vals[jj]
                                            else:
                                                jj -= 8
                                                cls = jj // 8
                                                val = vals[jj % 8]
                                                mm = bnd[bi] & (mask_up[bi] == cls)
                                            bias = torch.tensor([val, val, val], device=device, dtype=batch_hwc.dtype)
                                            batch_hwc[bi, 1] = torch.where(mm.unsqueeze(-1), (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                                    continue
                                if lh_r == 223 and lw_r == 222 and amp_r == 2:
                                    # f2 class-conditioned channel bias, radius 2.
                                    bsz = batch_hwc.shape[0]
                                    chs = arr_choices[0, pair_cursor + i : pair_cursor + i + bsz]
                                    if chs.numel() < bsz:
                                        chs = F.pad(chs, (0, bsz - chs.numel()))
                                    mask_up = F.interpolate(in_mask2.float().unsqueeze(1), size=(out_h, out_w), mode="nearest")[:, 0].to(torch.uint8)
                                    vals = [-2, -1, 1, 2]
                                    for bi in range(bsz):
                                        ch = int(chs[bi].item())
                                        if ch != 0:
                                            jj = ch - 1
                                            cls = jj // 12
                                            rem = jj % 12
                                            val = vals[rem // 3]
                                            typ = rem % 3
                                            if typ == 0:
                                                bias = torch.tensor([val, 0, 0], device=device, dtype=batch_hwc.dtype)
                                            elif typ == 1:
                                                bias = torch.tensor([0, val, 0], device=device, dtype=batch_hwc.dtype)
                                            else:
                                                bias = torch.tensor([0, 0, val], device=device, dtype=batch_hwc.dtype)
                                            mm = (mask_up[bi] == cls).unsqueeze(-1)
                                            batch_hwc[bi, 1] = torch.where(mm, (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                                    continue
                                if lh_r == 223 and lw_r == 224 and amp_r == 4:
                                    # f2 boundary/class-boundary channel bias, radius 4.
                                    bsz = batch_hwc.shape[0]
                                    chs = arr_choices[0, pair_cursor + i : pair_cursor + i + bsz]
                                    if chs.numel() < bsz:
                                        chs = F.pad(chs, (0, bsz - chs.numel()))
                                    mask_low = in_mask2.to(torch.uint8)
                                    mx = F.max_pool2d(mask_low.float().unsqueeze(1), 3, 1, 1)[:, 0]
                                    mn = -F.max_pool2d(-mask_low.float().unsqueeze(1), 3, 1, 1)[:, 0]
                                    bnd_low = (mx != mn).float().unsqueeze(1)
                                    mask_up = F.interpolate(mask_low.float().unsqueeze(1), size=(out_h, out_w), mode="nearest")[:, 0].to(torch.uint8)
                                    bnd = F.interpolate(bnd_low, size=(out_h, out_w), mode="nearest")[:, 0].to(torch.bool)
                                    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                                    for bi in range(bsz):
                                        ch = int(chs[bi].item())
                                        if ch != 0:
                                            jj = ch - 1
                                            if jj < 24:
                                                mm = bnd[bi]
                                                val = vals[jj // 3]
                                                typ = jj % 3
                                            else:
                                                jj -= 24
                                                cls = jj // 24
                                                rem = jj % 24
                                                val = vals[rem // 3]
                                                typ = rem % 3
                                                mm = bnd[bi] & (mask_up[bi] == cls)
                                            if typ == 0:
                                                bias = torch.tensor([val, 0, 0], device=device, dtype=batch_hwc.dtype)
                                            elif typ == 1:
                                                bias = torch.tensor([0, val, 0], device=device, dtype=batch_hwc.dtype)
                                            else:
                                                bias = torch.tensor([0, 0, val], device=device, dtype=batch_hwc.dtype)
                                            batch_hwc[bi, 1] = torch.where(mm.unsqueeze(-1), (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                                    continue
                                if lh_r == 223 and lw_r == 219 and amp_r == 4:
                                    # f2 width-2 boundary channel/class bias, radius 4.
                                    bsz = batch_hwc.shape[0]
                                    chs = arr_choices[0, pair_cursor + i : pair_cursor + i + bsz]
                                    if chs.numel() < bsz:
                                        chs = F.pad(chs, (0, bsz - chs.numel()))
                                    mask_low = in_mask2.to(torch.uint8)
                                    mx = F.max_pool2d(mask_low.float().unsqueeze(1), 3, 1, 1)[:, 0]
                                    mn = -F.max_pool2d(-mask_low.float().unsqueeze(1), 3, 1, 1)[:, 0]
                                    bnd_low = (mx != mn).float().unsqueeze(1)
                                    bnd_low = F.max_pool2d(bnd_low, 3, 1, 1)
                                    mask_up = F.interpolate(mask_low.float().unsqueeze(1), size=(out_h, out_w), mode="nearest")[:, 0].to(torch.uint8)
                                    bnd = F.interpolate(bnd_low, size=(out_h, out_w), mode="nearest")[:, 0].to(torch.bool)
                                    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                                    for bi in range(bsz):
                                        ch = int(chs[bi].item())
                                        if ch != 0:
                                            jj = ch - 1
                                            if jj < 24:
                                                mm = bnd[bi]
                                                val = vals[jj // 3]
                                                typ = jj % 3
                                            else:
                                                jj -= 24
                                                cls = jj // 24
                                                rem = jj % 24
                                                val = vals[rem // 3]
                                                typ = rem % 3
                                                mm = bnd[bi] & (mask_up[bi] == cls)
                                            if typ == 0:
                                                bias = torch.tensor([val, 0, 0], device=device, dtype=batch_hwc.dtype)
                                            elif typ == 1:
                                                bias = torch.tensor([0, val, 0], device=device, dtype=batch_hwc.dtype)
                                            else:
                                                bias = torch.tensor([0, 0, val], device=device, dtype=batch_hwc.dtype)
                                            batch_hwc[bi, 1] = torch.where(mm.unsqueeze(-1), (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                                    continue
                                if lh_r == 223 and lw_r == 218 and amp_r == 4:
                                    # f2 width-3 boundary channel/class bias, radius 4.
                                    bsz = batch_hwc.shape[0]
                                    chs = arr_choices[0, pair_cursor + i : pair_cursor + i + bsz]
                                    if chs.numel() < bsz:
                                        chs = F.pad(chs, (0, bsz - chs.numel()))
                                    mask_low = in_mask2.to(torch.uint8)
                                    mx = F.max_pool2d(mask_low.float().unsqueeze(1), 3, 1, 1)[:, 0]
                                    mn = -F.max_pool2d(-mask_low.float().unsqueeze(1), 3, 1, 1)[:, 0]
                                    bnd_low = (mx != mn).float().unsqueeze(1)
                                    bnd_low = F.max_pool2d(bnd_low, 3, 1, 1)
                                    bnd_low = F.max_pool2d(bnd_low, 3, 1, 1)
                                    mask_up = F.interpolate(mask_low.float().unsqueeze(1), size=(out_h, out_w), mode="nearest")[:, 0].to(torch.uint8)
                                    bnd = F.interpolate(bnd_low, size=(out_h, out_w), mode="nearest")[:, 0].to(torch.bool)
                                    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                                    for bi in range(bsz):
                                        ch = int(chs[bi].item())
                                        if ch != 0:
                                            jj = ch - 1
                                            if jj < 24:
                                                mm = bnd[bi]
                                                val = vals[jj // 3]
                                                typ = jj % 3
                                            else:
                                                jj -= 24
                                                cls = jj // 24
                                                rem = jj % 24
                                                val = vals[rem // 3]
                                                typ = rem % 3
                                                mm = bnd[bi] & (mask_up[bi] == cls)
                                            if typ == 0:
                                                bias = torch.tensor([val, 0, 0], device=device, dtype=batch_hwc.dtype)
                                            elif typ == 1:
                                                bias = torch.tensor([0, val, 0], device=device, dtype=batch_hwc.dtype)
                                            else:
                                                bias = torch.tensor([0, 0, val], device=device, dtype=batch_hwc.dtype)
                                            batch_hwc[bi, 1] = torch.where(mm.unsqueeze(-1), (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                                    continue
                                if lh_r == 223 and lw_r == 221 and amp_r == 4:
                                    # f2 class-conditioned channel bias, radius 4.
                                    bsz = batch_hwc.shape[0]
                                    chs = arr_choices[0, pair_cursor + i : pair_cursor + i + bsz]
                                    if chs.numel() < bsz:
                                        chs = F.pad(chs, (0, bsz - chs.numel()))
                                    mask_up = F.interpolate(in_mask2.float().unsqueeze(1), size=(out_h, out_w), mode="nearest")[:, 0].to(torch.uint8)
                                    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                                    for bi in range(bsz):
                                        ch = int(chs[bi].item())
                                        if ch != 0:
                                            jj = ch - 1
                                            cls = jj // 24
                                            rem = jj % 24
                                            val = vals[rem // 3]
                                            typ = rem % 3
                                            if typ == 0:
                                                bias = torch.tensor([val, 0, 0], device=device, dtype=batch_hwc.dtype)
                                            elif typ == 1:
                                                bias = torch.tensor([0, val, 0], device=device, dtype=batch_hwc.dtype)
                                            else:
                                                bias = torch.tensor([0, 0, val], device=device, dtype=batch_hwc.dtype)
                                            mm = (mask_up[bi] == cls).unsqueeze(-1)
                                            batch_hwc[bi, 1] = torch.where(mm, (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                                    continue
                                if lh_r == 223 and lw_r == 221 and amp_r == 4:
                                    # f2 class-conditioned channel bias, radius 4.
                                    bsz = batch_hwc.shape[0]
                                    chs = arr_choices[0, pair_cursor + i : pair_cursor + i + bsz]
                                    if chs.numel() < bsz:
                                        chs = F.pad(chs, (0, bsz - chs.numel()))
                                    mask_up = F.interpolate(in_mask2.float().unsqueeze(1), size=(out_h, out_w), mode="nearest")[:, 0].to(torch.uint8)
                                    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                                    for bi in range(bsz):
                                        ch = int(chs[bi].item())
                                        if ch != 0:
                                            jj = ch - 1
                                            cls = jj // 24
                                            rem = jj % 24
                                            val = vals[rem // 3]
                                            typ = rem % 3
                                            if typ == 0:
                                                bias = torch.tensor([val, 0, 0], device=device, dtype=batch_hwc.dtype)
                                            elif typ == 1:
                                                bias = torch.tensor([0, val, 0], device=device, dtype=batch_hwc.dtype)
                                            else:
                                                bias = torch.tensor([0, 0, val], device=device, dtype=batch_hwc.dtype)
                                            mm = (mask_up[bi] == cls).unsqueeze(-1)
                                            batch_hwc[bi, 1] = torch.where(mm, (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                                    continue
                                if lh_r == 222 and lw_r == 223 and amp_r == 4:
                                    # f2 class-conditioned all-channel bias, radius 4.
                                    bsz = batch_hwc.shape[0]
                                    chs = arr_choices[0, pair_cursor + i : pair_cursor + i + bsz]
                                    if chs.numel() < bsz:
                                        chs = F.pad(chs, (0, bsz - chs.numel()))
                                    mask_up = F.interpolate(in_mask2.float().unsqueeze(1), size=(out_h, out_w), mode="nearest")[:, 0].to(torch.uint8)
                                    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                                    for bi in range(bsz):
                                        ch = int(chs[bi].item())
                                        if ch != 0:
                                            cls = (ch - 1) // 8
                                            val = vals[(ch - 1) % 8]
                                            mm = (mask_up[bi] == cls).unsqueeze(-1)
                                            bias = torch.tensor([val, val, val], device=device, dtype=batch_hwc.dtype)
                                            batch_hwc[bi, 1] = torch.where(mm, (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                                    continue
                                if lh_r == 222 and lw_r == 222 and amp_r == 4:
                                    # f2 global RGB-bias oracle stage, radius 4.
                                    bsz = batch_hwc.shape[0]
                                    chs = arr_choices[0, pair_cursor + i : pair_cursor + i + bsz]
                                    if chs.numel() < bsz:
                                        chs = F.pad(chs, (0, bsz - chs.numel()))
                                    for bi in range(bsz):
                                        ch = int(chs[bi].item())
                                        if ch != 0:
                                            vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                                            val = vals[(ch - 1) // 4]
                                            typ = (ch - 1) % 4
                                            if typ == 0:
                                                bias = torch.tensor([val, val, val], device=device, dtype=batch_hwc.dtype)
                                            elif typ == 1:
                                                bias = torch.tensor([val, 0, 0], device=device, dtype=batch_hwc.dtype)
                                            elif typ == 2:
                                                bias = torch.tensor([0, val, 0], device=device, dtype=batch_hwc.dtype)
                                            else:
                                                bias = torch.tensor([0, 0, val], device=device, dtype=batch_hwc.dtype)
                                            batch_hwc[bi, 1] = (batch_hwc[bi, 1] + bias).clamp(0, 255).round()
                                    continue
                                zero_low = None
                                for st in range(arr_choices.shape[0]):
                                    chs = arr_choices[st, pair_cursor + i : pair_cursor + i + bsz]
                                    if chs.numel() < bsz:
                                        chs = F.pad(chs, (0, bsz - chs.numel()))
                                    if bool((chs == 0).all().item()):
                                        continue
                                    lows = []
                                    for bi in range(bsz):
                                        ch = int(chs[bi].item())
                                        if ch == 0:
                                            if zero_low is None:
                                                zero_low = torch.zeros((3, lh_r, lw_r), dtype=torch.float32, device=device)
                                            lows.append(zero_low)
                                        else:
                                            key = (lh_r, lw_r, amp_r, ch)
                                            if key not in randpat_cache:
                                                rng = np.random.default_rng(1000 + ch)
                                                arr = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(3, lh_r, lw_r)).astype(np.float32) * float(amp_r)
                                                randpat_cache[key] = torch.from_numpy(arr).to(device)
                                            lows.append(randpat_cache[key])
                                    pat = F.interpolate(torch.stack(lows, dim=0), size=(out_h, out_w), mode='nearest').permute(0, 2, 3, 1).contiguous()
                                    batch_hwc[:, 0] = (batch_hwc[:, 0] + pat).clamp(0, 255).round()
                        batch_comp = einops.rearrange(batch_hwc, "b t h w c -> b t c h w")

                    output_bytes = einops.rearrange(batch_comp, "b t c h w -> (b t) h w c").to(torch.uint8)
                    f_out.write(output_bytes.cpu().numpy().tobytes())
            pair_cursor += file_masks.shape[0]

if __name__ == "__main__":
    main()
