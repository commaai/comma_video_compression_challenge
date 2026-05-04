#!/usr/bin/env python
import io
import json
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


MAGIC_MODEL_COMPACT = b"QZMB1\0\0\0"
MAGIC_POSE_DV = b"QZPDV1\0\0"


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
        return (q * scales[:, None].to(torch.float32)).view(-1)[:flat_n].reshape(orig_shape)


def unpack_nibbles(packed: torch.Tensor, count: int) -> torch.Tensor:
    flat = packed.reshape(-1)
    out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = (flat >> 4) & 0x0F
    out[1::2] = flat & 0x0F
    return out[:count]


def arch_config_from_export(config: dict | None) -> dict:
    defaults = {
        "mask_decode_perm": [0, 1, 2, 3, 4],
        "emb_dim": 6,
        "trunk_c1": 56,
        "trunk_c2": 64,
        "head_hidden": 52,
        "cond_dim": 48,
        "depth_mult": 1,
        "mask_feature_mode": "none",
        "mask_repair_ch": 0,
        "output_adapter_ch": 0,
        "frame1_residual_ch": 0,
        "frame2_residual_ch": 0,
        "atlas_ch": 0,
        "time_feature_mode": "none",
        "luma_fiducial_ch": 0,
        "affine_latent_ch": 0,
    }
    out = defaults | (config or {})
    unsupported = {
        "mask_feature_mode": out["mask_feature_mode"] != "none",
        "mask_repair_ch": int(out["mask_repair_ch"]) != 0,
        "output_adapter_ch": int(out["output_adapter_ch"]) != 0,
        "frame1_residual_ch": int(out["frame1_residual_ch"]) != 0,
        "frame2_residual_ch": int(out["frame2_residual_ch"]) != 0,
        "atlas_ch": int(out["atlas_ch"]) != 0,
        "time_feature_mode": out["time_feature_mode"] != "none",
        "luma_fiducial_ch": int(out["luma_fiducial_ch"]) != 0,
        "affine_latent_ch": int(out["affine_latent_ch"]) != 0,
    }
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise ValueError("flatpup inflater only supports the promoted compact default architecture; unsupported: " + ", ".join(enabled))
    out["mask_decode_perm"] = [int(v) for v in out["mask_decode_perm"]]
    for key in ("emb_dim", "trunk_c1", "trunk_c2", "head_hidden", "cond_dim", "depth_mult"):
        out[key] = int(out[key])
    return out


def numpy_dtype_for_torch(dtype: torch.dtype):
    mapping = {
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.bool: np.bool_,
    }
    return mapping[dtype]


def read_tensor(payload: bytes, offset: int, shape, dtype: torch.dtype):
    np_dtype = numpy_dtype_for_torch(dtype)
    numel = int(np.prod(shape, dtype=np.int64))
    byte_count = numel * np.dtype(np_dtype).itemsize
    if offset + byte_count > len(payload):
        raise ValueError("truncated compact model payload")
    arr = np.frombuffer(payload, dtype=np_dtype, count=numel, offset=offset).copy()
    return torch.from_numpy(arr).reshape(tuple(shape)), offset + byte_count


class QConv2d(nn.Conv2d):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)


class QEmbedding(nn.Embedding):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)


class QLinear(nn.Linear):
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
        gamma, beta = self.film_proj(cond_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return self.act(residual + x * (1.0 + gamma) + beta)


class SharedMaskDecoder(nn.Module):
    def __init__(self, num_classes=5, emb_dim=6, c1=56, c2=64, depth_mult=1):
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
        stem = self.stem_conv(torch.cat([e2_up, coords], dim=1))
        s = self.stem_block(stem)
        z = self.down_block(self.down_conv(s))
        z = self.up(z)
        return self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))


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
        return torch.sigmoid(self.head(self.pre(x))) * 255.0


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
        return torch.sigmoid(self.head(self.pre(x))) * 255.0


class JointFrameGenerator(nn.Module):
    def __init__(self, arch_config: dict | None = None):
        super().__init__()
        self.arch_config = arch_config_from_export(arch_config)
        c1 = self.arch_config["trunk_c1"]
        c2 = self.arch_config["trunk_c2"]
        hidden = self.arch_config["head_hidden"]
        cond_dim = self.arch_config["cond_dim"]
        depth_mult = self.arch_config["depth_mult"]
        self.shared_trunk = SharedMaskDecoder(
            emb_dim=self.arch_config["emb_dim"],
            c1=c1,
            c2=c2,
            depth_mult=depth_mult,
        )
        self.pose_mlp = nn.Sequential(nn.Linear(6, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.frame1_head = FrameHead(c1, cond_dim=cond_dim, hidden=hidden, depth_mult=depth_mult)
        self.frame2_head = Frame2StaticHead(c1, hidden=hidden, depth_mult=depth_mult)

    def forward(self, mask2: torch.Tensor, pose6: torch.Tensor):
        coords = make_coord_grid(mask2.shape[0], 384, 512, mask2.device, torch.float32)
        shared_feat = self.shared_trunk(mask2, coords)
        cond_emb = self.pose_mlp(pose6)
        return self.frame1_head(shared_feat, cond_emb), self.frame2_head(shared_feat)


def make_coord_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


def model_export_layout(arch_config: dict):
    rng_state = torch.random.get_rng_state()
    try:
        layout_model = JointFrameGenerator(arch_config)
    finally:
        torch.random.set_rng_state(rng_state)

    quantized = []
    covered = set()
    for name, module in layout_model.named_modules():
        if isinstance(module, (QConv2d, QEmbedding, QLinear)):
            rec_type = "conv2d" if isinstance(module, QConv2d) else "embedding" if isinstance(module, QEmbedding) else "linear"
            bias_shape = tuple(module.bias.shape) if getattr(module, "bias", None) is not None else None
            quantized.append((name, rec_type, tuple(module.weight.shape), bias_shape))
            covered.add(f"{name}.weight")
            if bias_shape is not None:
                covered.add(f"{name}.bias")

    dense = []
    for name, tensor in layout_model.state_dict().items():
        if name not in covered:
            dense.append((name, tuple(tensor.shape), tensor.dtype))
    return quantized, dense


def deserialize_compact_model(payload: bytes):
    offset = len(MAGIC_MODEL_COMPACT)
    block_size, arch_len = struct.unpack_from("<HH", payload, offset)
    offset += 4
    arch_config = arch_config_from_export(json.loads(payload[offset : offset + arch_len].decode("utf-8")))
    offset += arch_len
    quant_layout, dense_layout = model_export_layout(arch_config)
    data = {"arch_config": arch_config, "quantized": {}, "dense": {}, "block_size": int(block_size)}

    for name, rec_type, weight_shape, bias_shape in quant_layout:
        kind = payload[offset]
        offset += 1
        rec = {"type": rec_type, "weight_shape": list(weight_shape)}
        weight_numel = int(np.prod(weight_shape, dtype=np.int64))
        if kind == 0:
            scale_count = (weight_numel + int(block_size) - 1) // int(block_size)
            packed_len = (scale_count * int(block_size) + 1) // 2
            rec["packed_weight"], offset = read_tensor(payload, offset, (packed_len,), torch.uint8)
            rec["scales_fp16"], offset = read_tensor(payload, offset, (scale_count,), torch.float16)
            rec["weight_kind"] = "fp4_packed"
        elif kind == 1:
            rec["weight_fp16"], offset = read_tensor(payload, offset, weight_shape, torch.float16)
            rec["weight_kind"] = "fp16"
        else:
            raise ValueError(f"unsupported compact model weight kind {kind}")
        if bias_shape is not None:
            rec["bias_fp16"], offset = read_tensor(payload, offset, bias_shape, torch.float16)
        else:
            rec["bias_fp16"] = None
        data["quantized"][name] = rec

    for name, shape, dtype in dense_layout:
        stored_dtype = torch.float16 if torch.is_floating_point(torch.empty((), dtype=dtype)) else dtype
        data["dense"][name], offset = read_tensor(payload, offset, shape, stored_dtype)

    if offset != len(payload):
        raise ValueError("compact model payload has trailing bytes")
    return data


def decoded_state_dict(data, device: torch.device):
    state_dict = {}
    for name, rec in data["quantized"].items():
        if rec["weight_kind"] == "fp4_packed":
            nibbles = unpack_nibbles(rec["packed_weight"].to(device), rec["packed_weight"].numel() * 2)
            weight = FP4Codebook.dequantize_from_nibbles(nibbles, rec["scales_fp16"].to(device), rec["weight_shape"])
        else:
            weight = rec["weight_fp16"].to(device).float()
        state_dict[f"{name}.weight"] = weight.float()
        if rec["bias_fp16"] is not None:
            state_dict[f"{name}.bias"] = rec["bias_fp16"].to(device).float()
    for name, tensor in data["dense"].items():
        state_dict[name] = tensor.to(device).float() if torch.is_floating_point(tensor) else tensor.to(device)
    return state_dict


def load_mask_video(path: str) -> torch.Tensor:
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="gray")
        frames.append(np.clip(np.round(img / 63.0), 0, 4).astype(np.uint8))
    container.close()
    return torch.from_numpy(np.stack(frames)).contiguous()


def apply_mask_decode_perm(masks: torch.Tensor, decode_perm: list[int]) -> torch.Tensor:
    if decode_perm == [0, 1, 2, 3, 4]:
        return masks
    lut = torch.tensor(decode_perm, dtype=masks.dtype, device=masks.device)
    return lut[masks.long()].contiguous()


def first_existing(data_dir: Path, *names: str) -> Path:
    for name in names:
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"none of {names} found in {data_dir}")


def read_payload(data_dir: Path, *names: str) -> bytes:
    return brotli.decompress(first_existing(data_dir, *names).read_bytes())


def read_signed_varints(payload: bytes, offset: int, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.int32)
    for i in range(count):
        shift = 0
        value = 0
        while True:
            byte = payload[offset]
            offset += 1
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        out[i] = (value >> 1) ^ -(value & 1)
    if offset != len(payload):
        raise ValueError("pose payload has trailing bytes")
    return out


def decode_pose_delta(payload: bytes) -> torch.Tensor:
    if payload[:8] != MAGIC_POSE_DV:
        return torch.from_numpy(np.load(io.BytesIO(payload))).float()
    n, d, bits = struct.unpack_from("<III", payload, 8)
    offset = 20
    lo = np.frombuffer(payload, dtype=np.float32, count=d, offset=offset).copy()
    offset += d * 4
    scale = np.frombuffer(payload, dtype=np.float32, count=d, offset=offset).copy()
    offset += d * 4
    dtype = np.uint8 if bits <= 8 else np.uint16
    first = np.frombuffer(payload, dtype=dtype, count=d, offset=offset).astype(np.int32)
    offset += d * np.dtype(dtype).itemsize
    deltas = read_signed_varints(payload, offset, (n - 1) * d).reshape(max(0, n - 1), d)
    q = np.empty((n, d), dtype=np.int32)
    q[0] = first
    if n > 1:
        q[1:] = first + np.cumsum(deltas, axis=0, dtype=np.int32)
    return torch.from_numpy(lo + q.astype(np.float32) * scale).float()


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python inflate.py <data_dir> <output_dir> <file_list_txt>")
        raise SystemExit(2)

    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    file_list = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_export = deserialize_compact_model(read_payload(data_dir, "w", "model.pt.br"))
    generator = JointFrameGenerator(model_export["arch_config"]).to(device)
    generator.load_state_dict(decoded_state_dict(model_export, device), strict=True)
    generator.eval()

    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
        tmp.write(read_payload(data_dir, "m", "mask.obu.br"))
        mask_obu_path = tmp.name
    try:
        masks = apply_mask_decode_perm(load_mask_video(mask_obu_path), model_export["arch_config"]["mask_decode_perm"])
    finally:
        os.remove(mask_obu_path)

    poses = decode_pose_delta(read_payload(data_dir, "p", "pose.dv.br"))
    files = [line.strip() for line in file_list.read_text().splitlines() if line.strip()]
    out_h, out_w = 874, 1164
    pairs_per_file = 600
    batch_size = 4
    cursor = 0

    with torch.inference_mode():
        for file_name in files:
            base_name = os.path.splitext(file_name)[0]
            file_masks = masks[cursor : cursor + pairs_per_file]
            file_poses = poses[cursor : cursor + pairs_per_file]
            cursor += pairs_per_file
            with (out_dir / f"{base_name}.raw").open("wb") as f_out:
                for i in tqdm(range(0, file_masks.shape[0], batch_size), desc=f"Decoding {file_name}"):
                    mask_batch = file_masks[i : i + batch_size].to(device).long()
                    pose_batch = file_poses[i : i + batch_size].to(device).float()
                    fake1, fake2 = generator(mask_batch, pose_batch)
                    fake1 = F.interpolate(fake1, size=(out_h, out_w), mode="bilinear", align_corners=False)
                    fake2 = F.interpolate(fake2, size=(out_h, out_w), mode="bilinear", align_corners=False)
                    batch = torch.stack([fake1, fake2], dim=1)
                    batch = einops.rearrange(batch, "b t c h w -> (b t) h w c")
                    f_out.write(batch.clamp(0, 255).round().to(torch.uint8).cpu().numpy().tobytes())


if __name__ == "__main__":
    main()
