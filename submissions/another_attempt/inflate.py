#!/usr/bin/env python
"""
another_attempt/inflate.py
Wider generator: c1=64, c2=80 (matching compress.py).
"""
import io, os, sys, tempfile
from pathlib import Path

import av
import brotli
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ─── FP4 dequantization ───────────────────────────────────────────────────────
class FP4Codebook:
    pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

    @staticmethod
    def dequantize_from_nibbles(nibbles, scales, orig_shape):
        flat_n     = int(torch.tensor(orig_shape).prod().item())
        block_size = nibbles.numel() // scales.numel()
        nibbles    = nibbles.view(-1, block_size)
        signs      = (nibbles >> 3).to(torch.int64)
        mag_idx    = (nibbles & 0x7).to(torch.int64)
        levels     = FP4Codebook.pos_levels.to(scales.device, torch.float32)
        q          = torch.where(signs.bool(), -levels[mag_idx], levels[mag_idx])
        return (q * scales[:, None].float()).view(-1)[:flat_n].reshape(orig_shape)

def unpack_nibbles(packed, count):
    flat = packed.reshape(-1)
    out  = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = (flat >> 4) & 0x0F
    out[1::2] = flat & 0x0F
    return out[:count]

def get_decoded_state_dict(payload_data, device):
    data = torch.load(io.BytesIO(payload_data), map_location=device)
    sd   = {}
    for name, rec in data["quantized"].items():
        if rec["weight_kind"] == "fp4_packed":
            nibs = unpack_nibbles(rec["packed_weight"].to(device), rec["packed_weight"].numel() * 2)
            w    = FP4Codebook.dequantize_from_nibbles(nibs, rec["scales_fp16"].to(device), rec["weight_shape"])
        else:
            w = rec["weight_fp16"].to(device).float()
        sd[f"{name}.weight"] = w.float()
        if rec.get("bias_fp16") is not None:
            sd[f"{name}.bias"] = rec["bias_fp16"].to(device).float()
    for name, tensor in data["dense_fp16"].items():
        sd[name] = tensor.to(device).float() if torch.is_floating_point(tensor) else tensor.to(device)
    return sd


# ─── Architecture (inference-only) — must match compress.py exactly ──────────
class QConv2d(nn.Conv2d):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)

class QEmbedding(nn.Embedding):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)

class SepConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        mid = in_ch * depth_mult
        self.dw   = QConv2d(in_ch, mid, k, stride=stride, padding=k//2, groups=in_ch, bias=False)
        self.pw   = QConv2d(mid, out_ch, 1, bias=True)
        self.norm = nn.GroupNorm(2, out_ch)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.norm(self.pw(self.dw(x))))

class SepConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        mid = in_ch * depth_mult
        self.dw = QConv2d(in_ch, mid, k, stride=stride, padding=k//2, groups=in_ch, bias=False)
        self.pw = QConv2d(mid, out_ch, 1, bias=True)
    def forward(self, x): return self.pw(self.dw(x))

class SepResBlock(nn.Module):
    def __init__(self, ch, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, depth_mult=depth_mult)
        self.conv2 = SepConv(ch, ch, depth_mult=depth_mult)
        self.norm2 = nn.GroupNorm(2, ch)
        self.act   = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(x + self.norm2(self.conv2(self.conv1(x))))

class FiLMSepResBlock(nn.Module):
    def __init__(self, ch, cond_dim, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1     = SepConvGNAct(ch, ch, depth_mult=depth_mult)
        self.conv2     = SepConv(ch, ch, depth_mult=depth_mult)
        self.norm2     = nn.GroupNorm(2, ch)
        self.film_proj = nn.Linear(cond_dim, ch * 2)
        self.act       = nn.SiLU(inplace=True)
    def forward(self, x, cond):
        xb = self.norm2(self.conv2(self.conv1(x)))
        gamma, beta = self.film_proj(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return self.act(x + xb * (1.0 + gamma) + beta)

class SharedMaskDecoder(nn.Module):
    def __init__(self, num_classes=5, emb_dim=6, c1=64, c2=80, depth_mult=1):
        super().__init__()
        self.embedding  = QEmbedding(num_classes, emb_dim)
        self.stem_conv  = SepConvGNAct(emb_dim + 2, c1, depth_mult=depth_mult)
        self.stem_block = SepResBlock(c1, depth_mult=depth_mult)
        self.down_conv  = SepConvGNAct(c1, c2, stride=2, depth_mult=depth_mult)
        self.down_block = SepResBlock(c2, depth_mult=depth_mult)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            SepConvGNAct(c2, c1, depth_mult=depth_mult),
        )
        self.fuse       = SepConvGNAct(c1 + c1, c1, depth_mult=depth_mult)
        self.fuse_block = SepResBlock(c1, depth_mult=depth_mult)

    def forward(self, mask2, coords):
        e2    = self.embedding(mask2.long()).permute(0, 3, 1, 2)
        e2_up = F.interpolate(e2, size=coords.shape[-2:], mode="bilinear", align_corners=False)
        s     = self.stem_block(self.stem_conv(torch.cat([e2_up, coords], dim=1)))
        z     = self.up(self.down_block(self.down_conv(s)))
        return self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))

class Frame2StaticHead(nn.Module):
    def __init__(self, in_ch, hidden=64, depth_mult=1):
        super().__init__()
        self.block1 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre    = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head   = QConv2d(hidden, 3, 1)
    def forward(self, feat):
        return torch.sigmoid(self.head(self.pre(self.block2(self.block1(feat))))) * 255.0

class FrameHead(nn.Module):
    def __init__(self, in_ch, cond_dim=48, hidden=64, depth_mult=1):
        super().__init__()
        self.block1 = FiLMSepResBlock(in_ch, cond_dim, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre    = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head   = QConv2d(hidden, 3, 1)
    def forward(self, feat, cond):
        return torch.sigmoid(self.head(self.pre(self.block2(self.block1(feat, cond))))) * 255.0

class JointFrameGenerator(nn.Module):
    def __init__(self, num_classes=5, pose_dim=6, cond_dim=48, depth_mult=1):
        super().__init__()
        self.shared_trunk = SharedMaskDecoder(num_classes, emb_dim=6, c1=64, c2=80, depth_mult=depth_mult)
        self.pose_mlp     = nn.Sequential(nn.Linear(pose_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.frame1_head  = FrameHead(in_ch=64, cond_dim=cond_dim, hidden=64, depth_mult=depth_mult)
        self.frame2_head  = Frame2StaticHead(in_ch=64, hidden=64, depth_mult=depth_mult)

    def forward(self, mask2, pose6):
        b      = mask2.shape[0]
        coords = make_coord_grid(b, 384, 512, mask2.device, torch.float32)
        feat   = self.shared_trunk(mask2, coords)
        return self.frame1_head(feat, self.pose_mlp(pose6)), self.frame2_head(feat)

def make_coord_grid(batch, height, width, device, dtype):
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width,  device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx*2-1, yy*2-1], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_mask_video(path):
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="gray")
        frames.append(np.clip(np.round(img / 63.0).astype(np.uint8), 0, 4))
    container.close()
    return torch.from_numpy(np.stack(frames)).contiguous()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 4:
        print("Usage: python inflate.py <data_dir> <output_dir> <file_list_txt>")
        sys.exit(1)

    data_dir       = Path(sys.argv[1])
    out_dir        = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files  = [l.strip() for l in file_list_path.read_text().splitlines() if l.strip()]

    # 1. Load model
    with open(data_dir / "model.pt.br", "rb") as f:
        weights_data = brotli.decompress(f.read())
    generator = JointFrameGenerator().to(device)
    generator.load_state_dict(get_decoded_state_dict(weights_data, device), strict=True)
    generator.eval()

    # 2. Load mask video (lossless AV1, brotli-wrapped)
    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
        tmp.write(brotli.decompress(open(data_dir / "mask.obu.br", "rb").read()))
        tmp_path = tmp.name
    mask_frames_all = load_mask_video(tmp_path)
    os.remove(tmp_path)

    # 3. Load poses
    pose_bytes      = brotli.decompress(open(data_dir / "pose.npy.br", "rb").read())
    pose_frames_all = torch.from_numpy(np.load(io.BytesIO(pose_bytes))).float()

    out_h, out_w   = 874, 1164
    batch_size     = 4
    pairs_per_file = 600
    cursor         = 0

    with torch.inference_mode():
        for file_name in files:
            base_name  = os.path.splitext(file_name)[0]
            raw_out    = out_dir / f"{base_name}.raw"
            file_masks = mask_frames_all[cursor : cursor + pairs_per_file]
            file_poses = pose_frames_all[cursor : cursor + pairs_per_file]
            cursor    += pairs_per_file

            with open(raw_out, "wb") as f_out:
                for i in tqdm(range(0, file_masks.shape[0], batch_size), desc=f"Decoding {file_name}"):
                    m = file_masks[i:i+batch_size].to(device).long()
                    p = file_poses[i:i+batch_size].to(device).float()

                    f1, f2 = generator(m, p)
                    f1_up  = F.interpolate(f1, (out_h, out_w), mode="bilinear", align_corners=False)
                    f2_up  = F.interpolate(f2, (out_h, out_w), mode="bilinear", align_corners=False)

                    batch_comp = einops.rearrange(
                        torch.stack([f1_up, f2_up], dim=1),
                        "b t c h w -> (b t) h w c"
                    )
                    f_out.write(batch_comp.clamp(0, 255).round().to(torch.uint8).cpu().numpy().tobytes())

if __name__ == "__main__":
    main()
