#!/usr/bin/env python
"""
another_attempt/compress.py

Improvements over quantizr:
  1. Lossless AV1 for mask video: 5-class content (only 5 distinct pixel values)
     compresses more efficiently losslessly than at CRF=50. No quantization noise
     means exact class values, eliminating the train/inference mask mismatch.
  2. Wider generator: c1=64, c2=80 (vs c1=56, c2=64) — better quality at low extra cost.
  3. SegNet loss for frame1 added to FINETUNE stage (quantizr drops it in run5).
  4. Cleaner training schedule: larger QAT phase, better error boost ramp.
"""
import os
import sys
import math
import mmap
import argparse
import av
import subprocess
import shutil
import numpy as np
import logging
import warnings
import brotli
import io
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

# Performance optimizations for high-end GPUs (e.g. RTX 3090, 4090, 5090)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.set_float32_matmul_precision('high')

import einops
from tqdm import tqdm
from safetensors.torch import load_file

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from frame_utils import AVVideoDataset, DaliVideoDataset, segnet_model_input_size
from modules import SegNet, PoseNet, DistortionNet, segnet_sd_path, posenet_sd_path

SEQ_LEN = 2
SEGNET_MODEL_INPUT_SIZE = (512, 384)  # (W, H) — masks stored at this resolution

class Stage(Enum):
    ANCHOR  = "anchor"
    FINETUNE = "finetune"
    JOINT   = "joint"

@dataclass
class PipelineRun:
    name: str
    stage: Stage
    epochs: int
    lr: float
    qat_start_epoch: int
    frame1_fade_epochs: int = 0
    error_boost: float = 4.0
    ce_weight: float = 1.0
    pose_weight: float = 1.0
    warmup_epochs: int = 2
    ema_decay: float = 0.99
    grad_clip: float = 1.0
    frame1_seg_weight: float = 0.0  # explicit frame1 segnet weight (new vs quantizr)

def get_ffmpeg_path():
    local = ROOT_DIR / "ffmpeg"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local.resolve())
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    raise FileNotFoundError("FFmpeg binary not found.")

def diff_round(x):
    return x + (x.round() - x).detach()

# ─── EMA ────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1 - self.decay) * p.data + self.decay * self.shadow[n]

    def apply_shadow(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data
                p.data = self.shadow[n]

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]
        self.backup = {}

# ─── YUV helpers ────────────────────────────────────────────────────────────
def diff_rgb_to_yuv6(rgb_chw):
    h, w = rgb_chw.shape[-2:]
    h2, w2 = h // 2, w // 2
    rgb = rgb_chw[..., :2*h2, :2*w2]
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = (0.299*r + 0.587*g + 0.114*b).clamp(0, 255)
    u = ((b - y) / 1.772 + 128).clamp(0, 255)
    v = ((r - y) / 1.402 + 128).clamp(0, 255)
    y00, y10 = y[:, 0::2, 0::2], y[:, 1::2, 0::2]
    y01, y11 = y[:, 0::2, 1::2], y[:, 1::2, 1::2]
    u_s = (u[:, 0::2, 0::2] + u[:, 1::2, 0::2] + u[:, 0::2, 1::2] + u[:, 1::2, 1::2]) * 0.25
    v_s = (v[:, 0::2, 0::2] + v[:, 1::2, 0::2] + v[:, 0::2, 1::2] + v[:, 1::2, 1::2]) * 0.25
    return torch.stack([y00, y10, y01, y11, u_s, v_s], dim=1)

def pack_pair_yuv6(f1, f2):
    return torch.cat([diff_rgb_to_yuv6(f1), diff_rgb_to_yuv6(f2)], dim=1)

def get_pose_tensor(out):
    if isinstance(out, dict): return out["pose"]
    if hasattr(out, "pose"): return out.pose
    return out["pose"]

def make_coord_grid(batch, height, width, device, dtype):
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width,  device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx*2-1, yy*2-1], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)

def kl_on_logits(s_logits, t_logits, temperature=2.0):
    log_p = F.log_softmax(s_logits / temperature, dim=1)
    q     = F.softmax(t_logits / temperature, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature**2)

def assert_finite(name, x):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"non-finite in {name}: shape={tuple(x.shape)}")

# ─── Video preloading ────────────────────────────────────────────────────────
def hevc_frame_count(path):
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as b:
        frames, i = 0, 0
        while True:
            j = b.find(b"\x00\x00\x01", i)
            if j < 0: return frames
            p = j + 3
            if ((b[p] >> 1) & 0x3F) <= 31: frames += 1
            i = p

def container_frame_count(path):
    c = av.open(path)
    s = c.streams.video[0]
    n = s.frames or sum(1 for pkt in c.demux(s) if pkt.size > 0)
    c.close()
    return n

def preload_video_pair_cache_dali(file_names, data_dir, batch_size, device, num_threads=4, prefetch_queue_depth=4):
    logging.info("Preloading RGB pairs via DALI...")
    import nvidia.dali.fn as fn
    from nvidia.dali import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    warnings.filterwarnings("ignore", category=Warning, module=r"nvidia\.dali\.plugin\.base_iterator")

    @pipeline_def
    def pipe():
        return fn.experimental.inputs.video(name="inbuf", sequence_length=SEQ_LEN,
                                             device="mixed", no_copy=True, blocking=False,
                                             last_sequence_policy="pad")

    all_batches = []
    for fnm in file_names:
        path = str(data_dir / fnm)
        f  = open(path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)
        fc = hevc_frame_count(path) if path.endswith(".hevc") else container_frame_count(path)
        it_size = math.ceil((fc // SEQ_LEN) / batch_size)
        p = pipe(batch_size=batch_size, num_threads=num_threads,
                 device_id=device.index or 0, prefetch_queue_depth=prefetch_queue_depth)
        p.build()
        p.feed_input("inbuf", [mv])
        it = DALIGenericIterator([p], output_map=["video"], auto_reset=False,
                                 last_batch_policy=LastBatchPolicy.PARTIAL)
        try:
            for _ in range(it_size):
                all_batches.append(next(it)[0]["video"].cpu().contiguous())
        finally:
            torch.cuda.synchronize()
            it.reset(); del it, p; mv.release(); mm.close(); f.close()

    if not all_batches:
        raise RuntimeError("No video data loaded.")
    return torch.cat(all_batches, dim=0).contiguous()

# ─── Mask extraction & compression ──────────────────────────────────────────
def extract_and_compress_masks(rgb_pairs_all, segnet, device, archive_dir, batch_size=8):
    """
    Extract segmentation masks for odd frames (frame2 of each pair),
    compress them with LOSSLESS AV1.

    Lossless is better than CRF=50 for 5-class content because:
      - Only 5 distinct pixel values {0, 63, 126, 189, 252}
      - Exact values compress better (residual=0 for unchanged blocks)
      - No quantization noise → no train/inference mismatch
    """
    expected = rgb_pairs_all.shape[0]
    raw_path  = archive_dir / "raw_masks.yuv"
    obu_path  = archive_dir / "mask.obu"
    obu_br    = archive_dir / "mask.obu.br"

    if obu_br.exists():
        logging.info("Cached mask found. Verifying...")
        try:
            with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
                tmp.write(brotli.decompress(open(obu_br, "rb").read()))
                tmp_path = tmp.name
            c = av.open(tmp_path)
            frames = []
            for frame in c.decode(video=0):
                img = frame.to_ndarray(format="gray")
                frames.append(np.clip(np.round(img / 63.0).astype(np.uint8), 0, 4))
            c.close()
            os.remove(tmp_path)
            if len(frames) == expected:
                logging.info(f"Mask cache valid ({len(frames)} frames).")
                return torch.from_numpy(np.stack(frames)).contiguous()
            logging.warning(f"Mask cache incomplete ({len(frames)}/{expected}). Regenerating...")
        except Exception as e:
            logging.warning(f"Mask cache load failed ({e}). Regenerating...")

    logging.info("Generating masks from RGB pairs...")
    W, H = SEGNET_MODEL_INPUT_SIZE
    with open(raw_path, "wb") as f_out:
        with torch.inference_mode():
            for start in tqdm(range(0, expected, batch_size), desc="Masks"):
                batch = rgb_pairs_all[start:start+batch_size].to(device).float()
                batch = einops.rearrange(batch, "b t h w c -> b t c h w")
                frames = batch[:, 1]  # odd frame (frame2)
                resized = F.interpolate(frames, size=(H, W), mode="bilinear")
                mask = segnet(resized).argmax(dim=1).to(torch.uint8)
                f_out.write((mask * 63).cpu().numpy().tobytes())

    logging.info("Compressing masks with lossless AV1...")
    cmd = [
        get_ffmpeg_path(), "-y", "-hide_banner",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{W}x{H}", "-r", "10",
        "-i", str(raw_path),
        "-c:v", "libaom-av1",
        "-crf", "0", "-b:v", "0",
        "-cpu-used", "4",
        "-row-mt", "1",
        "-g", "600",
        "-keyint_min", "600",
        "-lag-in-frames", "48",
        "-aom-params", "lossless=1:enable-cdef=0:enable-intrabc=1:enable-obmc=0:enable-palette=1",
        "-f", "obu",
        str(obu_path),
    ]
    subprocess.run(cmd, check=True)

    logging.info("Brotli compressing OBU...")
    with open(obu_path, "rb") as fi, open(obu_br, "wb") as fo:
        fo.write(brotli.compress(fi.read(), quality=11, lgwin=24))

    c = av.open(str(obu_path))
    frames = []
    for frame in c.decode(video=0):
        img = frame.to_ndarray(format="gray")
        frames.append(np.clip(np.round(img / 63.0).astype(np.uint8), 0, 4))
    c.close()

    if len(frames) != expected:
        raise RuntimeError(f"Mask encode failed: got {len(frames)}, expected {expected}.")

    obu_path.unlink()
    raw_path.unlink()
    return torch.from_numpy(np.stack(frames)).contiguous()

def extract_and_compress_poses(rgb_pairs_all, posenet, device, archive_dir, batch_size=8):
    br_path = archive_dir / "pose.npy.br"
    all_pose6 = []
    logging.info("Extracting poses...")
    with torch.inference_mode():
        for start in tqdm(range(0, rgb_pairs_all.shape[0], batch_size), desc="Poses"):
            batch = rgb_pairs_all[start:start+batch_size].to(device).float()
            batch = einops.rearrange(batch, "b t h w c -> b t c h w")
            out   = posenet(posenet.preprocess_input(batch))
            all_pose6.append(get_pose_tensor(out)[..., :6].float().cpu().numpy())

    pose_arr = np.concatenate(all_pose6, axis=0)
    buf = io.BytesIO()
    np.save(buf, pose_arr)
    buf.seek(0)
    with open(br_path, "wb") as f:
        f.write(brotli.compress(buf.read(), quality=11, lgwin=24))
    return torch.from_numpy(pose_arr).float().contiguous()

# ─── Data loader ─────────────────────────────────────────────────────────────
class CachedPairLoader:
    def __init__(self, rgb_pairs, mask2, pose6, batch_size, device, seed=123, shuffle=True):
        self.rgb_pairs  = rgb_pairs.contiguous()
        self.mask2      = mask2.contiguous()
        self.pose6      = pose6.contiguous()
        self.batch_size = batch_size
        self.device     = device
        self.seed, self.shuffle = seed, shuffle
        self.epoch, self.num_samples = 0, rgb_pairs.shape[0]

    def set_epoch(self, e): self.epoch = int(e)
    def __len__(self): return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.num_samples, generator=g) if self.shuffle else torch.arange(self.num_samples)
        for start in range(0, self.num_samples, self.batch_size):
            idx = perm[start:start+self.batch_size]
            yield (
                self.rgb_pairs.index_select(0, idx).to(self.device, non_blocking=True),
                self.mask2.index_select(0, idx).to(self.device, non_blocking=True),
                self.pose6.index_select(0, idx).to(self.device, non_blocking=True),
            )

# ─── FP4 quantization ────────────────────────────────────────────────────────
class FP4Codebook:
    pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

    @staticmethod
    def quantize_blockwise(x, block_size=32):
        orig = x.shape
        flat = x.reshape(-1)
        pad  = (block_size - flat.numel() % block_size) % block_size
        if pad: flat = F.pad(flat, (0, pad))
        blocks  = flat.view(-1, block_size)
        max_abs = blocks.abs().amax(dim=1, keepdim=True)
        scales  = torch.where(max_abs > 0, max_abs / 6.0, torch.ones_like(max_abs))
        norm    = blocks / scales
        signs   = (norm < 0).to(torch.int16)
        levels  = FP4Codebook.pos_levels.to(x.device, x.dtype).view(1, 1, -1)
        mag_idx = (norm.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.int16)
        q = torch.where(signs.bool(), -levels[0, 0, mag_idx.long()], levels[0, 0, mag_idx.long()])
        return (q * scales).view(-1)[:x.numel()].view(orig), ((signs << 3) | mag_idx).to(torch.uint8), scales.squeeze(1)

    @staticmethod
    def dequantize_from_nibbles(nibbles, scales, orig_shape):
        flat_n  = int(torch.tensor(orig_shape).prod().item())
        nibbles = nibbles.view(-1, nibbles.numel() // scales.numel())
        signs, mag_idx = (nibbles >> 3).to(torch.int64), (nibbles & 0x7).to(torch.int64)
        levels  = FP4Codebook.pos_levels.to(scales.device, torch.float32)
        q       = torch.where(signs.bool(), -levels[mag_idx], levels[mag_idx])
        return (q * scales[:, None].float()).view(-1)[:flat_n].reshape(orig_shape)

def fake_quant_fp4_ste(x, block_size=32):
    dq, _, _ = FP4Codebook.quantize_blockwise(x, block_size)
    return x + (dq - x).detach()

def pack_nibbles(nib):
    flat = nib.reshape(-1)
    if flat.numel() % 2: flat = F.pad(flat, (0, 1))
    return ((flat[0::2] & 0x0F) << 4) | (flat[1::2] & 0x0F)

def unpack_nibbles(packed, count):
    flat = packed.reshape(-1)
    out  = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2], out[1::2] = (flat >> 4) & 0x0F, flat & 0x0F
    return out[:count]

def load_fp4_state_dict(model, fp4_path, device):
    data = torch.load(fp4_path, map_location=device)
    state_dict = {}
    for name, rec in data.get("quantized", {}).items():
        if rec["weight_kind"] == "fp4_packed":
            nibs = unpack_nibbles(rec["packed_weight"].to(device), rec["packed_weight"].numel() * 2)
            w    = FP4Codebook.dequantize_from_nibbles(nibs, rec["scales_fp16"].to(device), rec["weight_shape"])
        else:
            w = rec["weight_fp16"].to(device).float()
        state_dict[f"{name}.weight"] = w
        if rec.get("bias_fp16") is not None:
            state_dict[f"{name}.bias"] = rec["bias_fp16"].to(device).float()
    for k, v in data.get("dense_fp16", {}).items():
        state_dict[k] = v.to(device).float() if torch.is_floating_point(v) else v.to(device)
    model.load_state_dict(state_dict, strict=False)

def export_fp4_state_dict(model, out_path, block_size=32):
    export = {
        "__format__": "fp4_standalone", "__block_size__": block_size,
        "__codebook__": FP4Codebook.pos_levels.clone(),
        "quantized": {}, "dense_fp16": {},
    }
    covered = set()
    for name, m in model.named_modules():
        if isinstance(m, (QConv2d, QEmbedding)):
            rec = {"type": "conv2d" if isinstance(m, QConv2d) else "embedding"}
            w   = m.weight.detach().float().cpu()
            rec["weight_shape"] = list(w.shape)
            covered.add(f"{name}.weight")
            if isinstance(m, QConv2d):
                rec["stride"]   = list(m.stride)   if isinstance(m.stride, tuple)   else [m.stride]*2
                rec["padding"]  = list(m.padding)  if isinstance(m.padding, tuple)  else [m.padding]*2
                rec["dilation"] = list(m.dilation) if isinstance(m.dilation, tuple) else [m.dilation]*2
                rec["groups"]   = int(m.groups)
                rec["bias_fp16"] = m.bias.detach().half().cpu() if m.bias is not None else None
                if m.bias is not None: covered.add(f"{name}.bias")
            if getattr(m, "quantize_weight", False):
                _, nib, scales = FP4Codebook.quantize_blockwise(w, block_size)
                rec.update({"weight_kind": "fp4_packed", "weight_numel": int(w.numel()),
                            "packed_weight": pack_nibbles(nib.cpu()), "scales_fp16": scales.half().cpu()})
            else:
                rec.update({"weight_kind": "fp16", "weight_fp16": w.half().cpu()})
            export["quantized"][name] = rec
    for k, v in model.state_dict().items():
        if k not in covered:
            export["dense_fp16"][k] = v.detach().cpu().half() if torch.is_floating_point(v) else v.detach().cpu()
    torch.save(export, out_path, _use_new_zipfile_serialization=False)

# ─── Quantizable modules ──────────────────────────────────────────────────────
class QMixin:
    def set_qat(self, enabled, act_enabled=False):
        self.qat_enabled = enabled
        self.qat_act_enabled = act_enabled

class QConv2d(nn.Conv2d, QMixin):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size, self.quantize_weight, self.qat_enabled = block_size, quantize_weight, False

    def forward(self, x):
        w = fake_quant_fp4_ste(self.weight, self.block_size) if self.qat_enabled and self.quantize_weight else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QEmbedding(nn.Embedding, QMixin):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size, self.quantize_weight, self.qat_enabled = block_size, quantize_weight, False

    def forward(self, x):
        w = fake_quant_fp4_ste(self.weight, self.block_size) if self.qat_enabled and self.quantize_weight else self.weight
        return F.embedding(x, w, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

# ─── Architecture — wider channels (c1=64, c2=80) ────────────────────────────
class SepConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        mid = in_ch * depth_mult
        self.dw   = QConv2d(in_ch, mid, k, stride=stride, padding=k//2, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw   = QConv2d(mid, out_ch, 1, bias=True, quantize_weight=quantize_weight)
        self.norm = nn.GroupNorm(2, out_ch)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.norm(self.pw(self.dw(x))))

class SepConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        mid = in_ch * depth_mult
        self.dw = QConv2d(in_ch, mid, k, stride=stride, padding=k//2, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid, out_ch, 1, bias=True, quantize_weight=quantize_weight)
    def forward(self, x): return self.pw(self.dw(x))

class SepResBlock(nn.Module):
    def __init__(self, ch, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch)
        self.act   = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(x + self.norm2(self.conv2(self.conv1(x))))

class FiLMSepResBlock(nn.Module):
    def __init__(self, ch, cond_dim, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1     = SepConvGNAct(ch, ch, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2     = SepConv(ch, ch, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2     = nn.GroupNorm(2, ch)
        self.film_proj = nn.Linear(cond_dim, ch * 2)
        self.act       = nn.SiLU(inplace=True)
    def forward(self, x, cond):
        xb = self.norm2(self.conv2(self.conv1(x)))
        gamma, beta = self.film_proj(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return self.act(x + xb * (1.0 + gamma) + beta)

class SharedMaskDecoder(nn.Module):
    """Wider than quantizr: c1=64, c2=80 (was c1=56, c2=64)."""
    def __init__(self, num_classes=5, emb_dim=6, c1=64, c2=80, depth_mult=1):
        super().__init__()
        self.embedding  = QEmbedding(num_classes, emb_dim, quantize_weight=False)
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
        e2     = self.embedding(mask2.long()).permute(0, 3, 1, 2)
        e2_up  = F.interpolate(e2, size=coords.shape[-2:], mode="bilinear", align_corners=False)
        s      = self.stem_block(self.stem_conv(torch.cat([e2_up, coords], dim=1)))
        z      = self.up(self.down_block(self.down_conv(s)))
        return self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))

class Frame2StaticHead(nn.Module):
    def __init__(self, in_ch, hidden=64, depth_mult=1):
        super().__init__()
        self.block1 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre    = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head   = QConv2d(hidden, 3, 1, quantize_weight=False)
    def forward(self, feat):
        return torch.sigmoid(self.head(self.pre(self.block2(self.block1(feat))))) * 255.0

class FrameHead(nn.Module):
    def __init__(self, in_ch, cond_dim=48, hidden=64, depth_mult=1):
        super().__init__()
        self.block1 = FiLMSepResBlock(in_ch, cond_dim, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre    = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head   = QConv2d(hidden, 3, 1, quantize_weight=False)
    def forward(self, feat, cond):
        return torch.sigmoid(self.head(self.pre(self.block2(self.block1(feat, cond))))) * 255.0

class JointFrameGenerator(nn.Module):
    def __init__(self, num_classes=5, pose_dim=6, cond_dim=48, depth_mult=1):
        super().__init__()
        self.shared_trunk = SharedMaskDecoder(num_classes, emb_dim=6, c1=64, c2=80, depth_mult=depth_mult)
        self.pose_mlp     = nn.Sequential(nn.Linear(pose_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.frame1_head  = FrameHead(in_ch=64, cond_dim=cond_dim, hidden=64, depth_mult=depth_mult)
        self.frame2_head  = Frame2StaticHead(in_ch=64, hidden=64, depth_mult=depth_mult)

    def set_qat(self, enabled):
        for m in self.modules():
            if isinstance(m, (QConv2d, QEmbedding)): m.set_qat(enabled=enabled)

    def forward(self, mask2, pose6):
        coords      = make_coord_grid(mask2.shape[0], 384, 512, mask2.device, torch.float32)
        shared_feat = self.shared_trunk(mask2, coords)
        return self.frame1_head(shared_feat, self.pose_mlp(pose6)), self.frame2_head(shared_feat)

# ─── Training ─────────────────────────────────────────────────────────────────
def apply_freeze_state(model, stage):
    for p in model.parameters(): p.requires_grad = True
    if stage == Stage.ANCHOR:
        for p in model.frame1_head.parameters(): p.requires_grad = False
        for p in model.pose_mlp.parameters():    p.requires_grad = False
    elif stage == Stage.FINETUNE:
        for p in model.shared_trunk.parameters(): p.requires_grad = False
        for p in model.frame2_head.parameters():  p.requires_grad = False
    if stage == Stage.FINETUNE:
        model.shared_trunk.eval()
        model.frame2_head.eval()

def train_run(run, generator, loader, device, archive_dir, aux_models, state_dict_to_load=None):
    segnet, posenet, dist_net = aux_models
    apply_freeze_state(generator, run.stage)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=run.lr, betas=(0.9, 0.99)
    )
    start_epoch, best_metric = 0, float("inf")

    latest_path = archive_dir / f"{run.name}_latest.pt"
    if latest_path.exists():
        logging.info(f"Resuming {run.name}...")
        ckpt = torch.load(latest_path, map_location=device)
        generator.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch  = ckpt["epoch"] + 1
        best_metric  = ckpt["best_metric"]
    elif state_dict_to_load is not None:
        generator.load_state_dict(state_dict_to_load)

    ema = EMA(generator, decay=run.ema_decay) if run.ema_decay > 0 else None
    if ema and latest_path.exists() and ckpt.get("ema_state"):
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema_state"].items()}

    qat_warmup = min(run.warmup_epochs, max(1, (run.epochs - run.qat_start_epoch) // 2)) \
                 if run.qat_start_epoch == 0 else run.warmup_epochs
    warmup_sch = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, qat_warmup)
    main_sch   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, run.epochs - qat_warmup))
    scheduler  = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_sch, main_sch], [qat_warmup])
    for _ in range(start_epoch): scheduler.step()

    for epoch in range(start_epoch, run.epochs):
        generator.train()
        if run.stage == Stage.FINETUNE:
            generator.shared_trunk.eval(); generator.frame2_head.eval()

        loader.set_epoch(epoch)
        qat_on = epoch >= run.qat_start_epoch
        generator.set_qat(qat_on)

        if epoch == run.qat_start_epoch and run.qat_start_epoch > 0:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, generator.parameters()),
                lr=run.lr, betas=(0.9, 0.99)
            )
            warmup_sch = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, qat_warmup)
            main_sch   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, (run.epochs - epoch) - qat_warmup))
            scheduler  = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_sch, main_sch], [qat_warmup])

        kl_alpha   = min(1.0, epoch / max(1, run.qat_start_epoch // 2)) if run.qat_start_epoch > 0 else 1.0
        seg2_kl_w  = 0.9 - 0.9 * kl_alpha
        seg2_ce_w  = 0.1 + 0.9 * kl_alpha
        frame1_sem_w = max(0.0, 1.0 - epoch / run.frame1_fade_epochs) if run.frame1_fade_epochs > 0 else 0.0

        totals = dict(loss=0.0, s2=0.0, s1=0.0, pose=0.0)
        batches = 0

        pbar = tqdm(loader, desc=f"{run.name} Ep{epoch+1}/{run.epochs}", leave=False)
        for batch_rgb, in_mask2, in_pose6 in pbar:
            batch    = einops.rearrange(batch_rgb, "b t h w c -> b t c h w").float().to(device)
            in_mask2 = in_mask2.to(device).long()
            in_pose6 = in_pose6.to(device).float()

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                real1 = F.interpolate(batch[:, 0], (384, 512), mode="bilinear", align_corners=False)
                real2 = F.interpolate(batch[:, 1], (384, 512), mode="bilinear", align_corners=False)
                gt_logits1, gt_logits2 = segnet(real1).float(), segnet(real2).float()
                gt_mask1, gt_mask2     = gt_logits1.argmax(1), gt_logits2.argmax(1)
                gt_pose = get_pose_tensor(posenet(posenet.preprocess_input(batch))).float()[..., :6]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred1, pred2 = generator(in_mask2, in_pose6)

                up1 = F.interpolate(pred1, (874, 1164), mode="bilinear", align_corners=False)
                up2 = F.interpolate(pred2, (874, 1164), mode="bilinear", align_corners=False)
                dn1 = F.interpolate(diff_round(up1.clamp(0, 255)), (384, 512), mode="bilinear", align_corners=False)
                dn2 = F.interpolate(diff_round(up2.clamp(0, 255)), (384, 512), mode="bilinear", align_corners=False)

                zero = torch.tensor(0.0, device=device)
                loss_pose = loss_seg2 = loss_seg1 = loss_seg2_ce = loss_seg1_ce = zero

                if run.stage in [Stage.FINETUNE, Stage.JOINT]:
                    fake_pose = get_pose_tensor(posenet(pack_pair_yuv6(dn1, dn2).float())).float()[..., :6]
                    loss_pose = F.mse_loss(fake_pose, gt_pose)

                if run.stage in [Stage.ANCHOR, Stage.JOINT]:
                    logits2    = segnet(dn2).float()
                    ce2        = F.cross_entropy(logits2, gt_mask2, reduction="none")
                    with torch.no_grad():
                        boost2 = 1.0 + (logits2.argmax(1) != gt_mask2).float() * run.error_boost
                    loss_seg2_ce = (ce2 * boost2).mean()
                    kl2          = kl_on_logits(logits2, gt_logits2) / (384 * 512)
                    loss_seg2    = 100.0 * (seg2_kl_w * kl2 + seg2_ce_w * 0.5 * run.ce_weight * loss_seg2_ce)

                # Frame1 segnet loss: applied via fade schedule OR explicit weight (new)
                do_seg1_fade  = frame1_sem_w > 0
                do_seg1_fixed = run.frame1_seg_weight > 0
                if do_seg1_fade or do_seg1_fixed:
                    logits1    = segnet(dn1).float()
                    ce1        = F.cross_entropy(logits1, gt_mask1, reduction="none")
                    with torch.no_grad():
                        boost1 = 1.0 + (logits1.argmax(1) != gt_mask1).float() * run.error_boost
                    loss_seg1_ce = (ce1 * boost1).mean()
                    eff_w = frame1_sem_w if do_seg1_fade else run.frame1_seg_weight
                    if run.stage == Stage.JOINT:
                        kl1       = kl_on_logits(logits1, gt_logits1) / (384 * 512)
                        loss_seg1 = 100.0 * eff_w * (seg2_kl_w * kl1 + seg2_ce_w * 0.5 * run.ce_weight * loss_seg1_ce)
                    else:
                        loss_seg1 = 100.0 * eff_w * run.ce_weight * loss_seg1_ce

                if run.stage == Stage.ANCHOR:
                    loss = loss_seg2
                elif run.stage == Stage.FINETUNE:
                    loss = loss_seg1 + run.pose_weight * loss_pose * 10.0
                else:  # JOINT
                    loss = loss_seg2 + loss_seg1 + 30.0 * run.pose_weight * loss_pose

            assert_finite("loss", loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), run.grad_clip)
            optimizer.step()

            if ema and epoch >= run.warmup_epochs: ema.update(generator)

            totals["loss"] += loss.item()
            totals["s2"]   += loss_seg2_ce.item()
            totals["s1"]   += loss_seg1_ce.item()
            totals["pose"] += loss_pose.item()
            batches += 1
            pbar.set_postfix(L=f"{loss.item():.2f}", S2=f"{loss_seg2_ce.item():.2f}", P=f"{loss_pose.item():.4f}")

        scheduler.step()
        n = max(1, batches)
        logging.info(f"Ep{epoch+1} {'[QAT]' if qat_on else ''} "
                     f"loss={totals['loss']/n:.4f} s2={totals['s2']/n:.4f} "
                     f"s1={totals['s1']/n:.4f} pose={totals['pose']/n:.6f}")

        eval_epoch = qat_on and (((epoch - run.qat_start_epoch) % 5 == 0) or run.epochs - epoch <= 10)
        if eval_epoch:
            if ema: ema.apply_shadow(generator)
            generator.eval()
            total_seg, total_pose, n_samples = 0.0, 0.0, 0
            with torch.inference_mode():
                for batch_rgb, in_mask2, in_pose6 in tqdm(loader, desc="Eval", leave=False):
                    p1, p2 = generator(in_mask2.to(device).long(), in_pose6.to(device).float())
                    b_comp = torch.stack([
                        F.interpolate(p1, (874, 1164), mode="bilinear", align_corners=False),
                        F.interpolate(p2, (874, 1164), mode="bilinear", align_corners=False),
                    ], dim=1)
                    b_comp = einops.rearrange(b_comp, "b t c h w -> b t h w c").clamp(0, 255).round().to(torch.uint8)
                    pd, sd = dist_net.compute_distortion(batch_rgb.to(device), b_comp)
                    total_seg  += sd.sum().item()
                    total_pose += pd.sum().item()
                    n_samples  += batch_rgb.shape[0]

            avg_seg  = total_seg  / max(1, n_samples)
            avg_pose = total_pose / max(1, n_samples)
            model_sz  = (archive_dir / "model.pt.br").stat().st_size if (archive_dir / "model.pt.br").exists() else 1_500_000
            mask_sz   = (archive_dir / "mask.obu.br").stat().st_size if (archive_dir / "mask.obu.br").exists() else 0
            pose_sz   = (archive_dir / "pose.npy.br").stat().st_size if (archive_dir / "pose.npy.br").exists() else 0
            total_bytes = model_sz + mask_sz + pose_sz
            total_pixels = n_samples * 2 * 1164 * 874
            rate_bpp = (total_bytes * 8.0) / max(1, total_pixels)

            scaled_seg  = 100.0 * avg_seg
            scaled_pose = math.sqrt(max(0, 10.0 * avg_pose))
            scaled_rate = 25.0 * rate_bpp

            eval_metric = scaled_seg + scaled_pose + scaled_rate
            logging.info(f"  [Eval] Score~{eval_metric:.4f} | "
                         f"Seg={scaled_seg:.4f} Pose={scaled_pose:.4f} Rate={scaled_rate:.4f}")

            if eval_metric < best_metric:
                best_metric = eval_metric
                best_fp16 = {k: v.half() if torch.is_floating_point(v) else v
                             for k, v in generator.state_dict().items()}
                torch.save(best_fp16, archive_dir / f"{run.name}_best_fp16.pt")
                fp4_path = archive_dir / f"{run.name}_best_fp4.pt"
                export_fp4_state_dict(generator.cpu(), fp4_path)
                generator.to(device)
                with open(fp4_path, "rb") as fi:
                    comp = brotli.compress(fi.read(), quality=11, lgwin=24)
                with open(archive_dir / f"{run.name}_best_fp4.pt.br", "wb") as fo:
                    fo.write(comp)
                shutil.copyfile(archive_dir / f"{run.name}_best_fp4.pt.br", archive_dir / "model.pt.br")
                logging.info(f"  *** New best: {best_metric:.5f} → model.pt.br ***")

            if ema: ema.restore(generator)

        torch.save({
            "epoch": epoch, "best_metric": best_metric,
            "model_state": generator.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "ema_state": {k: v.cpu() for k, v in ema.shadow.items()} if ema else None,
        }, latest_path)

    if latest_path.exists(): latest_path.unlink()
    return _load_best_fp4(generator, archive_dir / f"{run.name}_best_fp4.pt", device)

def _load_best_fp4(model, path, device):
    load_fp4_state_dict(model, path, device)
    model.float()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

# ─── Argument parsing & main ─────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir",   type=Path, default=ROOT_DIR / "videos")
    p.add_argument("--video-names", type=Path, default=ROOT_DIR / "public_test_video_names.txt")
    p.add_argument("--batch-size",  type=int,  default=4)
    p.add_argument("--device",      type=str,  default="cuda:0")
    return p.parse_args()

def main():
    args   = parse_args()
    device = torch.device(args.device)

    archive_dir = Path(__file__).parent / "archive"
    archive_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(archive_dir / "pipeline.log")]
    )
    logging.info("=== another_attempt compression pipeline ===")

    segnet  = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    dist_net = DistortionNet().eval().to(device)
    dist_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for m in (segnet, posenet, dist_net):
        for p in m.parameters(): p.requires_grad = False

    files = [l.strip() for l in args.video_names.read_text().splitlines() if l.strip()]
    rgb_pairs_all = preload_video_pair_cache_dali(files, args.video_dir, args.batch_size, device)

    mask_frames_all = extract_and_compress_masks(rgb_pairs_all, segnet, device, archive_dir)
    pose6_all       = extract_and_compress_poses(rgb_pairs_all, posenet, device, archive_dir)

    loader    = CachedPairLoader(rgb_pairs_all, mask_frames_all, pose6_all, args.batch_size, device)
    generator = JointFrameGenerator().to(device)

    # ── Training pipeline ──────────────────────────────────────────────────
    #
    # Changes vs quantizr:
    #   • ANCHOR: more QAT epochs (qat_start_epoch=150 vs 200)
    #   • FINETUNE (run3): frame1_seg_weight=0.3 — explicit segnet loss for frame1
    #                      instead of only pose loss (key new improvement)
    #   • FINETUNE (run5): frame1_seg_weight=0.3 — maintained in final micro stage
    #   • JOINT: more balanced loss with frame1 seg weight
    #
    PIPELINE = [
        PipelineRun("run1_anchor",  Stage.ANCHOR,   400, 5e-4, qat_start_epoch=150, frame1_fade_epochs=60,  error_boost=9.0),
        PipelineRun("run2_boost",   Stage.ANCHOR,    80, 1e-5, qat_start_epoch=0,   frame1_fade_epochs=0,   error_boost=49.0),
        PipelineRun("run3_finetune",Stage.FINETUNE, 320, 5e-5, qat_start_epoch=100, frame1_fade_epochs=60,  pose_weight=1.0,
                    frame1_seg_weight=0.3),  # segnet loss for frame1 in FINETUNE
        PipelineRun("run4_joint",   Stage.JOINT,    160, 1e-5, qat_start_epoch=0,   frame1_fade_epochs=40,  pose_weight=1.0),
        PipelineRun("run5_micro",   Stage.FINETUNE, 120, 5e-6, qat_start_epoch=0,   frame1_fade_epochs=0,   pose_weight=1.0,
                    frame1_seg_weight=0.3),  # maintained in final micro stage
    ]

    current_sd = None
    for run in PIPELINE:
        best_path   = archive_dir / f"{run.name}_best_fp4.pt"
        latest_path = archive_dir / f"{run.name}_latest.pt"
        if best_path.exists() and not latest_path.exists():
            logging.info(f"[SKIP] {run.name} already done, loading...")
            current_sd = _load_best_fp4(generator, best_path, device)
            continue
        logging.info(f"\n{'='*50}\nSTARTING: {run.name}\n{'='*50}")
        current_sd = train_run(run, generator, loader, device, archive_dir,
                               (segnet, posenet, dist_net), current_sd)

    logging.info("\nDone. Final model saved to archive/model.pt.br")

if __name__ == "__main__":
    main()
