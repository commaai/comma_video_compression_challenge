#!/usr/bin/env python
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
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from tqdm import tqdm
from safetensors.torch import load_file

# -----------------------------
# Path Resolution & Imports
# -----------------------------
# compress.py is in ./submissions/quantizr/, we need to access ./modules.py
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from frame_utils import AVVideoDataset, DaliVideoDataset, segnet_model_input_size
from modules import SegNet, PoseNet, DistortionNet, segnet_sd_path, posenet_sd_path

# -----------------------------
# Constants & Config
# -----------------------------
SEQ_LEN = 2
SEGNET_MODEL_INPUT_SIZE = (512, 384)

class Stage(Enum):
    ANCHOR = "anchor"     
    FINETUNE = "finetune" 
    JOINT = "joint"       

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

# -----------------------------
# System Helpers
# -----------------------------
def get_ffmpeg_path():
    """Find local ffmpeg binary first, fallback to system ffmpeg."""
    local_ffmpeg = ROOT_DIR / "ffmpeg"
    if local_ffmpeg.is_file() and os.access(local_ffmpeg, os.X_OK):
        return str(local_ffmpeg.resolve())
    
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
        
    raise FileNotFoundError("FFmpeg binary not found locally or in system PATH.")

def diff_round(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()

# -----------------------------
# EMA
# -----------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# -----------------------------
# Evaluator Helpers
# -----------------------------
def diff_rgb_to_yuv6(rgb_chw: torch.Tensor) -> torch.Tensor:
    h, w = rgb_chw.shape[-2:]
    h2, w2 = h // 2, w // 2
    rgb = rgb_chw[..., : 2 * h2, : 2 * w2]
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0)
    u = ((b - y) / 1.772 + 128.0).clamp(0.0, 255.0)
    v = ((r - y) / 1.402 + 128.0).clamp(0.0, 255.0)
    y00, y10, y01, y11 = y[:, 0::2, 0::2], y[:, 1::2, 0::2], y[:, 0::2, 1::2], y[:, 1::2, 1::2]
    u_sub = (u[:, 0::2, 0::2] + u[:, 1::2, 0::2] + u[:, 0::2, 1::2] + u[:, 1::2, 1::2]) * 0.25
    v_sub = (v[:, 0::2, 0::2] + v[:, 1::2, 0::2] + v[:, 0::2, 1::2] + v[:, 1::2, 1::2]) * 0.25
    return torch.stack([y00, y10, y01, y11, u_sub, v_sub], dim=1)

def pack_pair_yuv6(frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
    return torch.cat([diff_rgb_to_yuv6(frame1), diff_rgb_to_yuv6(frame2)], dim=1)

def get_pose_tensor(posenet_out):
    if isinstance(posenet_out, dict): return posenet_out["pose"]
    if hasattr(posenet_out, "pose"): return posenet_out.pose
    return posenet_out["pose"]

def make_coord_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)

def kl_on_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    log_p = F.log_softmax(student_logits / temperature, dim=1)
    q = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)

def assert_finite(name: str, x: torch.Tensor):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"non-finite tensor in {name}: shape={tuple(x.shape)}")

# -----------------------------
# Data Extractors & Preloaders
# -----------------------------
def hevc_frame_count(path: str) -> int:
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as b:
        frames, i = 0, 0
        find = b.find
        while True:
            j = find(b"\x00\x00\x01", i)
            if j < 0: return frames
            p = j + 3
            if ((b[p] >> 1) & 0x3F) <= 31: frames += 1
            i = p

def container_frame_count(path: str) -> int:
    container = av.open(path)
    stream = container.streams.video[0]
    n = stream.frames
    # If the container header lacks the frame count, demux and count manually
    if n == 0: 
        n = sum(1 for packet in container.demux(stream) if packet.size > 0)
    container.close()
    return n

def preload_video_pair_cache_dali(file_names, data_dir, batch_size, device, num_threads=4, prefetch_queue_depth=4):
    logging.info("Preloading raw video RGB pairs into memory via DALI...")
    import nvidia.dali.fn as fn
    from nvidia.dali import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    warnings.filterwarnings("ignore", category=Warning, module=r"nvidia\.dali\.plugin\.base_iterator")

    @pipeline_def
    def pipe():
        return fn.experimental.inputs.video(name="inbuf", sequence_length=SEQ_LEN, device="mixed", no_copy=True, blocking=False, last_sequence_policy="pad")

    all_batches = []
    for fnm in file_names:
        path = str(data_dir / fnm)
        f = open(path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)
        
        # Use the robust frame counter
        frames_count = hevc_frame_count(path) if path.endswith(".hevc") else container_frame_count(path)
        
        it_size = math.ceil((frames_count // SEQ_LEN) / batch_size)
        
        p = pipe(batch_size=batch_size, num_threads=num_threads, device_id=device.index or 0, prefetch_queue_depth=prefetch_queue_depth)
        p.build()
        p.feed_input("inbuf", [mv])
        it = DALIGenericIterator([p], output_map=["video"], auto_reset=False, last_batch_policy=LastBatchPolicy.PARTIAL)
        try:
            for _ in range(it_size): 
                all_batches.append(next(it)[0]["video"].cpu().contiguous())
        finally:
            torch.cuda.synchronize()
            it.reset(); del it, p; mv.release(); mm.close(); f.close()
            
    if not all_batches:
        raise RuntimeError("No video data was loaded. Please check if your video directory and file list are correct.")
        
    return torch.cat(all_batches, dim=0).contiguous()

def extract_and_compress_masks(rgb_pairs_all, segnet, device, crf, archive_dir, batch_size=8):
    expected_frames = rgb_pairs_all.shape[0]
    
    # Versioned filenames for caching
    raw_path = archive_dir / f"raw_masks_crf{crf}.yuv"
    obu_path = archive_dir / f"mask_crf{crf}.obu"
    obu_br_path = archive_dir / f"mask_crf{crf}.obu.br"
    
    # Stable filename for inflate.py compatibility
    stable_obu_br_path = archive_dir / "mask.obu.br"

    # --- 1. Cache Check & Validation ---
    if obu_br_path.exists():
        logging.info(f"Found cached mask for CRF {crf}. Verifying completeness...")
        try:
            with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp_obu:
                with open(obu_br_path, "rb") as f_in:
                    tmp_obu.write(brotli.decompress(f_in.read()))
                tmp_obu_path = tmp_obu.name
            
            container = av.open(tmp_obu_path)
            frames = []
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="gray")
                cls_img = np.clip(np.round(img / 63.0).astype(np.uint8), 0, 4)
                frames.append(cls_img)
            container.close()
            os.remove(tmp_obu_path)
            
            if len(frames) == expected_frames:
                logging.info(f"Cached video is complete ({len(frames)} frames). Skipping extraction.")
                # Ensure the stable filename is up to date with this cached version
                shutil.copyfile(obu_br_path, stable_obu_br_path)
                return torch.from_numpy(np.stack(frames)).contiguous()
            else:
                logging.warning(f"Cached video incomplete ({len(frames)}/{expected_frames} frames). Regenerating...")
        except Exception as e:
            logging.warning(f"Failed to load cached mask ({e}). Regenerating...")

    # --- 2. Generation & Extraction ---
    logging.info("Generating odd-frame raw masks from cached RGB pairs...")
    with open(raw_path, "wb") as f_out:
        with torch.inference_mode():
            for start in tqdm(range(0, expected_frames, batch_size), desc="Extracting Masks"):
                batch = rgb_pairs_all[start:start+batch_size].to(device).float()
                batch = einops.rearrange(batch, 'b t h w c -> b t c h w')
                odd_frames = batch[:, 1] 
                
                resized = torch.nn.functional.interpolate(
                    odd_frames, 
                    size=(SEGNET_MODEL_INPUT_SIZE[1], SEGNET_MODEL_INPUT_SIZE[0]), 
                    mode='bilinear'
                )
                
                out = segnet(resized)
                mask = out.argmax(dim=1).to(torch.uint8)
                mask_scaled = mask * 63 
                f_out.write(mask_scaled.cpu().numpy().tobytes())

    # --- 3. Compression ---
    logging.info(f"Compressing masks to OBU using FFmpeg (CRF {crf})...")
    ffmpeg_cmd = [
        get_ffmpeg_path(), "-y", "-hide_banner",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", "512x384", 
        "-r", "10", 
        "-i", str(raw_path),
        "-c:v", "libaom-av1",
        "-crf", str(crf),
        "-cpu-used", "0",
        "-row-mt", "1",
        "-g", "1200",
        "-keyint_min", "1200",
        "-lag-in-frames", "48",
        "-arnr-strength", "0",
        "-aq-mode", "0",
        "-aom-params", "enable-cdef=0:enable-intrabc=1:enable-obmc=0",
        "-f", "obu",
        str(obu_path)
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    logging.info("Applying Brotli compression to OBU...")
    with open(obu_path, "rb") as f_in, open(obu_br_path, "wb") as f_out:
        f_out.write(brotli.compress(f_in.read(), quality=11, lgwin=24))

    # Provide the stable filename for inflate.py
    shutil.copyfile(obu_br_path, stable_obu_br_path)

    # --- 4. Validation & RAM Decoding ---
    logging.info("Decoding OBU artifacts to RAM for training cache...")
    container = av.open(str(obu_path))
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="gray")
        cls_img = np.clip(np.round(img / 63.0).astype(np.uint8), 0, 4)
        frames.append(cls_img)
    container.close()
    
    # Completeness Check
    if len(frames) != expected_frames:
        raise RuntimeError(f"FFmpeg encoding failed! Generated {len(frames)} frames, expected {expected_frames}.")
    
    # Cleanup temporary files (keep the .obu.br files)
    obu_path.unlink()
    raw_path.unlink()
    
    return torch.from_numpy(np.stack(frames)).contiguous()

def extract_and_compress_poses(rgb_pairs_all, posenet, device, archive_dir, batch_size=8):
    br_path = archive_dir / "pose.npy.br"
    all_pose6 = []

    logging.info("Generating FP32 Poses from cached RGB pairs...")
    with torch.inference_mode():
        for start in tqdm(range(0, rgb_pairs_all.shape[0], batch_size), desc="Extracting Poses"):
            batch = rgb_pairs_all[start:start+batch_size].to(device).float()
            batch = einops.rearrange(batch, "b t h w c -> b t c h w")
            
            posenet_in = posenet.preprocess_input(batch)
            out = posenet(posenet_in)
            pose6 = out["pose"][..., :6].to(torch.float32)
            all_pose6.append(pose6.cpu().numpy())

    pose_arr = np.concatenate(all_pose6, axis=0)
    
    buffer = io.BytesIO()
    np.save(buffer, pose_arr)
    buffer.seek(0)
    
    logging.info("Applying Brotli compression to Poses...")
    with open(br_path, "wb") as f_out:
        f_out.write(brotli.compress(buffer.read(), quality=11, lgwin=24))
        
    return torch.from_numpy(pose_arr).float().contiguous()

class CachedPairLoader:
    def __init__(self, rgb_pairs_cpu, mask2_cpu, pose6_cpu, batch_size, device, seed=123, shuffle=True):
        self.rgb_pairs = rgb_pairs_cpu.contiguous()
        self.mask2 = mask2_cpu.contiguous()
        self.pose6 = pose6_cpu.contiguous()
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.num_samples = rgb_pairs_cpu.shape[0]

    def set_epoch(self, epoch: int): self.epoch = int(epoch)
    def __len__(self): return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.num_samples, generator=g) if self.shuffle else torch.arange(self.num_samples)
        for start in range(0, self.num_samples, self.batch_size):
            idx = perm[start : start + self.batch_size]
            yield (self.rgb_pairs.index_select(0, idx).to(self.device, non_blocking=True),
                   self.mask2.index_select(0, idx).to(self.device, non_blocking=True),
                   self.pose6.index_select(0, idx).to(self.device, non_blocking=True))

# -----------------------------
# FP4 Logic
# -----------------------------
class FP4Codebook:
    pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    @staticmethod
    def quantize_blockwise(x: torch.Tensor, block_size: int = 32):
        orig_shape = x.shape
        flat = x.reshape(-1)
        pad = (block_size - (flat.numel() % block_size)) % block_size
        if pad: flat = F.pad(flat, (0, pad))
        blocks = flat.view(-1, block_size)
        max_abs = blocks.abs().amax(dim=1, keepdim=True)
        scales = torch.where(max_abs > 0, max_abs / 6.0, torch.ones_like(max_abs))
        norm = blocks / scales
        signs = (norm < 0).to(torch.int16)
        levels = FP4Codebook.pos_levels.to(x.device, x.dtype).view(1, 1, -1)
        mag_idx = (norm.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.int16)
        q = torch.where(signs.bool(), -levels[0, 0, mag_idx.long()], levels[0, 0, mag_idx.long()])
        return (q * scales).view(-1)[:x.numel()].view(orig_shape), ((signs << 3) | mag_idx).to(torch.uint8), scales.squeeze(1)

    @staticmethod
    def dequantize_from_nibbles(nibbles, scales, orig_shape):
        flat_n = int(torch.tensor(orig_shape).prod().item())
        nibbles = nibbles.view(-1, nibbles.numel() // scales.numel())
        signs, mag_idx = (nibbles >> 3).to(torch.int64), (nibbles & 0x7).to(torch.int64)
        levels = FP4Codebook.pos_levels.to(scales.device, torch.float32)
        q = torch.where(signs.bool(), -levels[mag_idx], levels[mag_idx])
        return (q * scales[:, None].to(torch.float32)).view(-1)[:flat_n].reshape(orig_shape)

def fake_quant_fp4_ste(x, block_size=32):
    dq, _, _ = FP4Codebook.quantize_blockwise(x, block_size=block_size)
    return x + (dq - x).detach()

def pack_nibbles(nib):
    flat = nib.reshape(-1)
    if flat.numel() % 2 == 1: flat = F.pad(flat, (0, 1))
    return ((flat[0::2] & 0x0F) << 4) | (flat[1::2] & 0x0F)

def unpack_nibbles(packed, count):
    flat = packed.reshape(-1)
    out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2], out[1::2] = (flat >> 4) & 0x0F, flat & 0x0F
    return out[:count]

def load_fp4_state_dict(model, fp4_path, device):
    data = torch.load(fp4_path, map_location=device)
    state_dict, source = {}, data.get("tensors", data.get("quantized", {}))
    for name, rec in source.items():
        if rec["weight_kind"] == "fp4_packed":
            nibbles = unpack_nibbles(rec["packed_weight"].to(device), rec["packed_weight"].numel() * 2)
            w = FP4Codebook.dequantize_from_nibbles(nibbles, rec["scales_fp16"].to(device), rec["weight_shape"])
        else: w = rec["weight_fp16"].to(device).float()
        state_dict[f"{name}.weight"] = w
        if rec.get("bias_fp16") is not None: state_dict[f"{name}.bias"] = rec["bias_fp16"].to(device).float()
    for k, v in data.get("dense_fp16", {}).items(): state_dict[k] = v.to(device).float()
    model.load_state_dict(state_dict, strict=False)

def export_fp4_state_dict(model, out_path, block_size=32):
    export = {"__format__": "fp4_standalone", "__block_size__": block_size, "__codebook__": FP4Codebook.pos_levels.clone(), "quantized": {}, "dense_fp16": {}}
    covered_keys = set()
    for name, m in model.named_modules():
        if isinstance(m, (QConv2d, QEmbedding)):
            rec = {"type": "conv2d" if isinstance(m, QConv2d) else "embedding"}
            w = m.weight.detach().float().cpu()
            rec["weight_shape"] = list(w.shape)
            covered_keys.add(f"{name}.weight")
            if isinstance(m, QConv2d):
                rec["stride"], rec["padding"], rec["dilation"], rec["groups"] = list(m.stride) if isinstance(m.stride, tuple) else [m.stride]*2, list(m.padding) if isinstance(m.padding, tuple) else [m.padding]*2, list(m.dilation) if isinstance(m.dilation, tuple) else [m.dilation]*2, int(m.groups)
                rec["bias_fp16"] = m.bias.detach().half().cpu() if m.bias is not None else None
                if m.bias is not None: covered_keys.add(f"{name}.bias")
            if getattr(m, 'quantize_weight', False):
                _, nib, scales = FP4Codebook.quantize_blockwise(w, block_size=block_size)
                rec.update({"weight_kind": "fp4_packed", "weight_numel": int(w.numel()), "packed_weight": pack_nibbles(nib.cpu()), "scales_fp16": scales.half().cpu()})
            else: rec.update({"weight_kind": "fp16", "weight_fp16": w.half().cpu()})
            export["quantized"][name] = rec
    for k, v in model.state_dict().items():
        if k not in covered_keys: export["dense_fp16"][k] = v.detach().cpu().half() if torch.is_floating_point(v) else v.detach().cpu()
    torch.save(export, out_path, _use_new_zipfile_serialization=False)

# -----------------------------
# Quantizable Modules
# -----------------------------
class QMixin:
    def set_qat(self, enabled: bool, act_enabled: bool = False):
        self.qat_enabled = enabled
        self.qat_act_enabled = act_enabled

class QConv2d(nn.Conv2d, QMixin):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.quantize_weight = quantize_weight
        self.qat_enabled = False

    def forward(self, x):
        w = fake_quant_fp4_ste(self.weight, self.block_size) if self.qat_enabled and self.quantize_weight else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QEmbedding(nn.Embedding, QMixin):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.quantize_weight = quantize_weight
        self.qat_enabled = False

    def forward(self, x):
        w = fake_quant_fp4_ste(self.weight, self.block_size) if self.qat_enabled and self.quantize_weight else self.weight
        return F.embedding(x, w, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

# -----------------------------
# Architecture
# -----------------------------
class SepConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        mid_ch = in_ch * depth_mult
        self.dw = QConv2d(in_ch, mid_ch, k, stride=stride, padding=k//2, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid_ch, out_ch, 1, padding=0, bias=True, quantize_weight=quantize_weight)
        self.norm = nn.GroupNorm(2, out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.norm(self.pw(self.dw(x))))

class SepConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        mid_ch = in_ch * depth_mult
        self.dw = QConv2d(in_ch, mid_ch, k, stride=stride, padding=k//2, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid_ch, out_ch, 1, padding=0, bias=True, quantize_weight=quantize_weight)
    def forward(self, x): return self.pw(self.dw(x))

class SepResBlock(nn.Module):
    def __init__(self, ch, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(x + self.norm2(self.conv2(self.conv1(x))))

class FiLMSepResBlock(nn.Module):
    def __init__(self, ch, cond_dim, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch)
        self.film_proj = nn.Linear(cond_dim, ch * 2)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x, cond_emb):
        x_base = self.norm2(self.conv2(self.conv1(x)))
        gamma, beta = self.film_proj(cond_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return self.act(x + (x_base * (1.0 + gamma) + beta))

class SharedMaskDecoder(nn.Module):
    def __init__(self, num_classes=5, emb_dim=6, c1=56, c2=64, depth_mult=1):
        super().__init__()
        self.embedding = QEmbedding(num_classes, emb_dim, quantize_weight=False)
        self.stem_conv = SepConvGNAct(emb_dim + 2, c1, depth_mult=depth_mult)
        self.stem_block = SepResBlock(c1, depth_mult=depth_mult)
        self.down_conv = SepConvGNAct(c1, c2, stride=2, depth_mult=depth_mult)
        self.down_block = SepResBlock(c2, depth_mult=depth_mult)
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), SepConvGNAct(c2, c1, depth_mult=depth_mult))
        self.fuse = SepConvGNAct(c1 + c1, c1, depth_mult=depth_mult)
        self.fuse_block = SepResBlock(c1, depth_mult=depth_mult)

    def forward(self, mask2, coords):
        e2 = self.embedding(mask2.long()).permute(0, 3, 1, 2)
        e2_up = F.interpolate(e2, size=coords.shape[-2:], mode="bilinear", align_corners=False)
        s = self.stem_block(self.stem_conv(torch.cat([e2_up, coords], dim=1)))
        z = self.up(self.down_block(self.down_conv(s)))
        return self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))

class Frame2StaticHead(nn.Module):
    def __init__(self, in_ch, hidden=52, depth_mult=1):
        super().__init__()
        self.block1 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)
    def forward(self, feat): return torch.sigmoid(self.head(self.pre(self.block2(self.block1(feat))))) * 255.0

class FrameHead(nn.Module):
    def __init__(self, in_ch, cond_dim=48, hidden=52, depth_mult=1):
        super().__init__()
        self.block1 = FiLMSepResBlock(in_ch, cond_dim, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)
    def forward(self, feat, cond_emb): return torch.sigmoid(self.head(self.pre(self.block2(self.block1(feat, cond_emb))))) * 255.0

class JointFrameGenerator(nn.Module):
    def __init__(self, num_classes=5, pose_dim=6, cond_dim=48, depth_mult=1):
        super().__init__()
        self.shared_trunk = SharedMaskDecoder(num_classes=num_classes, emb_dim=6, c1=56, c2=64, depth_mult=depth_mult)
        self.pose_mlp = nn.Sequential(nn.Linear(pose_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.frame1_head = FrameHead(in_ch=56, cond_dim=cond_dim, hidden=52, depth_mult=depth_mult)
        self.frame2_head = Frame2StaticHead(in_ch=56, hidden=52, depth_mult=depth_mult)

    def set_qat(self, enabled: bool):
        for m in self.modules():
            if isinstance(m, (QConv2d, QEmbedding)): m.set_qat(enabled=enabled)

    def forward(self, mask2, pose6):
        coords = make_coord_grid(mask2.shape[0], 384, 512, mask2.device, torch.float32)
        shared_feat = self.shared_trunk(mask2, coords)
        return self.frame1_head(shared_feat, self.pose_mlp(pose6)), self.frame2_head(shared_feat)

# -----------------------------
# Freeze Control & Training Engine
# -----------------------------
def apply_freeze_state(model: JointFrameGenerator, stage: Stage):
    for p in model.parameters(): p.requires_grad = True
    
    if stage == Stage.ANCHOR:
        logging.info("STAGE: ANCHOR -> Freezing Frame 1 and Pose.")
        for p in model.frame1_head.parameters(): p.requires_grad = False
        for p in model.pose_mlp.parameters(): p.requires_grad = False
    elif stage == Stage.FINETUNE:
        logging.info("STAGE: FINETUNE -> Freezing Shared Trunk and Frame 2.")
        for p in model.shared_trunk.parameters(): p.requires_grad = False
        for p in model.frame2_head.parameters(): p.requires_grad = False
    elif stage == Stage.JOINT:
        logging.info("STAGE: JOINT -> All parameters unfrozen.")

    if stage == Stage.FINETUNE:
        model.shared_trunk.eval()
        model.frame2_head.eval()

def train_run(run: PipelineRun, generator: JointFrameGenerator, loader: CachedPairLoader, device, archive_dir, aux_models, state_dict_to_load=None):
    segnet, posenet, distortion_net = aux_models
    apply_freeze_state(generator, run.stage)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, generator.parameters()), lr=run.lr, betas=(0.9, 0.99))
    start_epoch, best_metric = 0, float("inf")
    
    latest_path = archive_dir / f"{run.name}_latest.pt"
    if latest_path.exists():
        logging.info(f"Resuming {run.name} from {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device)
        generator.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]
    elif state_dict_to_load is not None:
        logging.info("Loading previous stage state dict into Generator...")
        generator.load_state_dict(state_dict_to_load)

    ema = EMA(generator, decay=run.ema_decay) if run.ema_decay > 0 else None
    if ema and latest_path.exists() and checkpoint.get("ema_state"):
        ema.shadow = {k: v.to(device) for k, v in checkpoint["ema_state"].items()}

    qat_warmup = min(run.warmup_epochs, max(1, (run.epochs - run.qat_start_epoch) // 2)) if run.qat_start_epoch == 0 else run.warmup_epochs
    warmup_sch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=qat_warmup)
    main_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, run.epochs - qat_warmup))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sch, main_sch], milestones=[qat_warmup])

    for _ in range(start_epoch): scheduler.step()

    for epoch in range(start_epoch, run.epochs):
        generator.train()
        if run.stage == Stage.FINETUNE:
            generator.shared_trunk.eval()
            generator.frame2_head.eval()

        loader.set_epoch(epoch)
        qat_on = epoch >= run.qat_start_epoch
        generator.set_qat(qat_on)

        if epoch == run.qat_start_epoch and run.qat_start_epoch > 0:
            logging.info(f"--- QAT Phase Initiated. Resetting Optimizer ---")
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, generator.parameters()), lr=run.lr, betas=(0.9, 0.99))
            warmup_sch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=qat_warmup)
            main_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, (run.epochs - epoch) - qat_warmup))
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sch, main_sch], milestones=[qat_warmup])

        kl_ce_alpha = min(1.0, epoch / max(1, run.qat_start_epoch // 2)) if run.qat_start_epoch > 0 else 1.0
        seg2_kl_w = 0.9 - (0.9 * kl_ce_alpha)
        seg2_ce_w = 0.1 + (0.9 * kl_ce_alpha)
        frame1_sem_w = max(0.0, 1.0 - (epoch / run.frame1_fade_epochs)) if run.frame1_fade_epochs > 0 else 0.0

        total_loss_sum, total_seg2_ce, total_seg1_ce, total_pose_dist, batches = 0.0, 0.0, 0.0, 0.0, 0

        pbar = tqdm(loader, desc=f"Run: {run.name} | Epoch {epoch+1}/{run.epochs}", leave=False)
        for batch_rgb, in_mask2, in_pose6 in pbar:
            batch = einops.rearrange(batch_rgb, "b t h w c -> b t c h w").float().to(device)
            in_mask2, in_pose6 = in_mask2.to(device).long(), in_pose6.to(device).float()

            with torch.no_grad():
                real1 = F.interpolate(batch[:, 0], size=(384, 512), mode="bilinear", align_corners=False)
                real2 = F.interpolate(batch[:, 1], size=(384, 512), mode="bilinear", align_corners=False)
                gt_logits1, gt_logits2 = segnet(real1).float(), segnet(real2).float()
                gt_mask1, gt_mask2 = gt_logits1.argmax(dim=1), gt_logits2.argmax(dim=1)
                gt_pose = get_pose_tensor(posenet(posenet.preprocess_input(batch))).float()[..., :6]

            optimizer.zero_grad(set_to_none=True)
            pred_frame1, pred_frame2 = generator(in_mask2, in_pose6)

            fake1_up = F.interpolate(pred_frame1, size=(874, 1164), mode="bilinear", align_corners=False)
            fake2_up = F.interpolate(pred_frame2, size=(874, 1164), mode="bilinear", align_corners=False)
            fake1_down = F.interpolate(diff_round(fake1_up.clamp(0, 255)), size=(384, 512), mode="bilinear", align_corners=False)
            fake2_down = F.interpolate(diff_round(fake2_up.clamp(0, 255)), size=(384, 512), mode="bilinear", align_corners=False)

            loss, loss_pose, loss_seg2, loss_seg1 = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            loss_seg2_ce, loss_seg1_ce = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

            if run.stage in [Stage.FINETUNE, Stage.JOINT]:
                fake_pose = get_pose_tensor(posenet(pack_pair_yuv6(fake1_down, fake2_down).float())).float()[..., :6]
                loss_pose = F.mse_loss(fake_pose, gt_pose)

            if run.stage in [Stage.ANCHOR, Stage.JOINT]:
                fake_logits2 = segnet(fake2_down).float()
                ce_unreduced2 = F.cross_entropy(fake_logits2, gt_mask2, reduction='none')
                with torch.no_grad():
                    boost2 = 1.0 + ((fake_logits2.argmax(dim=1) != gt_mask2).float() * run.error_boost)
                loss_seg2_ce = (ce_unreduced2 * boost2).mean()
                loss_seg2_kl = kl_on_logits(fake_logits2, gt_logits2, 2.0) / (384 * 512)
                loss_seg2 = 100.0 * ((seg2_kl_w * loss_seg2_kl) + (seg2_ce_w * 0.5 * run.ce_weight * loss_seg2_ce))

            if frame1_sem_w > 0:
                fake_logits1 = segnet(fake1_down).float()
                ce_unreduced1 = F.cross_entropy(fake_logits1, gt_mask1, reduction='none')
                with torch.no_grad():
                    boost1 = 1.0 + ((fake_logits1.argmax(dim=1) != gt_mask1).float() * run.error_boost)
                loss_seg1_ce = (ce_unreduced1 * boost1).mean()
                
                if run.stage == Stage.JOINT:
                    loss_seg1_kl = kl_on_logits(fake_logits1, gt_logits1, 2.0) / (384 * 512)
                    loss_seg1 = 100.0 * frame1_sem_w * ((seg2_kl_w * loss_seg1_kl) + (seg2_ce_w * 0.5 * run.ce_weight * loss_seg1_ce))
                else: 
                    loss_seg1 = 100.0 * frame1_sem_w * (run.ce_weight * loss_seg1_ce)

            if run.stage == Stage.ANCHOR: loss = loss_seg2
            elif run.stage == Stage.FINETUNE: loss = loss_seg1 + (run.pose_weight * loss_pose * 10.0)
            elif run.stage == Stage.JOINT: loss = loss_seg2 + loss_seg1 + (30.0 * run.pose_weight * loss_pose)

            assert_finite("loss", loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=run.grad_clip)
            optimizer.step()

            if ema and epoch >= run.warmup_epochs: ema.update(generator)
            
            total_loss_sum += loss.item(); total_seg2_ce += loss_seg2_ce.item()
            total_seg1_ce += loss_seg1_ce.item(); total_pose_dist += loss_pose.item(); batches += 1
            pbar.set_postfix({"L": f"{loss.item():.2f}", "S2": f"{loss_seg2_ce.item():.2f}", "P": f"{loss_pose.item():.4f}"})

        scheduler.step()
        avg_loss, avg_s2, avg_s1, avg_p = total_loss_sum/max(1, batches), total_seg2_ce/max(1, batches), total_seg1_ce/max(1, batches), total_pose_dist/max(1, batches)

        logging.info(f"\nEpoch {epoch+1}/{run.epochs} {'[QAT ACTIVE]' if qat_on else ''}")
        logging.info(f"  Avg Loss:      {avg_loss:.4f}")
        if run.stage in [Stage.ANCHOR, Stage.JOINT]: logging.info(f"  Avg Seg2 CE:   {avg_s2:.4f}")
        if frame1_sem_w > 0: logging.info(f"  Avg Seg1 CE:   {avg_s1:.4f}")
        if run.stage in [Stage.FINETUNE, Stage.JOINT]: logging.info(f"  Avg Pose MSE:  {avg_p:.6f}")

        is_eval_epoch = qat_on and (((epoch - run.qat_start_epoch) % 5 == 0) or (run.epochs - epoch <= 10))
        
        if is_eval_epoch:
            if ema: ema.apply_shadow(generator)
            generator.eval()
            total_seg, total_pose, samples = 0.0, 0.0, 0
            
            with torch.inference_mode():
                eval_pbar = tqdm(loader, desc=f"Eval: {run.name} Ep {epoch+1}", leave=False)
                for batch_rgb, in_mask2, in_pose6 in eval_pbar:
                    batch_gt = batch_rgb.to(device)
                    p1, p2 = generator(in_mask2.to(device).long(), in_pose6.to(device).float())
                    
                    b_comp = torch.stack([F.interpolate(p1, size=(874, 1164), mode="bilinear", align_corners=False), 
                                          F.interpolate(p2, size=(874, 1164), mode="bilinear", align_corners=False)], dim=1)
                    b_comp = einops.rearrange(b_comp, "b t c h w -> b t h w c").clamp(0, 255).round().to(torch.uint8)
                    
                    p_dist, s_dist = distortion_net.compute_distortion(batch_gt, b_comp)
                    total_seg += s_dist.sum().item()
                    total_pose += p_dist.sum().item()
                    samples += batch_gt.shape[0]
            
            # --- Score Component Breakdown ---
            avg_seg = total_seg / max(1, samples)
            avg_pose = total_pose / max(1, samples)
            
            # 1. Estimate Rate (Bits Per Pixel)
            model_file = archive_dir / "model.pt.br"
            model_size = model_file.stat().st_size if model_file.exists() else 1500000 # ~1.5MB fallback estimate
            mask_file = archive_dir / "mask.obu.br"
            mask_size = mask_file.stat().st_size if mask_file.exists() else 0
            pose_file = archive_dir / "poses.npy.br"
            pose_size = pose_file.stat().st_size if pose_file.exists() else 0
            
            total_bytes = model_size + mask_size + pose_size
            total_pixels = samples * 2 * 1164 * 874
            rate_bpp = (total_bytes * 8.0) / max(1, total_pixels)
            
            # 2. Final Challenge Score Formula
            scaled_seg = 100.0 * avg_seg
            scaled_pose = math.sqrt(max(0, 10.0 * avg_pose))
            scaled_rate = 25.0 * rate_bpp
            
            # Use unified formula for all stages
            eval_metric = scaled_seg + scaled_pose + scaled_rate
            
            logging.info(f"  [Eval] Est Score: {eval_metric:.4f} | Seg(x100): {scaled_seg:.4f} | Pose(√x10): {scaled_pose:.4f} | Rate(x25): {scaled_rate:.4f} (bpp: {rate_bpp:.4f})")
            
            if eval_metric < best_metric:
                best_metric = eval_metric
                
                best_state_fp16 = {k: v.half() if torch.is_floating_point(v) else v for k, v in generator.state_dict().items()}
                torch.save(best_state_fp16, archive_dir / f"{run.name}_best_fp16.pt")

                fp4_path = archive_dir / f"{run.name}_best_fp4.pt"
                export_fp4_state_dict(generator.cpu(), fp4_path)
                generator.to(device) 

                with open(fp4_path, "rb") as f_in:
                    comp = brotli.compress(f_in.read(), quality=11, lgwin=24)
                with open(archive_dir / f"{run.name}_best_fp4.pt.br", "wb") as f_out:
                    f_out.write(comp)
                
                # Update final standalone checkpoint mapping
                shutil.copyfile(archive_dir / f"{run.name}_best_fp4.pt.br", archive_dir / "model.pt.br")

                logging.info(f"  *** New Best QAT Score: {best_metric:.5f} (Saved to model.pt.br) ***")

            if ema: ema.restore(generator)

        torch.save({
            "epoch": epoch, "best_metric": best_metric,
            "model_state": generator.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "ema_state": {k: v.cpu() for k, v in ema.shadow.items()} if ema else None
        }, latest_path)

    if latest_path.exists(): latest_path.unlink()
    return load_best_fp4(generator, archive_dir / f"{run.name}_best_fp4.pt", device)

def load_best_fp4(model, path, device):
    load_fp4_state_dict(model, path, device)
    model.float()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

# -----------------------------
# Main Setup
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", type=Path, default=ROOT_DIR / "videos")
    p.add_argument("--video-names", type=Path, default=ROOT_DIR / "public_test_video_names.txt")
    p.add_argument("--crf", type=int, default=50, help="CRF value for AV1 OBU mask compression")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    archive_dir = Path(__file__).parent / "archive"
    archive_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(archive_dir / "pipeline.log")])
    logging.info("Initializing Full End-to-End Pipeline Execution...")

    # Load Auxiliary Networks once to use in Extraction and Validation
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    dist_net = DistortionNet().eval().to(device)
    dist_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for model in (segnet, posenet, dist_net):
        for p in model.parameters(): p.requires_grad = False

    # --- 1. PRELOAD DATA TO RAM ---
    files = [line.strip() for line in args.video_names.read_text().splitlines() if line.strip()]
    rgb_pairs_all = preload_video_pair_cache_dali(files, args.video_dir, args.batch_size, device)

    # --- 2. EXTRACT MASKS & POSES ---
    mask_frames_all = extract_and_compress_masks(rgb_pairs_all, segnet, device, args.crf, archive_dir)
    pose6_all = extract_and_compress_poses(rgb_pairs_all, posenet, device, archive_dir)

    # Initialize Train Loader
    loader = CachedPairLoader(rgb_pairs_all, mask_frames_all, pose6_all, args.batch_size, device)
    generator = JointFrameGenerator().to(device)

    # --- 3. PIPELINE EXECUTION ---
    PIPELINE = [
        PipelineRun(name="run1_anchor", stage=Stage.ANCHOR, epochs=400, lr=5e-4, qat_start_epoch=200, frame1_fade_epochs=50, error_boost=9.0),
        PipelineRun(name="run2_anchor_boost", stage=Stage.ANCHOR, epochs=80, lr=1e-5, qat_start_epoch=0, frame1_fade_epochs=0, error_boost=49.0),
        PipelineRun(name="run3_finetune", stage=Stage.FINETUNE, epochs=320, lr=5e-5, qat_start_epoch=120, frame1_fade_epochs=60, pose_weight=1.0),
        PipelineRun(name="run4_finish", stage=Stage.JOINT, epochs=160, lr=1e-5, qat_start_epoch=0, frame1_fade_epochs=40, pose_weight=1.0),
        PipelineRun(name="run5_micro", stage=Stage.FINETUNE, epochs=120, lr=5e-6, qat_start_epoch=0, frame1_fade_epochs=0, pose_weight=1.0),
    ]

    current_state_dict = None
    for run in PIPELINE:
        best_path = archive_dir / f"{run.name}_best_fp4.pt"
        latest_path = archive_dir / f"{run.name}_latest.pt"
        
        # A run is ONLY fully complete if it has a best checkpoint AND the active latest.pt file was cleaned up.
        if best_path.exists() and not latest_path.exists():
            logging.info(f"\n[SKIP] Run '{run.name}' is already completed. Loading state to pass to next stage...")
            current_state_dict = load_best_fp4(generator, best_path, device)
            continue
        
        if latest_path.exists():
            logging.info(f"\n[RESUME] Found interrupted state for '{run.name}'. Resuming...")
        
        logging.info(f"\n" + "="*50)
        logging.info(f"STARTING PIPELINE RUN: {run.name}")
        logging.info("="*50)
        
        current_state_dict = train_run(run, generator, loader, device, archive_dir, (segnet, posenet, dist_net), current_state_dict)

    logging.info("\nEnd-to-End Execution Fully Completed. Final model saved to archive/model.pt.br")

if __name__ == "__main__":
    main()
