#!/usr/bin/env python
"""Self-contained pipeline to beat the #1 comma-compression submission via
mixed-precision QAT of the top HNeRV decoder. Works on CUDA (Colab) or MPS/CPU.

Pipeline:
  1. extract_base()         -> download #1 (fec6) archive, recover trained decoder+latents
  2. distill(bits, ...)     -> low-bit decoder reproduces the int8 base RGB outputs (fast, stable)
  3. polish(ckpt, ...)      -> short evaluator-in-the-loop refine on the real seg+pose metric
  4. evaluate_checkpoint()  -> faithful score (100*seg + sqrt(10*pose) + 25*rate)

Assumes the commaai repo is the CWD (frame_utils.py, modules.py, models/, videos/,
submissions/hnerv_muon/src on path).
"""
from __future__ import annotations

import io
import math
import struct
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(".").resolve()
for p in [str(ROOT), str(ROOT / "submissions" / "hnerv_muon" / "src"),
          str(ROOT / "submissions" / "hnerv_fec6_fixed_huffman_k16"),
          str(ROOT / "submissions" / "hnerv_fec6_fixed_huffman_k16" / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from frame_utils import AVVideoDataset, camera_size  # noqa: E402
from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402
from model import HNeRVDecoder  # noqa: E402  (muon == fec6, byte-identical)


def encode_latents(latents):
    """Inlined from muon codec (avoids the muon/fec6 'codec' module name collision).
    Per-dim minmax uint8 + temporal delta + zigzag + lo/hi byte split."""
    t = latents.detach().cpu().float()
    n, d = t.shape
    mins = t.min(dim=0).values
    maxs = t.max(dim=0).values
    scales = ((maxs - mins) / 254.0).clamp(min=1e-10)
    q = ((t - mins.unsqueeze(0)) / scales.unsqueeze(0)).round().clamp(0, 254).to(torch.uint8).numpy()
    delta = np.empty_like(q, dtype=np.int16)
    delta[0] = q[0]
    delta[1:] = q[1:].astype(np.int16) - q[:-1].astype(np.int16)
    delta_zz = np.where(delta >= 0, 2 * delta, -2 * delta - 1).astype(np.uint16)
    lo = (delta_zz & 0xFF).astype(np.uint8).tobytes()
    hi = (delta_zz >> 8).astype(np.uint8).tobytes()
    payload = struct.pack("<II", n, d)
    payload += mins.to(torch.float16).numpy().tobytes()
    payload += scales.to(torch.float16).numpy().tobytes()
    payload += lo + hi
    return payload

CAMERA_W, CAMERA_H = camera_size
ORIG_SIZE = 37_545_489
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)
GT_CACHE = WORK / "gt_targets.pt"
META_BYTES = 120
FEC6_URL = ("https://github.com/adpena/comma_video_compression_challenge/releases/"
            "download/fec6-frontier-submission-20260520/archive.zip")


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# base weights (the #1 submission's trained decoder + latents)
# ----------------------------------------------------------------------------
def extract_base(out=WORK / "base_decoder.pt"):
    if out.exists():
        return torch.load(out)
    azip = WORK / "fec6_archive.zip"
    if not azip.exists():
        print("downloading #1 archive ...")
        urllib.request.urlretrieve(FEC6_URL, azip)
    with zipfile.ZipFile(azip) as z:
        data = z.read("x")
    from inflate import parse_pr101_frame_selector_archive  # fec6 inflate
    from codec import parse_archive  # fec6 src/codec (compact)
    source_payload, kind, codes, specs = parse_pr101_frame_selector_archive(data)
    decoder_sd, latents, meta = parse_archive(source_payload)
    blob = {"decoder_sd": decoder_sd, "latents": latents, "meta": meta}
    torch.save(blob, out)
    print(f"base extracted: {sum(v.numel() for v in decoder_sd.values())} params, "
          f"latents {tuple(latents.shape)}")
    return blob


def make_decoder(meta, device):
    return HNeRVDecoder(latent_dim=meta["latent_dim"], base_channels=meta["base_channels"],
                        eval_size=tuple(meta["eval_size"])).to(device)


# ----------------------------------------------------------------------------
# frozen evaluator + ground-truth targets
# ----------------------------------------------------------------------------
def build_net(device):
    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def patch_bn_contiguous(module):
    """Only needed on MPS (BatchNorm backward stride bug). Harmless elsewhere."""
    n = 0
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.register_forward_pre_hook(lambda mod, inp: (inp[0].contiguous(),))
            m.register_forward_hook(lambda mod, inp, out: out.contiguous())
            n += 1
    return n


@torch.inference_mode()
def compute_gt(net, device, batch_size=16):
    if GT_CACHE.exists():
        d = torch.load(GT_CACHE)
        return d["seg_argmax"], d["pose"]
    # AVVideoDataset (PyAV) is CPU-only and asserts device.type != 'cuda'; it decodes
    # on CPU and yields CPU frames regardless, so pass a cpu device and move to GPU for the net.
    ds = AVVideoDataset(["0.mkv"], data_dir=ROOT / "videos", batch_size=batch_size,
                        device=torch.device("cpu"), num_threads=2, seed=1234, prefetch_queue_depth=4)
    ds.prepare_data()
    seg, pose = [], []
    for _, _, batch in ds:
        po, so = net(batch.to(device))
        seg.append(so.argmax(1).to(torch.uint8).cpu())
        pose.append(po["pose"][..., :6].to(torch.float16).cpu())
    seg, pose = torch.cat(seg), torch.cat(pose)
    torch.save({"seg_argmax": seg, "pose": pose}, GT_CACHE)
    return seg, pose


# ----------------------------------------------------------------------------
# quantization
# ----------------------------------------------------------------------------
def fake_quant(w, n_levels):
    """Per-output-channel symmetric fake-quant (STE) for dim>=2; per-tensor for 1-d.
    Per-channel ~halves low-bit distortion vs per-tensor (the lever top entries missed)."""
    if w.dim() >= 2:
        out = w.shape[0]
        flat = w.reshape(out, -1)
        ma = flat.abs().amax(1, keepdim=True).clamp(min=1e-12)
        scale = ma / n_levels
        q = (flat / scale).round().clamp(-n_levels, n_levels)
        deq = (q * scale).reshape(w.shape)
    else:
        ma = w.abs().max().clamp(min=1e-12)
        scale = ma / n_levels
        deq = (w / scale).round().clamp(-n_levels, n_levels) * scale
    return (deq - w).detach() + w


def parse_bits(spec):
    bits = {"__default__": 127}
    for kv in (spec or "").split(","):
        if not kv.strip():
            continue
        k, v = kv.split("=")
        k = k.strip()
        bits["__default__" if k in ("default", "*") else k] = int(v)
    return bits


def quant_layers(decoder, bits):
    orig = {}
    d = bits.get("__default__", 127)
    for name, m in decoder.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight"):
            nl = bits.get(name, d)
            orig[name] = m.weight.data.clone()
            m.weight.data = fake_quant(m.weight.data, nl)
    return orig


def restore_layers(decoder, orig):
    for name, m in decoder.named_modules():
        if name in orig:
            m.weight.data = orig[name]


def layer_of(name):
    return name.rsplit(".", 1)[0]


def quantize_mixed(sd, bits):
    """Per-output-channel symmetric quant for weights (dim>=2); per-tensor for biases.
    Returns (q_sd, deq) where q_sd[name] = (int_flat, scales_fp16_array, shape)."""
    q_sd, deq = {}, {}
    d = bits.get("__default__", 127)
    for name, tensor in sd.items():
        t = tensor.detach().cpu().float()
        nq = bits.get(layer_of(name), d) if name.endswith(".weight") else 127
        if t.dim() >= 2:
            out = t.shape[0]
            flat = t.reshape(out, -1)
            scales = (flat.abs().amax(1) / nq).clamp(min=1e-12)
            q = (flat / scales[:, None]).round().clamp(-nq, nq).to(torch.int16).numpy().flatten()
            deq[name] = (torch.from_numpy(q.astype(np.float32)).reshape(out, -1)
                         * scales[:, None]).reshape(tensor.shape)
            sc = scales.to(torch.float16).numpy()
        else:
            m = t.abs().max().item()
            s = m / nq if m > 0 else 1.0
            q = (t / s).round().clamp(-nq, nq).to(torch.int16).numpy().flatten()
            deq[name] = torch.from_numpy(q.astype(np.float32)).reshape(tensor.shape) * s
            sc = np.array([s], dtype=np.float16)
        q_sd[name] = (q, sc, tuple(tensor.shape))
    return q_sd, deq


def encode_decoder_mixed(q_sd):
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(q_sd)))
    for name, (q, scales, shape) in q_sd.items():
        nb = name.encode()
        buf.write(struct.pack("<I", len(nb))); buf.write(nb)
        buf.write(struct.pack("<I", len(shape)))
        for s in shape:
            buf.write(struct.pack("<I", s))
        buf.write(struct.pack("<I", scales.size))
        buf.write(scales.astype(np.float16).tobytes())
        buf.write(struct.pack("<I", q.size))
        arr = q.astype(np.int32)
        zz = np.where(arr >= 0, 2 * arr, -2 * arr - 1).astype(np.uint32)
        mx = int(zz.max()) if zz.size else 0
        if mx < 256:
            buf.write(b"\x01"); buf.write(zz.astype(np.uint8).tobytes())
        else:
            buf.write(b"\x02"); buf.write(zz.astype(np.uint16).tobytes())
    return brotli.compress(buf.getvalue(), quality=11)


# ----------------------------------------------------------------------------
# reconstruction + scoring
# ----------------------------------------------------------------------------
def recon_frames_float(decoder, z, eval_hw):
    decoded = decoder(z)
    b = z.shape[0]
    eh, ew = eval_hw
    flat = decoded.reshape(b * 2, 3, eh, ew)
    up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False).clamp(0, 255)
    return up.reshape(b, 2, 3, CAMERA_H, CAMERA_W).permute(0, 1, 3, 4, 2).contiguous()


@torch.inference_mode()
def score(net, decoder, latents, device, archive_size, seg_gt, pose_gt, batch=16):
    decoder = decoder.to(device).eval()
    eh, ew = decoder.eval_size
    seg_s = pose_s = n = 0
    for i in range(0, latents.shape[0], batch):
        z = latents[i:i + batch].to(device)
        frames = (recon_frames_float(decoder, z, (eh, ew)).round().to(torch.uint8))
        po, so = net(frames)
        b = z.shape[0]
        sp = so.argmax(1)
        gt = seg_gt[i:i + b].to(device).long()
        seg_s += (sp != gt).float().mean((1, 2)).sum().item()
        pp = po["pose"][..., :6]
        pg = pose_gt[i:i + b].to(device).float()
        pose_s += (pp - pg).pow(2).mean(1).sum().item()
        n += b
    seg, pose = seg_s / n, pose_s / n
    rate = archive_size / ORIG_SIZE
    sc = 100 * seg + math.sqrt(10 * pose) + 25 * rate
    return {"seg": seg, "pose": pose, "rate": rate, "score": sc, "size": archive_size}


def evaluate_checkpoint(ckpt, net, seg_gt, pose_gt, device, verbose=True):
    bits = ckpt.get("bits_map", {"__default__": 127})
    q_sd, deq = quantize_mixed(ckpt["decoder_sd"], bits)
    dec_bytes = len(encode_decoder_mixed(q_sd))
    lat_bytes = len(brotli.compress(encode_latents(ckpt["latents"]), quality=11))
    archive = dec_bytes + lat_bytes + META_BYTES
    dec = make_decoder(ckpt["meta"], device)
    dec.load_state_dict(deq)
    r = score(net, dec, ckpt["latents"], device, archive, seg_gt, pose_gt)
    r.update(dec_bytes=dec_bytes, lat_bytes=lat_bytes, bits=bits)
    if verbose:
        print(f"  bits={bits.get('__default__')}  dec={dec_bytes} lat={lat_bytes} "
              f"archive={archive}  seg={r['seg']:.6f}(100*={100*r['seg']:.4f}) "
              f"pose={r['pose']:.6f}(sqrt={math.sqrt(10*r['pose']):.4f}) "
              f"25*rate={25*r['rate']:.4f}  SCORE={r['score']:.4f}")
    return r


# ----------------------------------------------------------------------------
# training: distillation (fast, stable) then optional evaluator-loop polish
# ----------------------------------------------------------------------------
def distill(base_blob, bits_spec, device, steps=4000, lr=2e-4, batch=64,
            train_latents=False, lat_lr=5e-5, log=400):
    meta = base_blob["meta"]
    base = make_decoder(meta, device); base.load_state_dict(base_blob["decoder_sd"]); base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    dec = make_decoder(meta, device); dec.load_state_dict(base_blob["decoder_sd"])
    base_lat = base_blob["latents"].to(device)
    latents = base_lat.clone().requires_grad_(train_latents)
    bits = parse_bits(bits_spec)
    groups = [{"params": list(dec.parameters()), "lr": lr}]
    if train_latents:
        groups.append({"params": [latents], "lr": lat_lr})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n = latents.shape[0]
    g = torch.Generator().manual_seed(0)
    t0 = time.time(); dec.train()
    for s in range(steps):
        idx = torch.randperm(n, generator=g)[:batch].to(device)
        with torch.no_grad():
            target = base(base_lat[idx])
        orig = quant_layers(dec, bits)
        out = dec(latents[idx])
        restore_layers(dec, orig)
        loss = (out - target).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if s % log == 0 or s == steps - 1:
            print(f"  distill[{bits_spec.split(',')[0]}] step {s:5d}  L1 {loss.item():.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  [{time.time()-t0:.0f}s]", flush=True)
    dec.eval()
    return {"decoder_sd": {k: v.detach().cpu() for k, v in dec.state_dict().items()},
            "latents": latents.detach().cpu(), "meta": meta, "bits_map": bits}


def smooth_seg(seg_logits, hard, tau=0.3):
    tl = seg_logits.gather(1, hard.unsqueeze(1))
    masked = seg_logits.clone(); masked.scatter_(1, hard.unsqueeze(1), -1e9)
    margin = tl - masked.max(1, keepdim=True)[0]
    return torch.sigmoid(-margin / tau).mean()


def polish(ckpt, net, seg_gt, pose_gt, device, steps=600, lr=5e-5, batch=16,
           pose_w=3.0, train_latents=True, lat_lr=2e-5, log=100):
    """Short evaluator-in-the-loop refine on the real metric surrogate."""
    meta = ckpt["meta"]
    dec = make_decoder(meta, device); dec.load_state_dict(ckpt["decoder_sd"]); dec.train()
    latents = ckpt["latents"].to(device).clone().requires_grad_(train_latents)
    bits = ckpt.get("bits_map", {"__default__": 127})
    seg_t = seg_gt.to(device).long(); pose_t = pose_gt.to(device).float()
    groups = [{"params": list(dec.parameters()), "lr": lr}]
    if train_latents:
        groups.append({"params": [latents], "lr": lat_lr})
    opt = torch.optim.Adam(groups)
    n = latents.shape[0]; g = torch.Generator().manual_seed(1); t0 = time.time()
    eh, ew = dec.eval_size
    for s in range(steps):
        idx = torch.randperm(n, generator=g)[:batch].to(device)
        orig = quant_layers(dec, bits)
        frames = recon_frames_float(dec, latents[idx], (eh, ew))
        po, so = net(frames)
        restore_layers(dec, orig)
        seg_l = smooth_seg(so, seg_t[idx])
        pose_l = torch.sqrt(10 * (po["pose"][..., :6] - pose_t[idx]).pow(2).mean() + 1e-12)
        loss = 100 * seg_l + pose_w * pose_l
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if s % log == 0 or s == steps - 1:
            print(f"  polish step {s:4d}  100*seg {100*seg_l.item():.4f}  pose {pose_l.item():.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    dec.eval()
    return {"decoder_sd": {k: v.detach().cpu() for k, v in dec.state_dict().items()},
            "latents": latents.detach().cpu(), "meta": meta, "bits_map": bits}
