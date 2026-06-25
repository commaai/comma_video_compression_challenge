#!/usr/bin/env python
"""Train a SMALLER int8 HNeRV decoder distilled from the trained 229K teacher,
then polish vs the exact scorer. Goal: cut decoder bytes (rate) while holding
distortion, to beat 0.192.

Cheap because it distills from the teacher's dense per-pair outputs (not 50 GPU-h
from scratch). Per-tensor int8 QAT throughout (the regime that compresses well).
"""
from __future__ import annotations
import argparse, sys, time, io, struct, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, brotli

sys.path.insert(0, "work")
import beat_top as B
from model import HNeRVDecoder  # repo muon decoder (configurable base_channels)


# ---- per-tensor int8 (compresses well; per-channel does NOT) ----------------
def fake_quant_pt(w, n=127):
    ma = w.abs().max().clamp(min=1e-12)
    s = ma / n
    q = (w / s).round().clamp(-n, n)
    return (q * s - w).detach() + w


def qat_pt(dec, n=127):
    orig = {}
    for name, m in dec.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight"):
            orig[name] = m.weight.data.clone()
            m.weight.data = fake_quant_pt(m.weight.data, n)
    return orig


def restore_pt(dec, orig):
    for name, m in dec.named_modules():
        if name in orig:
            m.weight.data = orig[name]


def encode_int8_pertensor(sd):
    """zigzag + brotli over per-tensor int8 (mirrors muon codec)."""
    buf = io.BytesIO(); buf.write(struct.pack("<I", len(sd)))
    for name, t in sd.items():
        t = t.detach().cpu().float()
        m = t.abs().max().item(); s = m / 127 if m > 0 else 1.0
        q = (t / s).round().clamp(-127, 127).to(torch.int8).numpy().flatten()
        nb = name.encode(); buf.write(struct.pack("<I", len(nb))); buf.write(nb)
        buf.write(struct.pack("<I", t.ndim))
        for d in t.shape:
            buf.write(struct.pack("<I", d))
        buf.write(struct.pack("<f", s)); buf.write(struct.pack("<I", q.size))
        arr = q.astype(np.int32)
        zz = np.where(arr >= 0, 2 * arr, -2 * arr - 1).astype(np.uint8)
        buf.write(zz.tobytes())
    return brotli.compress(buf.getvalue(), quality=11)


def dequant_int8(sd):
    deq = {}
    for name, t in sd.items():
        t = t.detach().cpu().float()
        m = t.abs().max().item(); s = m / 127 if m > 0 else 1.0
        deq[name] = (t / s).round().clamp(-127, 127) * s
    return deq


def slice_init(student, teacher_sd):
    """Warm-start: copy teacher's leading channels into the smaller student.
    Works because HNeRV channel ordering is leading-major in every tensor
    (stem out = ch*48+s; PixelShuffle conv out = ch*4+i; conv in = channel)."""
    ssd = student.state_dict()
    copied = 0
    for name, sp in ssd.items():
        tp = teacher_sd.get(name)
        if tp is None or sp.ndim != tp.ndim:
            continue
        if any(sp.shape[d] > tp.shape[d] for d in range(sp.ndim)):
            continue
        sl = tuple(slice(0, sp.shape[d]) for d in range(sp.ndim))
        sp.copy_(tp[sl]); copied += 1
    student.load_state_dict(ssd)
    return copied


def train_student(C, device, steps=6000, lr=5e-4, batch=32, polish_steps=1500,
                  latent_dim=28, log=500, warm_start=True):
    base = B.extract_base()
    meta = dict(base["meta"]); meta["base_channels"] = C; meta["latent_dim"] = latent_dim
    teacher = B.make_decoder(base["meta"], device)
    teacher.load_state_dict(base["decoder_sd"]); teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teach_lat = base["latents"].to(device)

    student = HNeRVDecoder(latent_dim=latent_dim, base_channels=C,
                           eval_size=tuple(meta["eval_size"])).to(device)
    if warm_start and latent_dim == base["meta"]["latent_dim"]:
        nc = slice_init(student, base["decoder_sd"])
        print(f"[C={C}] warm-started {nc} tensors from teacher", flush=True)
    n_params = sum(p.numel() for p in student.parameters())
    latents = teach_lat.clone().requires_grad_(True)  # adapt latents to student
    opt = torch.optim.Adam([{"params": student.parameters(), "lr": lr},
                            {"params": [latents], "lr": lr * 0.25}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n = latents.shape[0]; g = torch.Generator().manual_seed(0); t0 = time.time()
    print(f"[C={C}] student params={n_params}  distill {steps} steps", flush=True)
    student.train()
    for s in range(steps):
        idx = torch.randperm(n, generator=g)[:batch].to(device)
        with torch.no_grad():
            tgt = teacher(teach_lat[idx])
        orig = qat_pt(student)
        out = student(latents[idx])
        restore_pt(student, orig)
        loss = (out - tgt).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if device.type == "mps" and s % 50 == 0:
            torch.mps.empty_cache()
        if s % log == 0 or s == steps - 1:
            print(f"  [C={C}] distill {s:5d}  L1 {loss.item():.4f}  [{time.time()-t0:.0f}s]", flush=True)
    ck = {"decoder_sd": {k: v.detach().cpu() for k, v in student.state_dict().items()},
          "latents": latents.detach().cpu(), "meta": meta}
    return ck, student, latents


def polish_student(ck, net, seg_gt, pose_gt, device, steps=1500, lr=1e-4,
                   batch=12, pose_w=3.0, log=300):
    """Evaluator-in-the-loop refine of the student on the real metric surrogate,
    with per-tensor int8 QAT so it stays int8-deployable."""
    meta = ck["meta"]
    dec = HNeRVDecoder(latent_dim=meta["latent_dim"], base_channels=meta["base_channels"],
                       eval_size=tuple(meta["eval_size"])).to(device)
    dec.load_state_dict(ck["decoder_sd"]); dec.train()
    latents = ck["latents"].to(device).clone().requires_grad_(True)
    seg_t = seg_gt.to(device).long(); pose_t = pose_gt.to(device).float()
    opt = torch.optim.Adam([{"params": dec.parameters(), "lr": lr},
                            {"params": [latents], "lr": lr * 0.25}])
    n = latents.shape[0]; g = torch.Generator().manual_seed(1); t0 = time.time()
    eh, ew = dec.eval_size
    for s in range(steps):
        idx = torch.randperm(n, generator=g)[:batch].to(device)
        orig = qat_pt(dec)
        frames = B.recon_frames_float(dec, latents[idx], (eh, ew))
        po, so = net(frames)
        restore_pt(dec, orig)
        seg_l = B.smooth_seg(so, seg_t[idx])
        pose_l = torch.sqrt(10 * (po["pose"][..., :6] - pose_t[idx]).pow(2).mean() + 1e-12)
        loss = 100 * seg_l + pose_w * pose_l
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if device.type == "mps" and s % 50 == 0:
            torch.mps.empty_cache()
        if s % log == 0 or s == steps - 1:
            print(f"  [C={meta['base_channels']}] polish {s:4d}  100*seg {100*seg_l.item():.4f}  "
                  f"pose {pose_l.item():.4f}  [{time.time()-t0:.0f}s]", flush=True)
    dec.eval()
    return {"decoder_sd": {k: v.detach().cpu() for k, v in dec.state_dict().items()},
            "latents": latents.detach().cpu(), "meta": meta}


def evaluate_student(ck, net, seg_gt, pose_gt, device, verbose=True):
    deq = dequant_int8(ck["decoder_sd"])
    dec_bytes = len(encode_int8_pertensor(ck["decoder_sd"]))
    lat_bytes = len(brotli.compress(B.encode_latents(ck["latents"]), quality=11))
    archive = dec_bytes + lat_bytes + B.META_BYTES
    dec = HNeRVDecoder(latent_dim=ck["meta"]["latent_dim"],
                       base_channels=ck["meta"]["base_channels"],
                       eval_size=tuple(ck["meta"]["eval_size"]))
    dec.load_state_dict(deq)
    r = B.score(net, dec, ck["latents"], device, archive, seg_gt, pose_gt)
    r.update(dec_bytes=dec_bytes, lat_bytes=lat_bytes,
             params=sum(v.numel() for v in ck["decoder_sd"].values()))
    if verbose:
        print(f"  [C={ck['meta']['base_channels']}] params={r['params']} dec={dec_bytes} "
              f"archive={archive}  seg={r['seg']:.6f}(100*={100*r['seg']:.4f}) "
              f"pose={r['pose']:.6f}(sqrt={math.sqrt(10*r['pose']):.4f}) "
              f"25*rate={25*r['rate']:.4f}  SCORE={r['score']:.4f}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="32,28,24")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--polish", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = B.get_device(args.device)
    print("device:", device, flush=True)
    net = B.build_net(device)
    if device.type == "mps":
        B.patch_bn_contiguous(net)
    seg_gt, pose_gt = B.compute_gt(net, device)
    print(f"\nBaseline teacher (C=36) for reference: ~0.197 (int8). Target: beat 0.192\n")
    results = {}
    for C in [int(x) for x in args.channels.split(",")]:
        ck, *_ = train_student(C, device, steps=args.steps, batch=args.batch)
        print(f"[C={C}] post-distill eval:")
        evaluate_student(ck, net, seg_gt, pose_gt, device)
        ck = polish_student(ck, net, seg_gt, pose_gt, device, steps=args.polish)
        print(f"[C={C}] post-polish eval:")
        r = evaluate_student(ck, net, seg_gt, pose_gt, device)
        torch.save(ck, f"work/student_C{C}.pt")
        results[C] = r["score"]
    print("\n=== student size sweep (lower=better; beat 0.192) ===")
    for C, sc in sorted(results.items()):
        print(f"  C={C}: {sc:.4f}")


if __name__ == "__main__":
    main()
