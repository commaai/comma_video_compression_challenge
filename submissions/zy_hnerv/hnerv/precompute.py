"""Decode 0.mkv once, cache SegNet/PoseNet targets + 384x512 GT frames.

Also runs cheap 'oracle' probes: feed *degraded ground truth* through the real
metric and see what distortion each degradation costs.  That gives a hard floor
for design choices (render resolution, bit depth) before spending hours training.
"""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
torch.set_num_threads(2)
from common import (REPO, CACHE, load_distortion_net, build_roundtrip, apply_roundtrip,
                    EVAL_H, EVAL_W, CAM_H, CAM_W)

CACHE.mkdir(parents=True, exist_ok=True)
net = load_distortion_net("cpu")
Ah, Aw = build_roundtrip()

import av
from frame_utils import yuv420_to_rgb


def pair_iter():
    container = av.open(str(REPO / "videos" / "0.mkv"))
    prev = None
    for frame in container.decode(container.streams.video[0]):
        arr = yuv420_to_rgb(frame)               # (874,1164,3) uint8
        if prev is None:
            prev = arr
            continue
        yield torch.stack([prev, arr])           # (2,874,1164,3) uint8
        prev = None
    container.close()


def metric_of(pair_full_u8):
    """pair as (B,2,H,W,3) float tensor at CAMERA size -> (seg_logits, pose6)."""
    with torch.inference_mode():
        pin, sin = net.preprocess_input(pair_full_u8)
        return net.segnet(sin), net.posenet(pin)["pose"][:, :6]


seg_t = np.lib.format.open_memmap(CACHE / "seg_targets.npy", mode="w+",
                                  dtype=np.uint8, shape=(600, EVAL_H, EVAL_W))
gt384 = np.lib.format.open_memmap(CACHE / "gt384.npy", mode="w+",
                                  dtype=np.uint8, shape=(600, 2, 3, EVAL_H, EVAL_W))
pose_t = np.zeros((600, 6), dtype=np.float32)

# oracle probes accumulate on a subset
probe_n = 60
probe = {k: [0.0, 0.0] for k in ("roundtrip", "render192", "render128", "u8quant")}

t0 = time.time()
n = 0
for pair in pair_iter():
    x = pair.unsqueeze(0).float()                                  # (1,2,874,1164,3)
    seg_logits, pose6 = metric_of(x)
    seg_t[n] = seg_logits.argmax(1)[0].to(torch.uint8).numpy()
    pose_t[n] = pose6[0].numpy()

    # GT at eval resolution: exactly what the harness feeds the nets
    chw = x[0].permute(0, 3, 1, 2)                                 # (2,3,874,1164)
    small = F.interpolate(chw, size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False)
    gt384[n] = small.round().clamp(0, 255).to(torch.uint8).numpy()

    if n < probe_n:
        def dist_of(cand_384):
            """cand at 384x512 float -> (seg_disagree, pose_mse) against this pair's GT."""
            c = cand_384.unsqueeze(0).permute(0, 1, 3, 4, 2).clamp(0, 255).round()
            with torch.inference_mode():
                pin, sin = net.preprocess_input(c)
                s = net.segnet(sin); p = net.posenet(pin)["pose"][:, :6]
            seg_d = (s.argmax(1) != seg_logits.argmax(1)).float().mean().item()
            pose_d = (p - pose6).square().mean().item()
            return seg_d, pose_d

        base = small.clone()
        # 1. what the inflate->harness resize round trip alone costs
        a, b = dist_of(apply_roundtrip(base, Ah, Aw)); probe["roundtrip"][0] += a; probe["roundtrip"][1] += b
        # 2. rendering at 192x256 then bilinear x2 (the cheap decoder option)
        for tag, size in (("render192", (192, 256)), ("render128", (128, 170))):
            lo = F.interpolate(base, size=size, mode="bilinear", align_corners=False)
            hi = F.interpolate(lo, size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False)
            a, b = dist_of(apply_roundtrip(hi, Ah, Aw)); probe[tag][0] += a; probe[tag][1] += b
        # 3. uint8 rounding alone
        a, b = dist_of(apply_roundtrip(base.round(), Ah, Aw)); probe["u8quant"][0] += a; probe["u8quant"][1] += b

    n += 1
    if n % 100 == 0:
        print(f"  {n} pairs  ({time.time()-t0:.0f}s)", flush=True)

np.save(CACHE / "pose_targets.npy", pose_t[:n])
seg_t.flush(); gt384.flush()
print(f"done: {n} pairs in {time.time()-t0:.0f}s", flush=True)

print("\n=== oracle probes (degrade GROUND TRUTH, measure real metric) ===")
print(f"{'degradation':<14} {'seg_dist':>12} {'pose_dist':>12} {'score contrib':>14}")
import math
for k, (a, b) in probe.items():
    a, b = a / probe_n, b / probe_n
    print(f"{k:<14} {a:>12.6f} {b:>12.8f} {100*a + math.sqrt(10*b):>14.4f}")
