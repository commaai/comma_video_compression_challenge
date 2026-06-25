#!/usr/bin/env python
"""Post-training quantization sweep (no retraining).

For each per-tensor symmetric bit level, quantize the extracted top-submission
decoder, measure (a) the brotli'd archive size and (b) the faithful score via
fasteval. This de-risks the whole plan: it shows the size/score Pareto reachable
with pure PTQ, before any QAT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import brotli
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "hnerv_muon" / "src"))
sys.path.insert(0, str(ROOT / "work"))

from model import HNeRVDecoder  # noqa: E402
import codec as muon_codec  # noqa: E402  (hnerv_muon/src/codec.py)
import fasteval  # noqa: E402


def quantize_dequantize(sd, n_quant):
    """Per-tensor symmetric quant to [-n_quant, n_quant], return (q_sd_for_encode, dequant_sd)."""
    q_sd = muon_codec.quantize_state_dict(sd, n_quant=n_quant)
    deq = {}
    for name, (q, scale, shape) in q_sd.items():
        t = torch.from_numpy(q.astype("float32")).reshape(shape) * scale
        deq[name] = t
    return q_sd, deq


def latents_size(latents):
    payload = muon_codec.encode_latents(latents)
    return len(brotli.compress(payload, quality=11))


def main():
    device = fasteval.get_device(sys.argv[1] if len(sys.argv) > 1 else None)
    print("device:", device)
    net = fasteval.build_distortion_net(device)
    seg_gt, pose_gt = fasteval.compute_gt_targets(net, device)

    blob = torch.load(ROOT / "work" / "base_decoder.pt")
    meta = blob["meta"]
    base_sd = blob["decoder_sd"]
    latents = blob["latents"]

    dec = HNeRVDecoder(latent_dim=meta["latent_dim"], base_channels=meta["base_channels"],
                       eval_size=tuple(meta["eval_size"]))

    lat_bytes = latents_size(latents)
    meta_bytes = 120  # approx for meta + framing, conservative
    print(f"latent blob (brotli): {lat_bytes} bytes; meta ~{meta_bytes}")

    # n_quant -> approx bits: 127=int8, 31=6b, 15=5b(~4.9), 7=4b, 5, 3=~3b
    levels = [127, 31, 15, 7, 5, 3]
    print(f"\n{'nq':>4} {'bits':>5} {'dec_bytes':>10} {'archive':>9} {'seg':>9} "
          f"{'pose':>9} {'rate':>8} {'SCORE':>8}")
    for nq in levels:
        q_sd, deq = quantize_dequantize(base_sd, nq)
        dec_bytes = len(muon_codec.encode_decoder(q_sd))
        archive = dec_bytes + lat_bytes + meta_bytes
        dec.load_state_dict(deq)
        r = fasteval.score_candidate(net, dec, latents, device, archive, seg_gt, pose_gt)
        import math
        bits = math.log2(2 * nq + 1)
        print(f"{nq:>4} {bits:>5.2f} {dec_bytes:>10} {archive:>9} {r['seg']:>9.6f} "
              f"{r['pose']:>9.6f} {r['rate']:>8.5f} {r['score']:>8.4f}")


if __name__ == "__main__":
    main()
