"""Post-training rate/distortion polish.

The training loop fixes INT8 weights and 255-level latents.  Neither is
necessarily score-optimal: coarser quantisation shrinks the archive (worth
25*bytes/37.5M) while adding distortion (worth 100*seg + sqrt(10*pose)).  Sweep
the grid and keep whichever point actually minimises the real score.
"""
import argparse, io, json, struct, sys
import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
torch.set_num_threads(2)
from common import (WORK, CACHE, Decoder, load_distortion_net, build_roundtrip,
                    apply_roundtrip, decode_weights, decode_latents, score_of,
                    _zig8, encode_latents)
from train import Targets, true_score


def encode_weights_n(sd, n_levels):
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(sd)))
    for name, t in sd.items():
        t = t.detach().cpu().float()
        m = t.abs().max().item()
        scale = m / n_levels if m > 0 else 1.0
        q = (t / scale).round().clamp(-n_levels, n_levels).to(torch.int8).numpy().ravel()
        nb = name.encode()
        buf.write(struct.pack("<I", len(nb))); buf.write(nb)
        buf.write(struct.pack("<I", len(t.shape)))
        for s in t.shape:
            buf.write(struct.pack("<I", s))
        buf.write(struct.pack("<f", scale))
        buf.write(struct.pack("<I", q.size))
        buf.write(_zig8(q).tobytes())
    import brotli
    return brotli.compress(buf.getvalue(), quality=11)


def encode_latents_n(lat, levels):
    import brotli
    t = lat.detach().cpu().float()
    n, d = t.shape
    mins, maxs = t.min(0).values, t.max(0).values
    scales = ((maxs - mins) / levels).clamp(min=1e-10)
    q = ((t - mins) / scales).round().clamp(0, levels).to(torch.uint8).numpy()
    delta = np.empty_like(q, dtype=np.int16)
    delta[0] = q[0]
    delta[1:] = q[1:].astype(np.int16) - q[:-1].astype(np.int16)
    zz = np.where(delta >= 0, 2 * delta, -2 * delta - 1).astype(np.uint16)
    payload = struct.pack("<II", n, d)
    payload += mins.to(torch.float16).numpy().tobytes()
    payload += scales.to(torch.float16).numpy().tobytes()
    payload += (zz & 0xFF).astype(np.uint8).tobytes() + (zz >> 8).astype(np.uint8).tobytes()
    return brotli.compress(payload, quality=11)


def build(sd, lat, meta, wn, ln):
    import brotli
    mb = brotli.compress(json.dumps(meta).encode(), quality=11)
    wb = encode_weights_n(sd, wn)
    lb = encode_latents_n(lat, ln)
    out = io.BytesIO()
    for p in (mb, wb, lb):
        out.write(struct.pack("<I", len(p))); out.write(p)
    return out.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--eval-pairs", type=int, default=64)
    ap.add_argument("--out", default="final_archive.bin")
    args = ap.parse_args()

    ck = torch.load(WORK / "runs" / args.run / args.ckpt, map_location="cpu")
    sd, lat = ck["dec"], ck["lat"]
    meta = {"latent_dim": lat.shape[1], "base_channels": 40, "n_pairs": int(lat.shape[0])}

    tg = Targets()
    net = load_distortion_net("cpu")
    Ah, Aw = build_roundtrip()
    idx = np.linspace(0, tg.n - 1, args.eval_pairs).astype(int)

    best = None
    print(f"{'w_lv':>5} {'l_lv':>5} {'bytes':>9} {'seg':>10} {'pose':>12} {'score':>8}")
    for wn in (127, 63, 31, 15):
        for ln in (255, 127, 63):
            blob = build(sd, lat, meta, wn, ln)
            buf = io.BytesIO(blob); parts = []
            for _ in range(3):
                n = struct.unpack("<I", buf.read(4))[0]; parts.append(buf.read(n))
            dsd = decode_weights(parts[1]); dlat = decode_latents(parts[2])
            dec = Decoder(latent_dim=meta["latent_dim"], base_channels=40).eval()
            dec.load_state_dict(dsd)
            seg, pose = true_score(dec, dlat, net, tg, Ah, Aw, idx)
            sc = score_of(seg, pose, len(blob))
            print(f"{wn:>5} {ln:>5} {len(blob):>9,} {seg:>10.6f} {pose:>12.8f} {sc:>8.4f}", flush=True)
            if best is None or sc < best[0]:
                best = (sc, wn, ln, blob, seg, pose)

    sc, wn, ln, blob, seg, pose = best
    p = WORK / "runs" / args.run / args.out
    p.write_bytes(blob)
    print(f"\nBEST w={wn} l={ln} bytes={len(blob):,} seg={seg:.6f} pose={pose:.8f} score={sc:.4f}")
    print(f"written to {p}")


if __name__ == "__main__":
    main()
