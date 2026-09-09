"""Is the seg surrogate actually pushing on the pixels that are wrong?

sigmoid(-margin/tau) saturates: a pixel that is *badly* wrong (very negative
margin) gets almost no gradient.  If the disagreeing pixels sit far below
-tau, the loss is blind to exactly the pixels the metric charges me for.
"""
import argparse, sys, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
torch.set_num_threads(2)
from common import WORK, Decoder, load_distortion_net, build_roundtrip, apply_roundtrip
from train import Targets

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--run", default="D")
ap.add_argument("--ckpt", default="best.pt")
args = ap.parse_args()

ck = torch.load(WORK / "runs" / args.run / args.ckpt, map_location="cpu")
tg = Targets(); net = load_distortion_net("cpu"); Ah, Aw = build_roundtrip()
dec = Decoder(latent_dim=ck["lat"].shape[1], base_channels=40).eval()
dec.load_state_dict(ck["dec"])

idx = np.linspace(0, tg.n - 1, 12).astype(int)
margins_wrong, margins_all = [], []
with torch.inference_mode():
    for s in range(0, len(idx), 4):
        b = idx[s:s + 4]
        out = apply_roundtrip(dec(torch.as_tensor(ck["lat"][b])), Ah, Aw)
        out = out.clamp(0, 255).round().permute(0, 1, 3, 4, 2)
        _, sin = net.preprocess_input(out)
        logits = net.segnet(sin)
        tgt = torch.from_numpy(np.ascontiguousarray(tg.seg[b])).long()
        tl = logits.gather(1, tgt.unsqueeze(1))
        masked = logits.masked_fill(
            torch.nn.functional.one_hot(tgt, logits.shape[1]).permute(0, 3, 1, 2).bool(), -1e9)
        margin = (tl - masked.max(1, keepdim=True)[0]).squeeze(1)
        wrong = logits.argmax(1) != tgt
        margins_wrong.append(margin[wrong]); margins_all.append(margin.flatten())

mw = torch.cat(margins_wrong); ma = torch.cat(margins_all)
print(f"disagreeing pixels: {mw.numel():,} / {ma.numel():,} = {mw.numel()/ma.numel():.5f}")
print(f"margin of WRONG pixels: mean {mw.mean():.3f}  median {mw.median():.3f}")
for q in (0.1, 0.25, 0.5, 0.75, 0.9):
    print(f"   q{q:.2f} = {mw.quantile(q):.3f}")
for tau in (0.3, 1.0, 3.0):
    g = torch.sigmoid(-mw / tau) * (1 - torch.sigmoid(-mw / tau)) / tau
    frac = (mw > -tau).float().mean()
    print(f"tau={tau}: mean |dL/dmargin| on wrong px = {g.mean():.5f}   "
          f"frac of wrong px within one tau of boundary = {frac:.3f}")
