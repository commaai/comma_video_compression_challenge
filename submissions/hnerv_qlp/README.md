<!-- SPDX-License-Identifier: MIT -->

# hnerv_qlp — quantization-aware latent polish

Takes the current frontier payload (PR
[#112](https://github.com/commaai/comma_video_compression_challenge/pull/112)
`rhnerv_comma`, a lossless re-code of PR #110 / #101 / #98 / #95) and performs
one new step: with the decoder **frozen**, each frame pair's 28-dim latent is
re-optimized by gradient descent directly against the SegNet/PoseNet distortion,
through the exact inflate chain. The re-optimized latents then match the #1's
distortion **without** its 607-byte latent-correction sidecar — so the sidecar
is dropped, and the archive is 611 bytes smaller.

No decoder training. The decoder weights and the FEC6 selector are reused
bit-for-bit; only the latent codes change (and the sidecar is removed).

## Result

| | SegNet dist | PoseNet dist | bytes | **score (CPU)** |
|---|---|---|---|---|
| #1 `rhnerv_comma` | `0.00056023` | `0.00002943` | `177,136` | `0.191126` |
| **hnerv_qlp** | `0.00056085` | `0.00003000` | `176,525` | **`0.190946`** |

Margin **−0.000180** vs the official #1. Distortion is essentially unchanged
(the polish recovers what dropping the sidecar would otherwise cost); the win is
the −611-byte rate saving. Local CPU eval reproduces the #1's published metrics
to ~1e-5, so the margin (~18× that) is real. Score is reported on the CPU
(leaderboard) axis via `evaluate.py --device cpu`.

## Archive identity

**Download:** <https://github.com/Bucky789/comma_video_compression_challenge/releases/download/hnerv-qlp-v1/archive.zip>

| Field | Value |
|---|---|
| Archive bytes | `176,525` |
| Archive SHA-256 | `ebd513903bd598b4d73d699a11c89600bd11747f27aab26737e518096675b813` |
| ZIP members | 1 (`x`, `ZIP_STORED`, 176,425 B) |
| Member layout | ctx container only (7-B header + decoder 161,104 + latents 15,066 + selector 248); **no sidecar** |
| Inflate deps | `torch`, `numpy`, `constriction` |
| Inflate GPU required | no (device pinned to CPU) |

## The new idea

PR #101's sidecar already hinted at this: it searched **one** latent dimension
per pair over a small fixed step table and was worth ≈ −0.001. `hnerv_qlp`
generalizes that to **continuous gradient descent over all 28 dimensions** of
every pair's latent, against the frozen quantized decoder. Two details make it
land on the real leaderboard axis rather than a GPU proxy:

1. **fp32 evaluation and selection.** The SegNet distortion is a discrete
   argmax; fp16 and fp32 disagree on borderline pixels. Selecting latents on an
   fp16 scorer overfits GPU noise and regresses on the CPU axis (an earlier
   fp16-selected build scored `0.191955`). Evaluation and per-pair selection
   run in fp32; training stays fp16 for speed (gradients are smooth).
2. **Exact-chain optimization.** The polish forward pass replicates the inflate
   chain bit-for-bit — bicubic 874×1164, the #98 channel biases, clamp/round,
   the FEC6 per-pair selector — so the gradient targets the scored pixels.

## Inflate

```bash
unzip archive.zip -d /tmp/data      # -> /tmp/data/x
echo "0.mkv" > /tmp/list.txt
bash inflate.sh /tmp/data /tmp/out /tmp/list.txt
```

## Reproduce the encode

Offline; needs a CUDA GPU and the PR #101 / #110 archives (fetched from their
releases, SHA-256-verified, never redistributed):

```bash
# 1. extract the frozen decoder + latents + selector from the #110 payload
python encoder/extract_payload.py
# 2. quantization-aware latent polish (fp32 eval/selection, ~1 h on an RTX 4060)
python encoder/polish.py --epochs 150 --lr 5e-4 --batch 3 --eval-every 10
# 3. pack the polished latents into the ctx container (decoder/selector verbatim)
python encoder/pack.py --codes work/pilot2/best_codes.pt --out archive.zip
```

See `THIRD_PARTY_NOTICES.md` for the full lineage (PRs #95/#98/#101/#110/#112).

## Files

| Path | Role |
|---|---|
| `archive.zip` | The submission. |
| `inflate.sh`, `inflate.py` | Contest-runtime decoder (CPU, no sidecar). |
| `codec.py`, `codec_ctx.py`, `frame_selector.py`, `model.py` | Vendored decode runtime (PR #112 / #110 / #95). |
| `encoder/` | Offline extract → polish → pack pipeline (new). |
| `LICENSE`, `THIRD_PARTY_NOTICES.md` | MIT + upstream attribution. |
