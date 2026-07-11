<!-- SPDX-License-Identifier: MIT -->

# rhnerv_latent_polish

Exact-score-gated **latent polish** of the current #1 payload
([#112](https://github.com/commaai/comma_video_compression_challenge/pull/112)
`rhnerv_comma`). Decoder weight bytes are byte-identical to PR
[#101](https://github.com/commaai/comma_video_compression_challenge/pull/101)/[#95](https://github.com/commaai/comma_video_compression_challenge/pull/95),
the FEC6 selector to PR
[#110](https://github.com/commaai/comma_video_compression_challenge/pull/110),
and the container/entropy coder and decode chain to PR #112. Two changes,
both confined to the stored latent codes:

1. **Sidecar folded in**: PR #101's 607-byte latent-correction sidecar is
   absorbed into the stored 8-bit latent codes and dropped from the archive.
2. **2,198 net latent code changes (±1/±2 grid steps)**, found by a discrete
   search in which every candidate step was scored with the official
   SegNet/PoseNet evaluators and the real re-encoded archive size, and kept
   only if the exact contest score improved. A single adjustment affects only
   its own frame pair, so candidates were evaluated exactly (600 per batched
   render) rather than estimated; accepted sets compose exactly across pairs.
   No gradient step was ever applied to stored values; no new training.

The search ran in two stages: a ±1-step search to plateau (14 rounds), then
six re-selection rounds **on the CPU axis** (the leaderboard axis; fp32,
fixed batch layout) with ±1 and ±2 steps, which re-audit earlier steps (a
reversal is just a step in the opposite direction) and remove the
GPU-vs-CPU bicubic-LSB selection bias. See `METHOD.md` for a step-by-step
walkthrough with a worked example.

## Archive identity

| Field | Value |
|---|---|
| Score (CPU, full precision) | `0.187946` = 100·seg + sqrt(10·pose) + 25·rate |
| seg / pose | `0.00053263` / `0.00002937` |
| rate | `0.00470179` (176,531 / 37,545,489) |
| Archive bytes | `176531` (#112: 177,136; −605 B, seg −4.9%, pose −0.2%) |
| Archive SHA-256 | `6d11284b051540be190b2613e615edad4efec7fddbfc627000d0d5fd0bd3f859` |
| Member SHA-256 | `2687049683aae7848bc9d0a23feb8809efe7875d8cf0ba34e58f41ab538f7827` |
| ZIP members | 1 (`x`, `compression_type=0` ZIP_STORED, 176,431 bytes) |
| Member layout | ctx container (7-B header + decoder 161,104 + latents + selector); **no trailing sidecar** |
| Inflate runtime deps | `numpy`, `torch`, `constriction` (harness base env) |
| Inflate GPU required | no (device pinned to CPU) |

## Quick reproducibility check (CPU only)

```bash
mkdir -p /tmp/data /tmp/out
unzip -oq archive.zip -d /tmp/data
echo "0.mkv" > /tmp/list.txt
bash inflate.sh /tmp/data /tmp/out /tmp/list.txt
sha256sum /tmp/out/0.raw   # see expected_output.sha256 (machine-dependent LSBs, see #112 README)
```

Or the full harness:

```bash
bash evaluate.sh --submission-dir ./submissions/rhnerv_latent_polish --device cpu
```

`F.interpolate(mode='bicubic')` LSBs differ across CPU microarchitectures
(see #112's README); metrics reproduce machine-independently.

## Files

| Path | Role |
|---|---|
| `inflate.sh`, `inflate.py` | Contest-runtime decoder (#112 chain minus the sidecar branch). |
| `compress.sh`, `compress.py` | Deterministic encoder: re-runs the ctx coder on `encoder/` inputs to rebuild `archive.zip` byte-for-byte (asserts member + archive SHA-256). |
| `encoder/decoder_streams.bin` | Raw HNeRV decoder weight streams, verbatim #101/#95 (frozen). |
| `encoder/selector_payload.bin` | Raw FEC6 selector wire payload, verbatim #110 (frozen). |
| `encoder/polished_latent_raw.bin` | This submission's polished per-pair latent payload (sidecar folded in; 2,198 net verified ±1/±2 code steps across 583 of 600 pairs). |
| `METHOD.md` | Step-by-step method walkthrough with a worked example. |
| `codec_ctx.py` | #112's context-modeled range coder (verbatim). |
| `codec.py` | #101 tensor reconstruction (verbatim #112 copy). |
| `frame_selector.py` | #110 FEC6 selector module (verbatim). |
| `model.py` | PR #95 HNeRV decoder (verbatim). |
| `expected_output.sha256` | Canonical CPU decode SHA on the author's machine. |
| `THIRD_PARTY_NOTICES.md` | Upstream attribution (PR #95 / #98 / #101 / #110 / #112). |

## Method note

The starting payload sits at a sharp optimum: gradient fine-tuning (even
entropy-regularized QAT with straight-through rounding) and any zero-shot
re-quantization measurably worsen the exact score, and one-step *weight*
code moves are all individually harmful (verified exhaustively down to
single flips). The stored latents are the one section with residual slack,
and only an exact-acceptance search can harvest it safely: distortion
deltas are computed exactly per candidate (pair-local rendering), byte
deltas by re-encoding with the real coder at accept time, and every
accepted set is re-verified end to end before being kept.
