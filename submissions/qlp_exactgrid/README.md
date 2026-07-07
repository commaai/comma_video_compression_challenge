<!-- SPDX-License-Identifier: MIT -->

# qlp_exactgrid

Exact-grid quantization-aware **latent polish** of the frozen PR #95/#101 HNeRV
decoder, plus a re-searched frame0 pose selector. Trained and searched entirely
against the official frozen SegNet/PoseNet scorers on a single Apple M5 Pro (MPS);
no CUDA, no decoder training.

## Scores (official `evaluate.sh`, this archive)

| axis | seg | pose | bytes | rate | score |
|---|---|---|---|---|---|
| CPU (leaderboard) | 0.00056055 | 0.00002902 | 176,337 | 0.00469662 | **0.190506** |
| MPS (cross-check) | 0.00056055 | 0.00002902 | 176,337 | 0.00469662 | 0.190506 |

For reference: #125 `hnerv_qlp` 0.190946, #112 `rhnerv_comma` 0.191126. This
archive is smaller (176,337 vs 176,525 B) with better pose and tied seg.

## Method

1. **Frozen decoder.** The PR #95 HNeRV decoder with PR #101's fine-tuned weights
   is reused bit-for-bit (`model.py`); no decoder training was performed.
2. **Exact-grid latent polish** (the contribution). All 600×28 per-pair latents
   are optimized by gradient descent against the frozen scorers through the
   *exact* inflate chain (bicubic up to 874×1164, PR #98 channel biases,
   straight-through clamp/round), with a **boundary seg loss**
   `sigmoid(-margin/τ)` (the smooth argmax-flip fraction, τ annealed 0.20→0.07)
   so gradient concentrates on the pixels that actually flip the SegNet metric.
   Critically, the straight-through latent quantizer replicates the container's
   packing grid bit-for-bit (per-dim fp16 min/scale), so the polish that the
   optimizer sees is exactly what ships — no train/pack gap. PR #101's 607-byte
   latent-correction sidecar is dropped (its role is absorbed into the polish).
3. **Entropy coding.** Decoder and latent streams are coded with PR #112's context
   range coder (`codec_ctx.py`), reused unchanged.
4. **FS1B frame0 pose selector.** A per-pair frame0 pixel-mode selector (concept
   from PR #110's FEC6, re-searched for this payload against the exact PoseNet,
   35 modes, +48 B) improves pose. SegNet sees only frame1, so this is pose-only
   and seg-neutral by construction.

## Inflate

`inflate.sh <archive_dir> <out_dir> <video_list>` — CPU-pinned, deterministic on
a given machine; deps: `numpy`, `torch`, `constriction` (harness base env; no
network). `expected_output.sha256` is the canonical decode on the build machine
(Apple M5 Pro, arm64); x86 differs only in bicubic LSBs (~1.5e-7 seg, see PR #112).

## Reproduction

`bash compress.sh` rebuilds `archive.zip` byte-for-byte and asserts its SHA-256.
It fetches the two inputs — the frozen decoder (`decoder.pt`) and the polished
latents (`latents.pt`) — from this submission's release assets, runs the bundled
packer (`pack_base.py`, the PR #112 ctx container) to build the base member, then
appends the FS1B pose-selector tail from the committed `selection.json`
(`fs1b_compress.py`). Everything it needs is in this directory or the release; no
campaign work-tree is required.

The polished latents themselves are produced by the (non-deterministic, ~90 min
Apple M5 Pro / MPS) exact-grid boundary-loss latent-polish run described under
**Method** — a latents-only optimization against the frozen scorers with the
decoder held fixed.

## Attribution

See `THIRD_PARTY_NOTICES.md`. Built on the MIT-licensed chain PR #95 (decoder),
#98 (channel biases), #101 (fine-tuned weights + sidecar concept), #110 (frame
selector concept), #112 (context range coder). The latent-polish approach is
concurrent with and shares its core idea with PR #125 `hnerv_qlp` (@Bucky789);
this submission was developed independently from the public PR description and
adds an exact-grid quantizer, boundary-pixel seg loss, and re-searched selector.
