# Third-Party Notices

This submission inherits from prior work in the contest repository. All
upstream code is reused under the contest repository's MIT license. This
document acknowledges the upstream contributions and identifies the
corresponding files in this submission.

This submission modifies only the stored latent codes: PR #101's 607-byte
sidecar corrections are folded into them, and 2,162 net one- and two-step code
adjustments were applied, each verified against the exact contest score.
Decoder weight bytes remain byte-identical to PR #101/#98/#95, the FEC6
selector to PR #110, and the container/entropy-coding layer plus decode
runtime to PR #112 (`rhnerv_comma`), minus the now-unused sidecar path.

## PR #95 — HNeRV decoder

- **Author**: @AaronLeslie138
- **PR**: https://github.com/commaai/comma_video_compression_challenge/pull/95
- **License**: MIT (inherited from the contest repository)
- **What this submission uses**: the HNeRV-style decoder architecture (229K
  parameters, per-frame-pair latent → 6 upsample stages → 384×512 RGB pair).
  `model.py` is a verbatim copy of the fec6 (#110) copy, which is
  byte-identical to PR #95's implementation. No new training was performed.

## PR #98 — finetuned weights + channel-bias inflate

- **Author**: @EthanYangTW
- **PR**: https://github.com/commaai/comma_video_compression_challenge/pull/98
- **License**: MIT (inherited from the contest repository)
- **What this submission uses**: the decoder weights and latents that #101
  packs are byte-identical to #98's finetune of the #95 weights, and the
  per-pair channel-bias step in `inflate.py` (frame0 R−1, frame0 B−1,
  frame1 G−1, applied before clamp/round) originates in #98's inflate.

## PR #101 — `hnerv_ft_microcodec` payload content

- **Author**: @SajayR
- **PR**: https://github.com/commaai/comma_video_compression_challenge/pull/101
- **License**: MIT (inherited from the contest repository)
- **What this submission uses**: the decoder weights (byte-identical) and
  the latent codes and sidecar corrections as the starting point of this
  submission's latent adjustments (the sidecar's effect is folded into the
  stored codes; the 607-byte sidecar itself is no longer carried). The
  tensor-reconstruction grammar in `codec.py` is PR #101's decode logic as
  published in-tree by PR #110/#112.

## PR #110 — `hnerv_fec6_fixed_huffman_k16` selector + inflate chain

- **Author**: @adpena
- **PR**: https://github.com/commaai/comma_video_compression_challenge/pull/110
- **License**: MIT (sole-author Alejandro Peña, per its in-tree LICENSE)
- **What this submission uses**: the FEC6 K=16 per-pair selector **content**
  (its 249-byte wire payload is regenerated bit-exactly by the ctx selector
  section) and the complete inflate transform chain — batching, bicubic
  upsample, clamp/round ordering, and the per-pair selector application order
  (transforms after bias+clamp+round, then a final batch clamp/round).
  `inflate.py` is derived from fec6's `inflate.py` (FEC6 mode table, Huffman
  decode, and transform-apply functions verbatim; container parsing replaced,
  device pinned to CPU); `frame_selector.py` is a verbatim copy; `codec.py`
  keeps fec6's function bodies verbatim with the entropy layer swapped (raw
  bytes in, no Brotli/LZMA).

## PR #112 — `rhnerv_comma` container + ctx entropy coder

- **PR**: https://github.com/commaai/comma_video_compression_challenge/pull/112
- **License**: MIT
- **What this submission uses**: the ctx container format and context-modeled
  range coder (`codec_ctx.py`, verbatim), the raw-stream tensor
  reconstruction (`codec.py`, verbatim), and the decode chain in `inflate.py`
  (verbatim minus the latent-sidecar branch, which this payload no longer
  needs).

## This submission's contributions

- The 2,162 net latent-code adjustments (±1/±2 grid steps, CPU-axis selected) themselves, found by an
  exact-score-gated discrete search (every candidate adjustment was scored
  with the official SegNet/PoseNet evaluators and the real re-encoded archive
  size; only exact improvements were kept), and the folding of PR #101's
  sidecar corrections into the stored codes (removing the 607-byte sidecar).
- `inflate.py` container parsing without the sidecar branch.

- `compress.py` / `compress.sh` — a deterministic encoder that re-runs the ctx
  coder on the raw inputs in `encoder/` to rebuild `archive.zip` byte-for-byte.

Inflate-time dependencies are `numpy`, `torch`, and **constriction**
(https://github.com/bamler-lab/constriction, MIT/Apache-2.0/BSL; range-coding
primitives used by the ctx coder), all in the harness base env.

## Concurrent independent work

A per-pair latent-polish idea appears concurrently in PR #125 (`hnerv_qlp`)
and in another open PR using an exact-grid quantization-aware gradient polish.
Those approaches optimize the same latent codes by **gradient descent** through
the inflate chain; this submission instead uses an **exact-score-gated
discrete search** (axis-aligned ±1/±2 code steps, each accepted only on a
verified improvement of the official contest score, selected on the CPU
leaderboard axis). The submissions were
developed independently; this note acknowledges the parallel direction.

All new code is MIT-licensed under the same terms as the contest repository.
