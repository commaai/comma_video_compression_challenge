<!-- SPDX-License-Identifier: MIT -->
# Third-Party Notices

This submission builds directly on prior work in the contest repository, all
reused under the repository's MIT license. It takes the current-frontier
payload (PR #112, itself a lossless re-code of PR #110 over PR #101 over
PR #98/#95) and performs one new step: a **quantization-aware gradient polish
of the per-pair latents** against the frozen decoder, followed by re-encoding.
The decoder weights and the frame selector are reused information-identically;
only the latent codes change, and the 607-byte latent sidecar is dropped.

## PR #95 — HNeRV decoder

- **Author**: @AaronLeslie138
- **PR**: https://github.com/commaai/comma_video_compression_challenge/pull/95
- **License**: MIT (inherited from the contest repository)
- **What this submission uses**: the HNeRV decoder architecture and its
  trained weights (229K params, per-pair latent → 6 upsample stages → 384×512
  RGB pair). `model.py` is a verbatim copy. The weights are reused **byte-for-
  byte** — the polish freezes them and never updates a single decoder
  parameter. No decoder training was performed.

## PR #98 — finetuned weights + channel-bias inflate

- **Author**: @EthanYangTW
- **PR**: https://github.com/commaai/comma_video_compression_challenge/pull/98
- **License**: MIT (inherited from the contest repository)
- **What this submission uses**: the frozen decoder weights are #98's
  finetune of #95, and the per-pair channel-bias step in `inflate.py`
  (frame0 R−1, frame0 B−1, frame1 G−1, before clamp/round) originates in #98.
  This bias is replicated inside the polish forward pass so the latents are
  optimized against the exact inflate math.

## PR #101 — `hnerv_ft_microcodec` payload substrate

- **Author**: @SajayR
- **PR**: https://github.com/commaai/comma_video_compression_challenge/pull/101
- **License**: MIT (inherited from the contest repository)
- **What this submission uses**: the decoder weight streams (reused
  bit-exactly) and the **latent quantization grammar** — per-dim fp16
  min/scale grid + dim-major temporal-delta uint8 codes. Our polished latents
  are re-encoded into this exact grammar (`encoder/latent_codec.py` is the
  inverse of #101's `decode_latents_compact`, verified byte-exact on the
  original payload). `codec.py` is #101's decode logic as published in-tree by
  #110. **Not used**: #101's 607-byte 1-dim latent-correction sidecar — the
  polish optimizes all 28 dims per pair and subsumes it. The offline encoder
  fetches #101's archive from its release (SHA-256
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`) and
  never redistributes it.

## PR #110 — `hnerv_fec6_fixed_huffman_k16` selector + inflate chain

- **Author**: @adpena
- **PR**: https://github.com/commaai/comma_video_compression_challenge/pull/110
- **License**: MIT (sole-author Alejandro Peña, per its in-tree LICENSE)
- **What this submission uses**: the FEC6 K=16 per-pair selector — its 249-byte
  wire payload is reused **bit-exactly** — and the complete inflate transform
  chain (batching, bicubic upsample, clamp/round ordering, per-pair selector
  applied after bias+clamp+round then a final batch clamp/round). Both are
  also replicated inside the polish forward pass so the latents are optimized
  against the selector-transformed, scored pixels. `frame_selector.py` is a
  verbatim copy; `inflate.py` is derived from fec6's (selector decode +
  transform apply verbatim; container parse swapped for #112's; sidecar
  removed). The encoder fetches #110's archive from its release (SHA-256
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`) and
  never redistributes it.

## PR #112 — `rhnerv_comma` context-model container

- **Author**: @mattneel (Matt Neel)
- **PR**: https://github.com/commaai/comma_video_compression_challenge/pull/112
- **License**: MIT (inherited from the contest repository)
- **What this submission uses**: the context-modeled range-coder container
  (`codec_ctx.py`, verbatim) that carries the decoder / latent / selector
  sections, and the member/inflate scaffolding. We re-encode the **new**
  polished latent section through the same `encode_latent_section`, keep the
  decoder and selector sections byte-identical to #112's, and drop the trailing
  sidecar. `inflate.py` and `codec.py` are #112's with the sidecar path
  removed.

## Open-source dependencies

- **constriction** (https://github.com/bamler-lab/constriction, MIT/Apache-2.0
  /BSL) — range coding primitives, used at both encode and inflate time.
- **PyTorch** (BSD-3) — the frozen decoder and scorer networks (encode-time
  polish) and the decoder (inflate time).
- **Brotli** (`brotli` PyPI, MIT) and **raw LZMA1** (`lzma` stdlib) — used at
  **encode time only**, to unwrap #101's source streams before re-coding. The
  inflate path uses neither.

## This submission's new contribution

Quantization-aware latent polish: with the decoder frozen, each frame pair's
28-dim latent is optimized by gradient descent (straight-through estimator on
the #101 uint8 grid) directly against the SegNet/PoseNet distortion through the
exact #110/#112 inflate chain. This is a strict generalization of #101's
sidecar (which searched 1 dim × a fixed step table per pair) to continuous
joint optimization over all 28 dims. `encoder/polish.py`, `encoder/pack.py`,
`encoder/extract_payload.py`, `encoder/latent_codec.py` are new.
