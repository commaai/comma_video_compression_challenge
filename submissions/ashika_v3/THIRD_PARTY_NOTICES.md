# Third-Party Notices

This submission inherits from prior work in the contest repository. All
upstream code is reused under the contest repository's MIT license. This
document acknowledges the upstream contributions and identifies the
corresponding files in this submission.

The submission is a **lossless re-coding**: every bit of video-specific
information in its `archive.zip` originates in PR #101 and PR #110; only the
entropy-coding layer that carries that information is new.

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
- **What this submission uses**: the decoder weights, latent codes, and the
  607-byte latent-correction sidecar — i.e. the entire model content of this
  archive — are PR #101's, reused **information-identically**: the weight
  streams and latent payload are re-coded losslessly by the ctx coder (their
  raw bytes are reproduced bit-exactly at inflate time), and the sidecar is
  carried verbatim. The tensor-reconstruction grammar in `codec.py` and the
  sidecar decode in `codec_sidecar.py` are ports of PR #101's decode logic as
  published in-tree by PR #110. The offline encoder (`compress.sh`) fetches
  PR #101's archive from its release (SHA-256:
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`) and
  never redistributes it.

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
  and `codec_sidecar.py` keep fec6's function bodies verbatim with the
  entropy layer swapped (raw bytes in, no Brotli/LZMA). The offline encoder
  fetches PR #110's archive from its release (SHA-256:
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`) and
  never redistributes it.

## Open-source dependencies

- **constriction** (https://github.com/bamler-lab/constriction, MIT/Apache-2.0
  /BSL) — range coding primitives used by the ctx coder at both encode and
  inflate time.
- **Brotli** (RFC 7932; `brotli` PyPI package, MIT) — used at **encode time
  only**, to unwrap PR #101's source streams before re-coding. The inflate
  path of this submission does not use Brotli.
- **raw LZMA1** (`lzma` stdlib) — used at encode time only, to unwrap PR
  #101's latent payload before re-coding.

## This submission's contributions

- `codec_ctx.py` — context-modeled range coder for the three payload
  sections (adaptive geometric-primed weight models, causal-prediction latent
  models with discrete-Gaussian residuals, adaptive selector model), with
  platform-independent decode-side model construction. New code.
- `compress.py` / `compress.sh` — pinned-input encoder driver with byte-exact
  round-trip verification and deterministic packaging. New code.
- `inflate.py` — composes the ctx container decode with PR #101's tensor
  reconstruction and PR #110's transform chain.

All new code is MIT-licensed under the same terms as the contest repository.
