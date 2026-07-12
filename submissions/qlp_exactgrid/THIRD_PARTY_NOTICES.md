# Third-party notices

All upstream works are MIT-licensed submissions to this repository
(commaai/comma_video_compression_challenge). This submission **imports** the
shared decode/codec modules directly from the merged `submissions/rhnerv_comma/`
(PR #112) rather than vendoring copies — `codec_ctx.py`, `codec.py`, and
`model.py` are not duplicated here.

- **PR #95 — `hnerv_muon` (@AaronLeslie138)**: HNeRV decoder architecture
  (`model.py`, imported from #112). https://github.com/commaai/comma_video_compression_challenge/pull/95
- **PR #98 — channel-bias correction (@AaronLeslie138)**: the frame0 R−1/B−1,
  frame1 G−1 decode biases, reused in the inflate chain.
- **PR #101 — `hnerv_ft_microcodec` (@SajayR)**: the fine-tuned decoder weights
  used as the frozen decoder + polish initialization; tensor payload
  reconstruction (`codec.py`, imported from #112).
  https://github.com/commaai/comma_video_compression_challenge/pull/101
- **PR #110 — `hnerv_fec6_fixed_huffman_k16` (@adpena)**: the per-pair frame0
  perturbation-selector concept our researched pose selector builds on.
  https://github.com/commaai/comma_video_compression_challenge/pull/110
- **PR #112 — `rhnerv_comma` (@mattneel)**: the context-modeled range coder
  (`codec_ctx.py`) and base-member container format, imported unchanged from
  `submissions/rhnerv_comma/`.
  https://github.com/commaai/comma_video_compression_challenge/pull/112
- **PR #125 — `hnerv_qlp` (@Bucky789)**: independent, concurrent work sharing the
  core idea of quantization-aware gradient polish of the per-pair latents against
  the frozen scorers with the #101 sidecar dropped. This submission was developed
  from the public PR description and differs in implementation: an exact packing-
  grid straight-through quantizer (no train/pack gap), a boundary-pixel seg loss,
  a rate-aware QAT fine-tune init, and a re-searched frame0 pose selector.
  https://github.com/commaai/comma_video_compression_challenge/pull/125
- **constriction** (Bamler; MIT/Apache-2.0): range-coding backend.
- **PyTorch, NumPy**: runtime dependencies (BSD-style licenses).

New code in this submission (MIT, © 2026 Ryan Li): the exact-grid boundary-loss
latent polish (`train_qlp3.py`), the FS1B tail format (`fs1b.py`,
`fs1b_palette.py`), the selector search, and the composed `inflate.py`.
