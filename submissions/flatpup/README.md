# flatpup

`flatpup` is the final cleaned-up package for the best promoted quantizr-derived
submission.

Final promoted score: `0.317848`

Packaging verification: the same `archive.zip` SHA256
`67494f08f4637d5c21b182b4a9e7c65adf93e4b7f3d6ec078b88d00741e655c5`
rescored from this directory at `0.317862` in a later DALI/NVDEC run. The small
difference was confined to PoseNet jitter; the archive bytes and SegNet term
matched the promoted experiment.

Component breakdown:

- SegNet distortion: `0.00065235` -> `0.065235`
- PoseNet distortion: `0.00043372` -> `0.065858`
- Archive size: `284,396` bytes -> `0.186756`

This package is based on the best CRF50 reproduction, not on the later
higher-CRF follow-ups.

The submission format stays intentionally small:

- `archive.zip` contains compact stored entries `m`, `w`, and `p`.
- `inflate.py` only supports the compact promoted generator architecture.
- `inflate.sh` is a thin evaluator entry point.
- `compress.py` is a deterministic archive packer for already-trained payloads.

Main changes versus the historical quantizr baseline:

- Reproduced the old fp32/live-target training path with unseeded model
  initialization and fixed loader seed `123`.
- Added early lowpass-luma reconstruction on stable classes `0,2,4` with a
  weight of `0.2`, faded over the first `200` epochs.
- Added a tiny early PoseNet warmup (`0.001`, ramped over `80` epochs).
- Kept the compact payload stack: `libaom-av1 --cpu-used 0` masks, `QZMB1`
  compact model export, and delta-varint pose payloads.

Experiments tried but not shipped:

- Higher CRF masks reduced bytes but crossed the SegNet distortion knee. The
  best full-stage CRF51 follow-up scored `0.319933`; CRF56 scored `0.331996`.
  PoseNet was repairable, but the SegNet penalty outweighed the mask savings.
- Skipping the anchor-boost stage at CRF50 improved PoseNet slightly but hurt
  SegNet enough to score `0.319387`, worse than the promoted package.
- Larger or more expressive model variants, including mask repair features,
  one-hot/edge mask features, residual adapters, affine heads, and fiducial or
  texture-style PoseNet hints, did not beat the compact default architecture.
  Most either increased bytes, destabilized PoseNet, or failed to improve the
  decoded-mask SegNet term.
- AV1-only mask studies and class-permutation probes were useful diagnostics,
  but they did not produce a promoted payload.
- Seeded, AMP, teacher-cache, and transfer variants helped expose reproducibility
  traps, but the historical-quality result required the old fp32/live-target
  training path with unseeded model initialization.
