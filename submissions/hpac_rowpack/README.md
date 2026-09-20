# Lossless HPAC row repacking

This candidate changes the storage of integer weight rows in the existing [PR #135 F26 submission](https://github.com/commaai/comma_video_compression_challenge/pull/135). It restores the entire original model section byte for byte before the inherited decoder runs. It reuses the upstream trained models and compressed video representation; it does not claim independent training or a new semantic renderer.

The current archive is **185,767 bytes**, **957 bytes smaller** than the 186,724-byte reference. Its SHA-256 is:

```text
81918e585ce7df7de8a7412d391c2532fe5c6fe46848b38ef615b93210623581
```

**Local validation completed:** the unchanged official evaluator scored all 600 samples on Apple MPS at **0.20** (PyTorch 2.10.0, batch size 2). Official Linux/T4 evaluation and ranking remain pending. If reconstruction and distortion remain identical to the previously published result, the rate term would decrease by `25 × 957 / 37,545,489 = 0.000637227018137918`. Subtracting that from the prior published score predicts `0.1616311946814479`; this is a conditional calculation, not a measured score. Full local CUDA inflation and evaluation have not been performed.

## Storage change

The inherited model container stores signed integer HPAC weights at variable bit widths. The new `HRP1` representation:

1. Reads the original and reduced row-depth tables, retaining both unchanged.
2. Groups nonzero-width rows by their stored signed width, preserving row order within each group. Zero-width rows remain implicit zeros.
3. Maps signed values with reversible zigzag coding: nonnegative `x` becomes `2x`; negative `x` becomes `-2x−1`.
4. Writes each group as bitplanes, from least significant bit to most significant bit. This changes the byte patterns seen by the existing outer raw-LZMA compressor.
5. Retains the remaining HPAC fields and the rest of the model container unchanged. The new header records lengths and the CRC32 of the original complete model section.

The decoder reverses these operations, restores the original packed signed rows, validates padding, bounds, and CRC32, and passes the resulting `F24S` bytes to the inherited parser. The residual correction table and arithmetic token stream remain byte-identical. The transformation is lossless; the gain comes from improved storage of the existing weight values.

All learned and video-specific assets remain in the single counted ZIP member `p`: HPAC weights and metadata, semantic renderer weights, pose-carrier basis and coefficients, frame selector, residual table, and arithmetic token stream. Row-count constants in the decoder describe the existing architecture, not trained values. Inflation does not fetch assets or use external cached tokens.

## Rebuild and evaluate

Run from the challenge repository root with its Python environment activated. Repacking requires NumPy and the Python standard library. Inflation uses the existing PyTorch dependencies and a C compiler for the bundled native RC64 decoder.

Repack a local copy of the pinned upstream archive:

```bash
python submissions/hpac_rowpack/compress.py \
  --source /path/to/upstream/archive.zip
```

Or download the pinned reference release and repack it:

```bash
bash submissions/hpac_rowpack/compress.sh
```

The source ZIP must match SHA-256 `12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`. The command writes the new `archive.zip` and an `archive.report.json` documenting byte parity and size. It reconstructs the complete original model section and verifies the unchanged residual/token suffix before publishing the output file.

Run the fresh official evaluation on CUDA:

```bash
bash evaluate.sh --submission-dir submissions/hpac_rowpack --device cuda
```

The inflater defaults to CUDA and retains the inherited CUDA arithmetic. For an explicit MPS experiment:

```bash
HPAC_DEVICE=mps bash evaluate.sh \
  --submission-dir submissions/hpac_rowpack --device mps
```

The MPS path corrects a numerical issue: a power operation can approximate mathematically exact dyadic scale factors, changing subsequent half-integer rounding and arithmetic-decoder probabilities. For MPS only, the small scale-factor tensors are computed on CPU and copied to MPS. CPU and CUDA retain their existing calculation. The full 600-frame token decode passed the published token, corrected-logit, CDF-input, and compressed-stream hashes. Local rendering and evaluation completed as described below; CUDA pixel parity remains unmeasured.

## Validation and output accounting

```bash
python -m unittest discover -s submissions/hpac_rowpack/tests -v
```

For the tests involving the reference archive, set `HPAC_REFERENCE_ZIP=/path/to/upstream/archive.zip` if it is not available in the local experiment directory. Nine production tests passed locally, including deterministic archive rebuilding, exact decoded-asset parity with the reference, rejection of a changed reference, malformed-stream rejection, and payload accounting. The native C decoder compiled and loaded locally; this does not substitute for Linux/CUDA evaluation.

The parser permits exactly one ZIP member, bounds the member and decompressed model section to 1 MiB, and bounds the outer ZIP to 2 MiB. The extracted payload must equal the payload inside the actual `archive.zip` charged by the evaluator. The public artifact supports exactly `0.mkv`.

Inflation uses the production renderer and frame selector, then checks for exactly **1,200 RGB frames at 1164 × 874**, represented by **3,662,409,600 raw bytes**. A complete output replaces the final file only after validation; failed or incomplete output remains a `.partial` file. `inflate_report.json` records the actual archive hash, device, byte count, and token-decoder diagnostics. A cached-token experiment is not part of this submission or its inflation command.

## Completed local evaluation

| Item | Result |
| --- | ---: |
| Samples | 600 |
| Raw frames | 1,200 |
| Raw bytes | 3,662,409,600 |
| Archive bytes | 185,767 |
| PoseNet distortion | 0.00014749 |
| SegNet distortion | 0.00042714 |
| Rate | 0.00494778 |
| Score printed by official evaluator | 0.20 |

The local run used Apple MPS, Python 3.11.16 and PyTorch 2.10.0. The official evaluator, models, original video, and scoring formula were not changed. The local MPS result differs from the upstream CUDA result; exact entropy-token parity does not establish cross-device renderer or evaluator parity.

Validation was staged: a private porting probe ran the inherited entropy decoder with the exact-power correction, decoding all 117,964,800 tokens in 1,171.71 seconds. Its token SHA-256 matched the published `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`; corrected logits, CDF inputs, and stream hashes also matched. A separate diagnostic then rendered the full video from the new archive's production model/carrier/selector assets using that hash-verified token tensor. Rendering plus raw-size/hash verification took 100.93 seconds. Raw SHA-256: `56a7890a8d47a0486535234eda6dac4127d48726e0ae1656c9f911ce30794bf3`.

The unchanged official evaluator processed this full output in approximately 127 seconds after setup. This was not a fresh invocation of the public `inflate.sh` from ZIP through scoring, and no such claim is made. The public inflater includes full token decoding and has no external-cache path. Model/stream byte parity, the staged numerical checks, and source review support the implementation; the fresh official-compatible CUDA pipeline remains for validation.

## Attribution, license, and AI disclosure

The inherited source is pinned to PR #135 commit `6dcf77164ccbdcc1e0e41c99312e65ace4bc1fb4`; the challenge checkout is `db52c5a9f05d5298e314fb0fa130290c4a350c4e`. The [upstream release](https://github.com/codexblack/comma_video_compression_challenge/releases/download/semantic-pose-HPAC_CPR1_polished-f26/archive.zip) supplies the learned artifact. Its authors retain credit for the models, renderer, carrier, selector, and entropy coding. Original copied-file hashes appear in `UPSTREAM_FILES.json`; the MIT notice is retained in `LICENSE.upstream`. See `LINEAGE.md` for the scope of inherited and changed work.

AI agents generated the new implementation, experiments, tests, and this description at the user's direction. The user has not claimed to have personally written or reviewed this code. This disclosure does not change attribution of inherited work.
