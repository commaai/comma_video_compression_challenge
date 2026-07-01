<!-- SPDX-License-Identifier: MIT -->

# rhnerv_comma_mgrid

Lossless **micro-recoding of the current #1 payload**. This submission keeps
the `rhnerv_comma` decoded video byte-identical on this CPU while shaving
56 additional archive bytes by using six decode-compatible changes:

- decoder `M_GRID`: add `4096.0`, saving 4 bytes through range-coder word alignment
- latent `LATENT_ADAPT_M`: `480.0 -> 1440.0`, saving 4 bytes through range-coder word alignment
- latent `MAX_CROSS`: `4 -> 5` cross-dim predictors, saving 4 bytes through better residual prediction
- container header: implicit decoder length (VERSION 2, `u16` latent + `u16` selector), saving 2 bytes
- decoder `SHARED_MODEL_TENSORS`: expand to `{3,5,7,9,11,13,15,17,19,21,23,25,27}`, saving 18 bytes through a tighter shared adaptive model
- selector Markov coding (`selector_markov.py`): first-order transition tables, saving 24 bytes vs order-0 adaptive

The decoded video remains byte-identical to PR
[#110](https://github.com/commaai/comma_video_compression_challenge/pull/110)
(`hnerv_fec6_fixed_huffman_k16`), whose information content (PR
[#101](https://github.com/commaai/comma_video_compression_challenge/pull/101)
decoder weights + latents + sidecar, plus #110's FEC6 selector) is carried in
1,395 fewer bytes by a context-modeled range coder (`codec_ctx.py`,
constriction-based): per-tensor adaptive 256-ary models with geometric-primed
priors for the decoder weight streams, per-dim causal prediction with
discrete-Gaussian residual models for the latents, and an adaptive 16-ary
model for the selector. No new training; distortion is unchanged by
construction.

## Archive identity

| Field | Value |
|---|---|
| Score (CPU, full precision) | `0.191088478` = 100·seg + sqrt(10·pose) + 25·rate |
| seg / pose (identical to #110) | `0.00056023` / `0.00002943` |
| rate | `0.00471641` (177,080 / 37,545,489) |
| Archive bytes | `177080` (#112: 177,136; saving −56 B, −0.000037 score) |
| Archive SHA-256 | `e82f97e5f5569b504e8efa0fc2ba5fd57ed2735a65be83b7025007c53a5e9d79` |
| ZIP members | 1 (`x`, `compression_type=0` ZIP_STORED, 176,980 bytes) |
| Member layout | ctx container 176,373 B (5-B v2 header + decoder 161,082 + latents 15,062 + selector 224) ++ verbatim 607-B #101 sidecar (trailing, length-implicit) |
| Inflate runtime deps | `numpy`, `torch`, `constriction` (all in the harness base env; no `brotli`) |
| Inflate GPU required | no (device pinned to CPU) |
| Inflate cost | ~2 m 50 s wall at 4 threads, peak RSS ≈ 2.3 GB (ctx entropy decode itself is < 1 s) |

## What is in the member

The ctx container losslessly re-codes the three entropy-coded sections of the
#110 member and reproduces, bit-exactly, (a) the 7 raw decoder weight streams
that #101's Brotli layer carried, (b) the 16,912-byte raw latent payload that
#101's raw-LZMA1 layer carried, and (c) the 249-byte FEC6 selector wire
payload of #110. The 607-byte #101 latent-correction sidecar is carried
verbatim (measured re-coding upside < ~7 B — not worth a format). From those
bytes to pixels, `inflate.py` runs the **exact** fec6 (#110) chain:
`HNeRVDecoder` (verbatim PR #95 architecture), 16-pair batches, bicubic
upsample to 874×1164 (`align_corners=False`), the #98 channel biases
(frame0 R−1, frame0 B−1, frame1 G−1), `clamp(0,255).round()`, the per-pair
FEC6 selector transforms (after bias+clamp+round, with a final batch
`clamp_/round_`), uint8 → NHWC, streamed write.

## Expected output SHA (machine-dependent LSBs)

`F.interpolate(mode='bicubic')` on CPU is bit-stable on a given machine but
its least-significant bits differ across CPU microarchitectures, so the same
archive yields different (but locally deterministic) `0.raw` hashes on
different hosts. Because this submission's decode math is identical to #110's,
its `0.raw` is byte-identical to #110's **on the same hardware**, whatever the
hardware:

| Machine | 0.raw SHA-256 |
|---|---|
| canonical here (AMD Zen 5, CPU; this repo's `expected_output.sha256`) | `e592c742dfd64fd8d8d42c83ad7066f8c012ed677cbb55644e1e57381106040a` |
| #110 author's machine (their `expected_output.sha256`) | `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c` |

Both machines reproduce the official #110 CI metrics to all 8 printed
decimals (seg `0.00056023`, pose `0.00002943`), so the metric gate is
hash-independent.

## Quick reproducibility check (≈60 s setup + ~3 min inflate, CPU only)

```bash
# 1) rebuild archive.zip from the pinned upstream releases (SHA-256-verified;
#    deterministic: IEEE-exact float64 model construction on any platform)
bash compress.sh           # or: --pr101 <local>/archive.zip --pr110 <local>/archive.zip
sha256sum archive.zip
# expect: e82f97e5f5569b504e8efa0fc2ba5fd57ed2735a65be83b7025007c53a5e9d79  (177,080 B)

# 2) inflate against this submission dir on CPU
mkdir -p /tmp/data /tmp/out
unzip -oq archive.zip -d /tmp/data
echo "0.mkv" > /tmp/list.txt
bash inflate.sh /tmp/data /tmp/out /tmp/list.txt

# 3) verify byte-stable decode against the canonical SHA for this machine
sha256sum /tmp/out/0.raw   # see expected_output.sha256 + table above
```

Or run the full harness from the challenge repo:

```bash
uv run --no-sync --group cpu bash evaluate.sh --device cpu --submission-dir <this dir>
```

## Files

| Path | Role |
|---|---|
| `compress.sh`, `compress.py` | Encoder: curl + SHA-check the pinned #101/#110 archives, entropy-decode their payloads to raw bytes, re-code with the ctx coder, byte-exact round-trip assert, deterministic zip. |
| `inflate.sh`, `inflate.py` | Contest-runtime decoder (accepts member `x` or `<base>.bin`). |
| `codec_ctx.py` | Context-modeled range coder (encoder + decoder; the new code). |
| `codec.py` | #101 tensor reconstruction, ported from fec6 to take raw stream bytes (entropy layer swapped, math verbatim). |
| `codec_sidecar.py` | #101 607-B enum-rank sidecar decode (verbatim fec6 bodies, other formats dropped). |
| `frame_selector.py` | Verbatim fec6 module (blue-chroma tile + transform families). |
| `model.py` | Verbatim fec6 copy of the PR #95 HNeRV decoder. |
| `expected_output.sha256` | Canonical CPU decode SHA on this machine (see table above). |
| `THIRD_PARTY_NOTICES.md` | Upstream attribution (PR #95 / #98 / #101 / #110). |

## Reproduction recipe

Inputs (not redistributed; fetched + SHA-256-pinned by `compress.sh`):
PR #101 `archive.zip` (`b83bf348…`) and PR #110 `archive.zip` (`6bae0201…`)
from their respective releases. `compress.py` cross-checks that #110 embeds
#101's source payload byte-for-byte before extracting the selector. The
encoder is deterministic and runs in seconds; the decode-side model
construction uses only IEEE-exact float64 multiply/add/divide, so encoder and
decoder build bit-identical probability tables on any IEEE-754 platform.
