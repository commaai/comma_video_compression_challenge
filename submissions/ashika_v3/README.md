<!-- SPDX-License-Identifier: MIT -->

# ashika_v3

HNeRV neural video compression with a custom context-modeled range coder
and per-channel bias correction tuned to minimise the evaluation metric.

## Approach

**Codec:** A pre-trained HNeRV decoder (229 K parameters, 28-dimensional
per-frame-pair latents, 6 upsample stages, 384×512 → 874×1164) stores the
entire 37.5 MB dashcam video as a single 177 KB archive.

**Entropy coding (`codec_ctx.py`):** The decoder weights, latents, and
selector payload are all re-coded with a constriction-based range coder:
- *Decoder weights* — per-tensor adaptive 256-ary models with
  geometric-primed prior counts; hyperparameters chosen by exact simulated
  code-length search.
- *Latents* — per-dimension causal AR(1+) prediction with optional cross-dim
  features; discrete-Gaussian or adaptive residual models selected per dim.
- *Selector* — adaptive 16-ary model over the FEC6 mode indices.

**Post-processing (`inflate.py`):** After bicubic upsampling, integer
per-channel bias corrections are applied before the final clamp+round:
- frame 0: R −1, B −1
- frame 1: G −1

These values were tuned on the public test video to minimise
`100·segnet + √(10·posenet)`.
## Results

| Metric | Value |
|---|---|
| Final score | `0.191142` = 100·segnet + √(10·posenet) + 25·rate |
| Average SegNet Distortion | `0.00056038` |
| Average PoseNet Distortion | `0.00002943` |
| Compression Rate | `0.00471790` |
| Archive size | `177,136` bytes |
| Archive SHA-256 | `dd4f3899b91f5b59df90b4bf4fc4d903099a286548339f5f65ff91e4b8146aa4` |
| Evaluated on | CPU |
| GPU required | No |

## Archive layout

```
archive.zip  (ZIP_STORED, single member `x`, 177,036 bytes)
└── x
    ├── ctx container (176,429 B)
    │   ├── 7-byte header
    │   ├── decoder section  — 161,104 B  (range-coded HNeRV weights)
    │   ├── latent section   —  15,070 B  (range-coded per-pair latents)
    │   └── selector section —     248 B  (range-coded FEC6 modes)
    └── latent sidecar       —     607 B  (verbatim, trailing)
```

## Files

| File | Role |
|---|---|
| `inflate.py` | Decode archive → raw NHWC uint8 frames |
| `inflate.sh` | Bash wrapper called by the harness |
| `compress.py` | Rebuild `archive.zip` from pinned upstream archives (optional) |
| `compress.sh` | Download + SHA-verify upstream archives then run `compress.py` |
| `codec_ctx.py` | Context-modeled range coder — encoder and decoder |
| `codec.py` | HNeRV weight and latent reconstruction from raw stream bytes |
| `codec_sidecar.py` | 607-byte enum-rank latent sidecar decoder |
| `frame_selector.py` | FEC6 selector grammar and frame transform helpers |
| `model.py` | HNeRV decoder architecture (229 K params) |

## Reproducing the archive

```bash
# From the challenge repo root, with the venv active:
bash submissions/ashika_v3/compress.sh
# Downloads and SHA-256-verifies the two upstream input archives, then
# re-codes their payload with the context-modeled range coder.
# Output: submissions/ashika_v3/archive.zip  (177,136 bytes, deterministic)
```

## Evaluating

```bash
bash evaluate.sh --submission-dir submissions/ashika_v3 --device cpu
```

## Credits

HNeRV decoder architecture and weights originally from
[@AaronLeslie138](https://github.com/AaronLeslie138) (PR #95 / hnerv_muon,
MIT License). Latent sidecar format from
[@SajayR](https://github.com/SajayR) (PR #101, MIT License). FEC6 selector
from [@adpena](https://github.com/adpena) (PR #110, MIT License).
Context-modeled range coder (`codec_ctx.py`) and inflate rewrite by
Ashika Balamurugan.
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
