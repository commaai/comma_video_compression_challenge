<!-- SPDX-License-Identifier: MIT -->

# frontier_study

A **reproducible negative-results study** of why the leaderboard's top cluster
(0.192–0.199) is a saturated Pareto frontier, plus a functional reproduction baseline.

**This is a write-up submission, not a leaderboard improvement.** It ships a working,
fully-attributed reproduction of the open HNeRV decoder (score ≈ 0.197, *worse* than the
current #1's 0.192); its value is the analysis in [`WRITEUP.md`](WRITEUP.md), which maps
out four dead ends so others don't re-walk them.

## What's here

| path | role |
|---|---|
| [`WRITEUP.md`](WRITEUP.md) | the study: score-budget decomposition + four mechanistically-explained dead ends + the one path with real upside |
| `inflate.sh`, `inflate.py` | contest-runtime decoder (archive → raw frames) |
| `compress.sh`, `compress.py` | reproduce `archive.zip` from the attributed open weights (no training) |
| `src/model.py` | HNeRV decoder (byte-identical to PR #95) |
| `src/codec.py` | self-contained int8 + brotli codec |
| `study/` | the analysis toolkit (faithful fast scorer + the four experiment scripts + two self-contained Colab notebooks) |

## Headline findings (see `WRITEUP.md` for full data)

- ~**62%** of the top score is archive size; the decoder weights are **93%** of the archive.
- **Sub-int8 quantization** is a catch-22 (per-tensor distortion explodes; per-channel won't compress).
- A **per-pair scorer-exploit oracle** (rate-free) only reaches distortion 0.0739 ≈ the #1's 0.073 — seg is unreachable by cheap global perturbations.
- **Distilling a smaller decoder** hits a capacity wall (validated by a C=36 control that reproduces the teacher exactly).
- **Weight palettization** fails — the weights use ~150 distinct int8 values/tensor; int8+brotli is at the entropy floor.

## Reproduce

```bash
# from repo root
bash submissions/frontier_study/compress.sh                       # builds archive.zip (no training)
bash evaluate.sh --submission-dir ./submissions/frontier_study --device cpu
# study experiments (self-contained Colab notebooks in study/, or the scripts):
python submissions/frontier_study/study/ptq_sweep.py mps
python submissions/frontier_study/study/palettize.py
```

## Attribution

The HNeRV decoder weights are **byte-identical to PR
[#95](https://github.com/commaai/comma_video_compression_challenge/pull/95)**
(@AaronLeslie138), recovered from PR
[#110](https://github.com/commaai/comma_video_compression_challenge/pull/110)'s public
release. No new model was trained. The "C1a" regularizer / 8-stage curriculum referenced
in the study are from PR #95. This submission's contribution is the analysis + toolkit.
