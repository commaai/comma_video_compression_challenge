# Why the comma compression frontier is saturated — a negative-results study

**TL;DR.** The comma video-compression leaderboard's top cluster (0.192–0.199) is a
genuinely *saturated Pareto frontier*. We decompose the score, show that ~60% of it is
the compressed archive size and that the size is dominated by the HNeRV decoder's
weight bits, and then rigorously rule out four natural ways to do better — sub-int8
quantization, a cheap per-pair "scorer-exploit", distilling a smaller decoder, and
weight palettization. Each fails for a concrete, measurable reason. The only remaining
path with real upside is training a fundamentally better/smaller architecture from
scratch against the frozen scorer (real GPU-hours), which is exactly what the leaders
already did. This writeup is the map of the dead ends so the next person doesn't re-walk
them.

> Numbers below are from an in-process *faithful* re-implementation of `evaluate.py`'s
> distortion, run on Apple MPS. MPS vs the leaderboard CPU axis differ by ~0.005 in
> absolute score (bicubic/round are not bit-identical across backends), but **relative**
> comparisons hold. Our base reproduces muon's published 0.199 at **0.197**, and the #1
> archive's weights reproduce at **0.1972** — see §2.

---

## 1. Where the score actually goes

The objective (lower is better) is

```
score = 100 · segnet_distortion  +  √(10 · posenet_distortion)  +  25 · rate
```

where `rate = compressed_size / 37,545,489`. We extracted the **#1 submission's** trained
weights (HNeRV decoder, **228,958 params** + 600×28 per-pair latents) and decomposed its
score:

| term | value | share of 0.192 |
|---|---|---|
| `25 · rate` (178 KB archive) | **0.119** | **~62%** |
| `100 · segnet` | ~0.058 | ~30% |
| `√(10 · posenet)` | ~0.020 | ~10% |

Two facts drive everything:

1. **The score is rate-dominated.** Archive = 178 KB; the **decoder weights are ~162 KB
   (93%)**, latents ~16 KB. So "beat the top" ≈ "shrink the decoder weight bits without
   raising distortion."
2. **Distortion is seg-dominated**, and SegNet only sees the **last frame** of each
   non-overlapping pair at 512×384. PoseNet sees the (f0,f1) pair as a 6-D MSE. seg
   disagreements are *local/spatial*; this matters in §3.2.

Exchange rate: **~15 KB of archive ≈ 0.01 score.**

---

## 2. Methodology

- **Teacher.** Downloaded the #1 release archive (`archive.zip`, SHA-256
  `6bae…3fcf`), parsed out the decoder state-dict + latents. (The HNeRV decoder is
  byte-identical across PRs #95/#101/#110.)
- **Faithful fast scorer.** Re-implemented `evaluate.py`'s distortion in-process:
  decode each pair with the decoder → bicubic-upsample to 874×1164 → feed the frozen
  SegNet+PoseNet → exact `argmax`-disagreement and 6-D pose MSE; cache the original
  video's targets once. ~1 min/candidate vs ~15 min for the full `evaluate.sh`.
  Sanity: re-encoding the teacher reproduces **0.1972** (seg 0.000582, pose 0.000040,
  archive 178,717).

---

## 3. Four things that do not work

### 3.1 Sub-int8 quantization of the decoder — a catch-22

Post-training quantization (per-tensor symmetric), no retrain:

| precision | archive | seg | score |
|---|---|---|---|
| int8 (baseline) | 178,703 | 0.000582 | **0.197** |
| 6-bit | 147,993 | 0.002366 | 0.389 |
| 5-bit | 118,599 | 0.004923 | 0.702 |
| 4-bit | 88,218 | 0.012074 | 1.917 |

Distortion explodes far faster than rate falls. **QAT barely helps**: 5-bit
evaluator-in-the-loop QAT only moved 0.702 → 0.56; output-distillation at 6-bit moved
0.389 → 0.368 (seg essentially flat).

**Per-channel** quantization halves the low-bit *distortion* (6-bit seg 0.00237 →
0.00122) — a lever the per-tensor-int8 leaders didn't use — **but destroys
compressibility**: giving each channel its own scale spreads the integers across the
full range, raising entropy, so brotli yields ~160 KB (≈ int8's 162 KB). Net: **no rate
win**. The two effects cancel.

> **Mechanism.** Per-tensor low-bit compresses (values cluster near 0) but loses
> precision; per-channel keeps precision but stops compressing. int8 sits exactly at the
> useful corner.

### 3.2 A cheap per-pair "scorer-exploit" — already maxed by the leaders

The recent leaderboard moves (#101→#110) came from a per-pair *selector*: cheap frame
perturbations (luma/RGB/chroma bias, 1-px roll) chosen offline against the frozen
scorer, entropy-coded in ~250 bytes. We searched a **65-mode** palette (both frames)
and computed the **oracle** (rate-free, per-pair best):

| | seg | pose | distortion |
|---|---|---|---|
| base | 0.000582 | 0.000040 | 0.0782 |
| **oracle (free)** | 0.000565 | 0.000030 | **0.0739** |

The oracle ≈ the #1's achieved 0.073. **seg barely moved** (0.000582 → 0.000565):
global per-frame perturbations cannot fix *local* segmentation-boundary disagreements,
which is where seg distortion lives. The pose gains were already harvested by #110.

### 3.3 Distilling a smaller decoder — a capacity wall

Warm-start a smaller student (channel-slice from the teacher) and distill it to the
teacher's per-pair RGB outputs, then polish vs the scorer:

| student | params | post-distill L1 | seg | score |
|---|---|---|---|---|
| **C=36 (sanity)** | 228,958 | **0.0000** | 0.000582 | **0.1972** |
| C=28 | 148,038 | **plateaus ≈17** | 0.0167 | 2.78 |

The C=36 control (an exact teacher copy) confirms the pipeline is correct — so C=28's
plateau at L1≈17 (after ~150 steps; *converged*, not slow) is a **genuine capacity wall
for the distillation target**, not a bug.

> **Caveat (important).** This only kills the *distillation shortcut*. Training a smaller
> decoder *from scratch directly against the scorer* (the leaders' method) is **not**
> ruled out — see §5.

### 3.4 Weight palettization — weights are at their precision floor

Hope: muon's "C1a" entropy regularizer collapses weights to a handful of values, so a
small palette + entropy-coded indices beats int8+brotli. **Diagnostic killed it:** the
int8 weights use **69–206 distinct values per tensor** (big layers 118–206) — *not*
collapsed.

| scheme | dec bytes | bits/wt | seg | score |
|---|---|---|---|---|
| int8 + brotli | 162,247 | 5.67 | 0.000582 | 0.197 |
| palette K=64 | 152,844 | 5.34 | 0.001923 | 0.351 |
| palette K=32 | 139,624 | 4.88 | 0.003002 | 0.483 |
| palette K=16 | 112,502 | 3.93 | 0.005327 | 0.766 |

The network needs ~7–8 effective bits; int8+brotli already encodes near that entropy.
Palettizing below it trades a little size for a lot of distortion — same wall as §3.1.

---

## 4. Why the #1 is hard to beat

The top submission already executed the **entire** marginal-gains stack, each piece at
or beyond what the cheap levers allow:

- int8 decoder — at the precision floor (§3.1, §3.4).
- **LZMA latents = 15,387 bytes** — *tighter* than our brotli (15,857).
- a **near-oracle selector** (§3.2).

There is essentially no headroom left for compression tricks or scorer-exploits on top
of the existing decoder. The cheap wins are taken.

---

## 5. The one path with real upside

Everything above attacks a *fixed* 229K decoder. The leaders got their decoder by
**training from scratch against the frozen SegNet/PoseNet** (~50 GPU-h, an 8-stage
curriculum, the Muon optimizer). The untested, genuinely-promising direction is to do
the same but search for a **more parameter-efficient architecture** — smaller/factorized
convs, coordinate-MLP heads, a better decoder-capacity ↔ latent-capacity split — trained
*directly on the scorer* (not distilled), with compression-aware training. That can move
the (bits, distortion) Pareto itself, which no post-hoc trick can. It needs real GPU
budget; it is not a Mac-feasible shortcut.

---

## 6. Reproducibility

All code is small, dependency-light (torch, av, timm, smp, brotli), and CPU/MPS/CUDA:

- `beat_top.py` — teacher extraction, faithful fast scorer, quantization (per-tensor &
  per-channel), distillation, evaluator-loop polish.
- `ptq_sweep.py` / `palettize.py` — §3.1 / §3.4 sweeps and the distinct-value diagnostic.
- `selector_search.py` — §3.2 65-mode per-pair oracle.
- `student_train.py` (+ `student_retrain_colab.ipynb`) — §3.3 smaller-decoder distillation,
  CUDA-ready.

### Appendix — headline numbers (MPS faithful scorer)

```
top archive (#1):           178,517 B   rate term 0.119 (~62% of 0.192)
decoder:                    228,958 params  ≈162 KB int8+brotli (93% of archive)
distinct int8 vals/tensor:  69–206  (net needs ~7–8 bits)
PTQ:        6b 0.389 | 5b 0.702 | 4b 1.917
per-channel 6b:             distortion halved, size unchanged → no win
selector oracle (free):     distortion 0.0782 → 0.0739 (≈ #1's 0.073)
smaller decoder C=28:       distillation walls at L1≈17 (score 2.78)
palette K=64:               saves 10 KB, seg ×3.3 → 0.351
```
