# metricwarp_av1

CPU-only, classical-codec submission: **full-resolution SVT-AV1** plus two tiny
metric-guided side-channels (**1,686 B** of per-pair pose warps + **5,617 B** of per-tile
seg fixes, brotli) searched against the actual metric networks. No neural weights in the
archive, no lineage from prior payloads, no GPU anywhere in the pipeline.

## Verified score (CPU, PyAV path, full 600 pairs, end-to-end through archive + inflate)

| component | value | term |
|---|---|---|
| SegNet distortion | 0.00531573 | 0.5316 |
| PoseNet distortion | 0.00094302 | 0.0971 |
| rate (464,856 B / 37,545,489 B) | 0.01238114 | 0.3095 |
| **score** | | **0.93821** |

archive sha256 `9cb7c817f69b5d63192589344870eb41bd14b27a1669e7738a00af90c93e9d7d`

## How it works

**Observations about the evaluator** (verified numerically in this repo):

1. Both metric nets consume frames bilinear-resized (no antialias, `align_corners=False`)
   to 512×384. The taps of that resize form **pairwise-disjoint 2×2 blocks** (stride
   1164/512 ≈ 2.27 > 2), so writing a constant into a tap block sets the net's input
   pixel **exactly**; ~23% of full-res pixels have zero weight in the metric.
2. SegNet sees only the **odd** frame of each non-overlapping pair (`x[:, -1]`).
3. PoseNet output is dominated by dim 0 (forward speed); its response to codec noise is a
   per-pair, low-dimensional bias that tiny global transforms can cancel.
4. Pose error within a pair comes from artifact *inconsistency* between the two frames —
   frame decimation/synthesis fails catastrophically, but a per-pair global warp of one
   frame recovers almost everything.

**Layer 0 — encoder.** The original 1164×874 video (decoded with the harness's own
PyAV/BT.601 path) is encoded at **full resolution** with SVT-AV1
(`preset 2, crf 56, 10-bit, tune=2, keyint 1200` — a single keyframe — `scd=0`,
film grain off, `enable-qm=1:qm-min=0`), then stripped to a raw OBU stream (−14.4 KB of
IVF framing). Full-resolution encoding beat every downscaled variant we measured: any
resample chain carries a floor the metric never forgives (0.18–0.32 score for 640–1024 px
chains), while at full resolution the floor is exactly zero.

**Layer 1 — seg-fix side-channel (5,617 B).** For each odd frame, greedy search over
16×16-px tiles: nudge a tile by an integer RGB delta (direction = class-pair mean-color
difference, amplitude ∈ {6,12,18}) if it flips SegNet pixels back to the ground-truth
classes. All arithmetic is uint8-exact, so the decoder reproduces the searched frames
bit-for-bit. Fixes 17% of all flipped pixels.

**Layer 2 — pose side-channel (1,686 B).** For each pair, a correction
`(dx, dy, rot, zoom, luma bias, gain)` for the even frame is searched with the actual
PoseNet (batched coordinate descent; hard pairs get a widened second stage) in **metric
space**: candidate = `A(decoded frame)` (A = the evaluator's exact bilinear downsample),
warped, biased, **rounded to uint8**. Rounding inside the search loop is essential —
optimizing on floats and rounding afterwards costs 20× in pose (the optimum is that
sharp, and biases like −0.5 park pixels on rounding boundaries). Pose MSE drops
1.23 → 0.00094 (term 3.51 → 0.097).

**Layer mixing.** Seg-fix tile edges sometimes make pose harder to correct. Pairs are
independent under this metric, so both worlds are searched and the better one is chosen
**per pair** (562/600 keep the seg-fix; 38 revert). The decoder infers the branch from
side-channel presence.

**Decoder** (`inflate.py`, numpy+torch+av+brotli only, ~35 s wall):
- odd frames with seg-fix entries: `round(A(decode))` → integer tile edits → exact-grid
  placement into the 1164×874 canvas (each metric tap block gets the edited value;
  metric-invisible pixels are filled with a bilinear upsample for visual plausibility);
- odd frames without entries: decoded full-res frame written directly (zero floor);
- even frames: `A(decode)` → per-pair warp → round → exact-grid placement.

## Reproduce

```bash
# from the repo root, venv active, git-lfs assets present
bash submissions/metricwarp_av1/compress.sh   # ~4-5 h on 8 CPU cores (search dominates)
bash evaluate.sh --submission-dir submissions/metricwarp_av1 --device cpu
```

The search tooling lives in this repo's `work/` directory (fast cached-GT scorer,
sweep harness, correction/seg-fix searchers, layer mixer, packager); `compress.sh`
drives it end to end. The searches are deterministic given the same encoder output.

## Files

| file | role |
|---|---|
| `archive.zip` | OBU stream (456,276 B) + `corr.bin` (1,686 B) + `segfix.bin` (5,617 B) + `manifest.json` |
| `inflate.sh` / `inflate.py` | decoder (see above) |
| `compress.sh` | end-to-end reproduction script |
| `expected_output.sha256` | canonical decode hash on this machine (bilinear LSBs are µarch-dependent; metrics reproduce regardless) |
