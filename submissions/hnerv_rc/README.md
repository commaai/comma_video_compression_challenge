# hnerv_rc

Score **0.19258** on the CPU eval axis (leaderboard #1 = 0.19284).

This builds directly on **hnerv_muon** (leaderboard #95, Aaron Leslie, MIT) —
same 229K-param HNeRV decoder, same 8-stage training curriculum (CE → τ-Softplus
→ smooth-disagreement → +QAT → +L7+C1a → λ-sweep → σ-sweep → +Muon). Two changes
here:

1. **A longer/better-converged retrain** of that pipeline (0.1987 → ~0.1931
   before the coder change).
2. **An adaptive range coder** (`src/codec_rc.py`, archive format v2 in
   `src/codec.py`) replacing the brotli entropy stage. Per-tensor adaptive
   order-0 model, uniform prior + fixed count increment, encoder and decoder
   maintaining identical integer tables so no frequency table is transmitted.
   On these weights brotli sits at 163,237 B — above the per-tensor order-0
   entropy (160,387 B) — so a clean adaptive coder beats it by ~1.1 KB. Lossless:
   only bytes change, not the decoded weights, so SegNet/PoseNet distortion are
   identical to the brotli archive. Round-trip and full inflate are bit-exact.

The range-coding direction follows **rhnerv_comma** (#112).

## Inflate

`evaluate.sh --submission-dir ./submissions/hnerv_rc --device cpu` unzips
`archive.zip` and runs `inflate.py`. CPU only (no GPU required for inflation).

## Compress (reproduce)

```bash
bash submissions/hnerv_rc/compress.sh   # ~50h on one GPU from random init
```

Then the codec stage builds `archive.zip` with the v2 range coder.
