# hnerv_pose_warmup_selector

**Score: 1.68** (segnet_dist 0.01013, posenet_dist 0.02539, rate 0.00640 / 240,272 bytes).
CPU inflation ~1 min; no GPU required for evaluation.

## Approach

A per-video neural representation (HNeRV-style): a small conv decoder is overfit to the dashcam
video, and the **quantized decoder weights + per-frame latents are the compressed archive**.
Frames are reconstructed by running the decoder at inflation time — the source video is not used.

Key ingredients:

1. **Metric-aware training.** Instead of pixel MSE, the decoder is trained directly against the
   two scorer networks: PoseNet distortion (MSE on the scored pose outputs) and a differentiable
   SegNet surrogate (cross-entropy of render logits vs ground-truth classes, boundary+focal
   weighted). A brightness-weighted pixel term anchors it. This optimizes what is actually scored
   rather than perceptual fidelity the scorers ignore.

2. **Pose-weight warmup.** PoseNet distortion starts ~130 and shrinks to ~0.04 during training; a
   fixed small pose weight starves it at convergence. Ramping the pose weight up over training
   (0.005 → 0.1) was the single biggest lever — it took posenet_dist from ~1.3 to ~0.04.

3. **Compact architecture (`sc4_5x`).** 449k params, internal resolution 648×810 (then bilinear to
   native 1164×874). A small per-frame spatial embedding (4×4×5) keeps the latent cost low; most
   capacity is in the shared decoder. Quantization: decoder weights at 6-bit, embeddings at 4-bit,
   then Brotli (a trained decoder needs ≥6-bit; 4-bit collapses pose).

4. **Per-pair perturbation selector.** A post-hoc, decoder-agnostic distortion reducer: for each
   consecutive frame pair, one of 28 tiny candidate transforms (small luma/chroma biases, single
   pixel rolls, a faint chroma checkerboard) is chosen offline to minimize that pair's SegNet +
   PoseNet distortion. Only a 1-byte index per pair is stored (`selector.bin`, ~600 bytes total).
   This cut posenet_dist ~40% for negligible rate.

## Files

- `archive.zip` — `meta.json` (config + per-tensor quant scales), `weights.br` (Brotli-compressed
  quantized decoder + embeddings), `selector.bin` (per-pair transform indices).
- `inflate.sh` / `inflate.py` — decode the archive and render `.raw` frames. Self-contained.
- `src/` — vendored decoder (`model.py`), codec (`codec.py`), and selector (`selector.py`).

## Reproduction

Training was done on a GPU (Colab). The pipeline: train the HNeRV with metric-aware loss + pose
warmup, quantize (`--bits 6 --emb-bits 4`) + Brotli into `archive.zip`, then run the offline
selector to add `selector.bin`. Inflation (this submission) is CPU-only and deterministic.
