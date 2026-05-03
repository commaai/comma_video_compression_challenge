# delta_codec

Experimental residual codec for the comma video compression challenge.

Instead of learning a latent space, this codec stores each 2-frame pair as:

- a low-resolution base frame, taken from the first frame in the pair
- a low-resolution signed residual for the second frame

Inflation reconstructs:

```text
frame0 = base
frame1 = base + residual
```

Both base and residual streams are encoded as regular video streams, then packed
into `archive.zip` with a small `meta.json`.

Quick run:

```bash
bash submissions/delta_codec/compress.sh --width 256 --height 192 --base-crf 34 --delta-crf 24 --delta-step 2
bash evaluate.sh --submission-dir submissions/delta_codec --device cpu
```

Best known local run:

```bash
bash submissions/delta_codec/compress.sh \
  --width 256 \
  --height 192 \
  --preset 8 \
  --base-crf 28 \
  --delta-crf 17 \
  --delta-step 1 \
  --deadzone 1 \
  --outside-delta-step 2 \
  --outside-deadzone 1.5 \
  --roi-feather 8
```

Local score:

```text
Average PoseNet Distortion: 0.41731468
Average SegNet Distortion: 0.01060491
Submission file size: 1,096,117 bytes
Original uncompressed size: 37,545,489 bytes
Compression Rate: 0.02919437
Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 3.83
```

Useful knobs:

- `--width`, `--height`: spatial detail retained before upscaling
- `--delta-step`: residual quantization; lower is more accurate and larger
- `--delta-crf`: residual stream quality; lower preserves motion better
- `--skip-threshold`: set tiny-motion residuals to zero so the residual stream compresses better
- `--outside-delta-step`: coarser residual quantization outside the driving corridor
- `--outside-deadzone`: larger residual deadzone outside the driving corridor
