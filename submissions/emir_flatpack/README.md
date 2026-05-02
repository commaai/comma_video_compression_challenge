# emir_flatpack

Lossless rate-optimized derivative of the public Quantizr-style neural
renderer. **Same renderer, same weights, same masks, same pose vectors as
[qpose14 (PR #63)](https://github.com/commaai/comma_video_compression_challenge/pull/63)
— just packed more efficiently.**

## What's new vs qpose14

1. **Flat-binary model weights.** qpose14 ships its 91,582-byte
   `torch.save()` blob (with ~25 KB of pickle/header overhead — long key
   names like `shared_trunk.fuse_block.conv2.norm2.weight`, type tags,
   nested-dict structure). `emir_flatpack` strips it all: the decoder
   rebuilds the state-dict from a fixed
   `(name, kind, shape, has_bias)` schema baked into `inflate.py`, and
   the archive carries only the raw weight bytes (66,650 bytes raw).
2. **Single brotli stream over all three payloads** with a 1-byte order
   tag in front + 12-byte length header. The encoder tries all 6
   orderings and stores the winner; brotli's window can find
   cross-stream redundancy this way. (`unified_brotli` PR #64 also did
   single-brotli, but it dropped pose rotation — that costs ~0.011 in
   SegNet. We keep the full 6-D pose.)

What we did **not** change vs qpose14: the model floats, the AV1 .obu
mask bytes (verbatim copy), the 6-D quantized pose stream.

## Result

- Archive: **281,948 bytes** (qpose14 was 287,573 → **−5,625 bytes**,
  −1.96%).
- Distortion: **identical to qpose14** (same renderer, same weights, same
  inputs).
- Rate: 0.00750950 (qpose14 was 0.00765932).
- **Expected score on the T4 evaluator: ~0.32** (qpose14 numbers minus
  the rate delta: 0.323 → 0.319).

Local CPU/MPS validation (the leaderboard runs on T4, where the
distortion floor is ~0.13 lower than on CPU due to float-precision):

```text
Average PoseNet Distortion: 0.00066266   (== qpose14 on this machine)
Average SegNet Distortion : 0.00072222   (== qpose14 on this machine)
Submission file size      : 281,948 bytes
Original uncompressed size: 37,545,489 bytes
Compression Rate          : 0.00750950
Local score               : 0.34
```

## Files

| file          | role                                                      |
|---------------|-----------------------------------------------------------|
| `inflate.py`  | flat-binary decoder + qpose14 architecture (verbatim)     |
| `inflate.sh`  | bash wrapper, locates `.venv/bin/python`                  |
| `archive.zip` | single-member `p`: brotli(order_tag + lengths + payload)  |
| `repack.py`   | one-shot encoder used to build `archive.zip`              |
| `_schema.py`  | dump of the model schema for reproducibility (not used at decode) |

## Inflation requires a GPU

For the T4-validated qpose14 distortion floor. CPU/MPS still produces
correct frames but distortion drifts by ~0.0001 due to float precision.

## Credit

Built on top of three public submissions:

- [qpose14 (PR #63)](https://github.com/commaai/comma_video_compression_challenge/pull/63)
  — the renderer architecture and trained weights.
- [quantizr (PR #55)](https://github.com/commaai/comma_video_compression_challenge/pull/55)
  — the FP4 quantization scheme.
- [unified_brotli (PR #64)](https://github.com/commaai/comma_video_compression_challenge/pull/64)
  — the single-brotli concatenation idea (keeping the full 6-D pose this
  time).
