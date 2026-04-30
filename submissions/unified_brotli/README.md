# unified_brotli

`unified_brotli` is a rate-optimized variant of the public quantizr-style neural renderer (credit: [quantizr PR #55](https://github.com/commaai/comma_video_compression_challenge/pull/55), [qpose14 PR #63](https://github.com/commaai/comma_video_compression_challenge/pull/63)).

The main changes vs qpose14:

- **Single-stream brotli** of the concatenated raw mask + model + pose payloads (vs three separately-brotli'd streams), so the entropy coder can find cross-stream redundancy.
- **Delta-encoded velocity**: first uint16 + 599 int16 deltas, exploiting brotli's sensitivity to smoothly-varying low-entropy sequences.
- Drops rotation entirely; PoseNet conditioning is dominated by velocity.

CUDA validation result:

```text
Average PoseNet Distortion: 0.00052868
Average SegNet Distortion: 0.00072205
Submission file size: 287,165 bytes
Original uncompressed size: 37,545,489 bytes
Compression Rate: 0.00764846
Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 0.34
```

Inflation requires a GPU.
