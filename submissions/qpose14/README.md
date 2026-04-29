# qpose14

`qpose14` is a rate-optimized variant of the public Quantizr-style neural renderer.

The main changes are:

- replace the original float32 `pose.npy.br` side channel with a quantized pose stream;
- pack the compressed mask, model, and pose streams into one zip member to reduce zip header overhead.

CUDA validation result:

```text
Average PoseNet Distortion: 0.00052154
Average SegNet Distortion: 0.00061261
Submission file size: 287,573 bytes
Original uncompressed size: 37,545,489 bytes
Compression Rate: 0.00765932
Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 0.32
```

Inflation requires a GPU for the submitted score.
