# qzs3_range_mask

This submission uses the qpose/r55 neural renderer, but stores the SegNet mask stream with a compact adaptive range-coded class tensor instead of an AV1/Brotli mask video. The archive member `p` contains all counted reconstruction data: range-coded masks, compact QZS3 model payloads, and the pose stream.

The decoder compiles `range_mask_codec.cpp` at inflate time, decodes the mask tensor from `archive.zip`, then renders frames with the included Python inflater. No video-specific payload is embedded in the decoder source.

Local Modal CUDA verification:

```text
archive.zip: 215,960 bytes
PoseNet:     0.00058400
SegNet:      0.00060989
score:       0.2812077922144353
```

Prepared for submission by `ottokunkel` and exact-evaluated locally with the repository's Modal CUDA harness.
