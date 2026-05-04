# henosis_final_bias

This submission builds on the semantic-mask neural-renderer family of submissions. It stores an exact 5-class SegNet mask stream with an adaptive range coder, a compact QZS3 renderer payload, pose data, renderer correction streams, and a small final per-pair RGB micro-bias side channel.

The final micro-bias side channel is stored inside `archive.zip` as member `fb` and is counted in the compressed size. It contains 600 packed 4-bit choices selecting one of nine tiny RGB biases for each generated frame pair.

## Verified local CUDA result

```text
archive.zip: 236,676 bytes
PoseNet:     0.00023557
SegNet:      0.00068982
score:       0.28
```

Exact local score is approximately `0.27511`.

## Key files

| File | Description |
| --- | --- |
| `inflate.sh` | entrypoint |
| `inflate.py` | inflater, payload parser, renderer reconstruction, postprocess, and raw writer |
| `range_mask_codec.cpp` | adaptive range decoder for the semantic mask tensor |

## Notes

CUDA/GPU inflation is required. The archive contains all submission data used by the inflater; no compression script is included.
