# OPAL temporal-projector transcode

This submission is a bit-exact entropy transcode of F26. The semantic renderer, carrier, HPAC model, frame selector, fixed residual table, and all 117,964,800 decoded tokens are unchanged. The only modified archive component is the arithmetic-coded token stream.

The promoted archive is **182,040 bytes** with SHA-256 `bd9a47149b52a8f4986758e9274e509836bfa9c89f9b5cb069e90837eeb18400`. Its exact score under the unchanged certified F26 distortions is **0.1591495384**.

## Decoder

`inflate.sh` compiles `runtime/entropy/rc64_backend.c`, which wraps F26's five-class law in the maximal-projector/complement transfer operator. The backend maintains 6,175,440 gradient/curvature sector pairs (49,403,520 bytes) and 55 causal projector families. No adaptive table or fitted correction weight is present in the archive; encoder and decoder reconstruct state from the decoded prefix.

Inflation still requires the challenge's Linux NVIDIA CUDA environment because the inherited F26 renderer is unchanged. The entropy layer itself is CPU C code.


Decoded token SHA-256:

`c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`

## Rate result

| Metric | F26 | OPAL |
|---|---:|---:|
| Archive | 186,724 B | **182,040 B** |
| Token stream | 114,706 B | **110,022 B** |
| Exact score | 0.1622684217 | **0.1591495384** |

The full causal ideal gain is 37,472.661262 bits. Production byte counts come from the actual 63-bit arithmetic stream, not from an ideal-rate estimate. The emitted stream is 4.980829 bits above the continuous ideal.
