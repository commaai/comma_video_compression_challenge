# hnerv_ft_microcodec

Built on top of PR #95 and PR #98. Adds a self-contained entropy repack of the
decoder, temporal latents, correction sidecar, and related payload
optimizations.

Contents:

- schema-driven decoder packing with fixed tensor order, per-tensor byte maps,
  fp16 scales, and self-delimiting split Brotli streams;
- compact centered-delta uint8 latent packing under raw LZMA;
- split-packed sidecar with an in-archive ranked Huffman length vector and
  a compact combination-ranked no-op table, selected from
  `dim in 0..27` and `delta in {+-0.01, +-0.02, +-0.03, +-0.04,
  +-0.05, +-0.06, +-0.08, +-0.10}`;
- PR #98's decode-side channel postprocess.

Official local CPU evaluation:

```text
archive.zip:        178,258 bytes
SegNet distortion:  0.00056018
PoseNet distortion: 0.00003286
compression rate:   0.00474779
score:              0.19284
```

The sidecar is included in `archive.zip`; inflation does not read source video
or any repository assets outside the submission code and normal installed
libraries.
