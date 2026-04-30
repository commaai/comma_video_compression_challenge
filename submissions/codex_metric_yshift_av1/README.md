# codex_metric_yshift_av1

AV1 submission with a compact metric-selected side channel.

Verified public-test score:

- Final score: `1.24`
- PoseNet distortion: `0.00085142`
- SegNet distortion: `0.00566394`
- Compression rate: `0.02310142`
- Archive size: `867354` bytes

The archive stores six nonuniform AV1 segments plus a three-byte-per-frame
side channel. The side channel applies a small luma bias and integer `dy`/`dx`
shift per decoded frame. The values are selected during compression by testing
candidate reconstructions against the official PoseNet/SegNet metric.

Inflation is deterministic and light: decode the AV1 segments, apply the tuned
resize/color/temporal schedule, then apply the stored side-channel adjustment.
It does not require a GPU for evaluation.

`compress.sh` is included for reproducibility. Full compression is intentionally
slow because it searches side-channel values with the official metric models.
