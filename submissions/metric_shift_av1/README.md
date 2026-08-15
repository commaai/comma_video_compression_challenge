# metric_shift_av1

`metric_shift_av1` is a compact AV1-based submission for the comma video
compression challenge. It compresses the public 37.5 MB dashcam clip to an
866 KB archive, then reconstructs the required raw RGB frames with a small
per-frame brightness side channel.

The goal is not to make a visually perfect video. The goal is to preserve the
signals that the challenge actually scores: SegNet semantic agreement,
PoseNet temporal consistency, and archive size.

## Result

Public `0.mkv` evaluation:

| Metric | Value |
| --- | ---: |
| Final score | `2.04` |
| Archive size | `866,558` bytes |
| Compression rate | `0.02308022` |
| PoseNet distortion | `0.07881037` |
| SegNet distortion | `0.00571624` |

For comparison, `baseline_fast` scores `4.46` on this checkout with an archive
around 2.25 MB. This submission improves mainly by spending far fewer bytes
while keeping the segmentation error in the same general range.

## Approach

The archive stores six AV1 segments and one tiny side-channel file.

The AV1 segments carry the video content at low resolution with film-grain
synthesis enabled. The segment boundaries are intentionally uneven: the first
24 seconds are encoded as one longer chunk, while the middle/end of the clip use
shorter sections with separate CRF, scale, saturation, and film-grain settings.

The side channel stores one signed byte per frame. During compression,
`generate_sidechannel.py` decodes the compressed video, compares each frame to
the source frame, and writes a quantized mean-luma correction. During inflation,
`inflate.py` applies that correction after resizing the decoded frame back to
the camera resolution.

This keeps the method simple:

- AV1 handles most of the rate reduction.
- Film grain gives the metric cheap high-frequency texture.
- The luma side channel fixes frame-level brightness drift for about 1.2 KB.
- Inflation stays deterministic and CPU-friendly.

## Why This Exists

I started from the grid search results and treated the best x265 settings as a
baseline. The strongest plain-codec runs clustered around score `3.0`, usually
near 45 percent scale and CRF 27. Pushing the rate lower helped the archive term
but quickly hurt PoseNet.

AV1 with film grain was a better tradeoff for this metric. It does not look
clean to a human viewer, but it preserves enough structure for SegNet while
keeping the archive much smaller. The luma side channel is the smallest useful
extra correction I found that still keeps the submission easy to inspect.

## Files

| File | Purpose |
| --- | --- |
| `archive.zip` | Compressed submission archive |
| `inflate.sh` | Challenge entrypoint; writes raw RGB frames |
| `inflate.py` | Decodes AV1 segments, resizes, applies postprocessing |
| `compress.sh` | Rebuilds `archive.zip` from the original videos |
| `generate_sidechannel.py` | Builds the per-frame luma correction stream |
| `report.txt` | Official local evaluation report |

## Reproduce

From the repository root:

```bash
bash submissions/metric_shift_av1/compress.sh
PATH=.venv/bin:$PATH bash evaluate.sh --submission-dir submissions/metric_shift_av1 --device cpu
```

The `PATH=.venv/bin:$PATH` prefix is only needed on machines where `python`
does not point at the challenge virtualenv.

## Tuning Notes

Most codec settings are exposed through `SHIFTAV1_*` environment variables in
`compress.sh`. That made it easier to compare runs without changing the inflate
contract.

The script also keeps hooks for slower metric-guided side-channel searches, such
as `SHIFTAV1_SIDECHANNEL_MODE=metric-y-coordinate-fast`. Those modes can improve
the correction quality, but they are not the default because they are slow on a
CPU-only machine.
