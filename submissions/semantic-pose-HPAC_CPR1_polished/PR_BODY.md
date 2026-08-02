# submission name:

semantic-pose-HPAC_CPR1_polished

# upload zipped `archive.zip`

Archive URL: <https://github.com/codexblack/comma_video_compression_challenge/releases/download/semantic-pose-HPAC_CPR1_polished-f26/archive.zip>

SHA-256: `12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`

Size: `186,724` bytes

The archive contains one stored ZIP member, `p` (`186,624` bytes).

# report.txt

```
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00000688
  Average SegNet Distortion: 0.00029639
  Submission file size: 186,724 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00497327
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.16
```

The underlying score before report rounding was `0.16226842169958583`.

# does your submission require gpu for evaluation (inflation)?

Yes. Inflation targets the challenge's Linux NVIDIA T4 CUDA environment.

# did you include the compression script? and want it to be merged?

Yes. `compress.sh` is included for reproducible archive preparation and is
intended to be merged. It fetches the published F26 artifact by default,
verifies its byte size, SHA-256, and ZIP layout, then writes `archive.zip`.
The exploratory training and search pipeline remains in the separate frozen
experiment repository and is not part of this submission.

# is this submission competitive or innovative? explain why

F26 is a 186,724-byte F24S candidate evaluated over all 600 public pairs. It
combines fixed-boundary int6 residual coding, RC64 token decoding, and a
14-byte rate-charged frame-0 selector with joint FiLM/carrier optimization.
The full evaluation measured PoseNet distortion `0.00000688438922225032`,
SegNet distortion `0.00029639352578669786`, and rate
`0.004973273886511373`.

# additional comments

F26 follows [PR #130](https://github.com/commaai/comma_video_compression_challenge/pull/130).
Its constrained basis, scales, and int12 carrier representation acknowledge
[PR #133](https://github.com/commaai/comma_video_compression_challenge/pull/133).
The F26 changes are exact evaluator-gated carrier refinement, joint renderer
and carrier compensation, the sparse selector, and lossless F24S/RC64
packing.

This PR changes only `submissions/semantic-pose-HPAC_CPR1_polished`. The
`cpr1` directory contains the retained integer model and renderer, while
`runtime` contains only the F26 parser and decoder components. Training,
search, archive-building, unused floating-point HPAC, and unused codec paths
were removed. `inflate.sh` makes no network requests, and
`verify_submission.py` validates the promoted wire format before evaluation.
The refactored decoder reconstructs the exact 3,662,409,600-byte evaluator
input with SHA-256
`cb5b9dd36f2dee33419919f1a705aa4bb41baf986f3a972b83de70a1981dd63c`.
