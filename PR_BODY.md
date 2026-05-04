# submission name:

henosis_final_bias

# upload zipped `archive.zip`

[archive.zip](https://github.com/henosis-us/comma_video_compression_challenge/releases/download/henosis-final-bias-v1/archive.zip)

# report.txt

```txt
=== Evaluation config ===
  batch_size: 16
  device: cuda
  num_threads: 2
  prefetch_queue_depth: 4
  report: submissions/henosis_final_bias/report.txt
  seed: 1234
  submission_dir: submissions/henosis_final_bias
  uncompressed_dir: /workspace/videos
  video_names_file: /workspace/public_test_video_names.txt
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00023557
  Average SegNet Distortion: 0.00068982
  Submission file size: 236,676 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00630371
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.28
```

# does your submission require gpu for evaluation (inflation)?

yes

# did you include the compression script? and want it to be merged?

no

# additional comments

CUDA/GPU inflation is required. Locally validated with the repo evaluator on CUDA. Exact local score is approximately `0.27511`.

This uses the semantic-mask/QZS3 renderer approach with an additional small final per-pair RGB micro-bias side channel. The side-channel choices are included in `archive.zip` as member `fb` and counted in the compressed size.
