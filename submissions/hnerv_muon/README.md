# hnerv_muon

A 178 KB archive containing a 229K-parameter HNeRV decoder + 28-d-per-frame-pair latents that, on inflation, produces a video whose frames activate the official frozen SegNet and PoseNet evaluators almost identically to the original. The pipeline is an 8-stage curriculum (CE → τ-Softplus → smooth-disagreement → +QAT → +L7+C1a → λ-sweep → σ-sweep → +Muon) ending in INT8 quantization-aware training with the C1a regularizer shaping the weight distribution for compression and the [Muon optimizer](https://github.com/KellerJordan/Muon) running on hidden conv tensors.

Full writeup: https://aaronleslie.dev/blog/comma-compression

## Inflate

`evaluate.sh --submission-dir ./submissions/hnerv_muon` will unzip `archive.zip` and call `inflate.sh`, which iterates the video list and runs `inflate.py` per video.

## Compress (reproduce)

```bash
bash submissions/hnerv_muon/compress.sh
```

~50 hours on a single GPU from random init.
