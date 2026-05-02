# Neural compressor for the comma video compression challenge

End-to-end trainable scale-hyperprior compressor (Ballé 2018-style) optimized
directly against the challenge score:

    score = 100 * segnet_distortion + 25 * rate + sqrt(10 * posenet_distortion)

## Files

| file            | what it does                                                                    |
|-----------------|---------------------------------------------------------------------------------|
| `compressor.py` | Encoder/Decoder/HyperEncoder/HyperDecoder + compressai entropy modules          |
| `loss.py`       | Differentiable surrogate for the score (soft SegNet disagreement, PoseNet MSE, bpp) |
| `dataset.py`    | Streams `(x_t, x_{t+1})` pairs from comma2k19 driving videos                    |
| `train.py`      | Training loop with two-optimizer setup (main + aux for entropy bottleneck)      |
| `compress.py`   | Inference: videos → bitstreams in `archive/`                                    |
| `decompress.py` | Inference: `archive/` → reconstructed videos as lossless HEVC                   |
| `compress.sh`   | Wrapper: produces `archive.zip` from videos                                     |
| `inflate.sh`    | Wrapper called by the comma evaluator                                           |

## Setup (Vast.ai or local)

```bash
# clone the challenge repo first (you need its modules.py + models/)
git clone https://github.com/commaai/comma_video_compression_challenge.git
cd comma_video_compression_challenge
git lfs install && git lfs pull          # pulls the SegNet/PoseNet weights

# put this directory inside submissions/
mkdir -p submissions
cp -r /path/to/this/folder submissions/my_submission

# install (uv works as in the README; here's a minimal pip install)
pip install torch torchvision av einops timm safetensors \
            segmentation-models-pytorch compressai numpy

# get the 2.4 GB training videos
mkdir test_videos && cd test_videos
wget https://huggingface.co/datasets/commaai/comma2k19/resolve/main/compression_challenge/test_videos.zip
unzip test_videos.zip && cd ..
```

## Train

```bash
cd submissions/my_submission
python train.py \
  --repo_root ../.. \
  --video_dir ../../test_videos \
  --segnet_ckpt ../../models/segnet.safetensors \
  --posenet_ckpt ../../models/posenet.safetensors \
  --out_dir ./checkpoints \
  --N 64 --M 128 --N_hyp 64 \
  --crop 256 256 --batch_size 8 \
  --epochs 10 --steps_per_epoch 2000 \
  --lam_rate 0.01
```

What to watch in the logs:
* `seg` should drop from ~0.5 → low single digits over the first few hundred steps.
* `pose` will be noisier; trend down.
* `bpp` is bits/pixel of the **noised latent** — proxy for rate.
* `~score` is an approximate leaderboard score (use it for early stopping).

**Hyperparam sweep**: train 3 models with `--lam_rate 0.001 / 0.01 / 0.1`. Pick
whichever gives the lowest evaluator score (not lowest training loss).

## Compress and submit

```bash
# 1. produce archive.zip
CHECKPOINT=./checkpoints/compressor_final.pt bash compress.sh

# 2. sanity-check the round trip locally
mkdir reconstructed
python decompress.py --archive_dir _archive_build/archive --out_dir reconstructed --device cuda

# 3. evaluate
cd ../..
bash evaluate.sh --submission-dir ./submissions/my_submission --device cuda
```

## Critical: the rate budget

The test video is 37.5 MB. A baseline rate of 5.98% means **the entire
`archive.zip` (model weights + all bitstreams) must fit under ~2.24 MB** to
match the baseline, less to beat it.

Weights of the default `N=64, M=128` model in fp16: ~3 MB. That's already
over budget *before any video data*. Strategies:

1. **Drop to `N=32, M=64`** → ~700 KB fp16, leaves ~1.5 MB for bitstreams.
2. **int8 quantize** the trained weights (post-training).
3. **Train one model that compresses ALL test videos** so the weight cost is
   amortized — already what we do.
4. **Distill into a smaller decoder** after training the larger one.

Start with `N=32` to confirm the budget closes; scale up if you have headroom.

## Architecture sketch

```
    x  (B, 3, H, W) [0..255]
    │
    ▼  g_a (4 strided convs, GDN)
    y  (B, M, H/16, W/16)
    │
    ▼  h_a (3 strided convs)
    z  (B, N_hyp, H/64, W/64)
    │
    ▼  EntropyBottleneck (factorized prior on z)   ──→ z bits
    ẑ
    │
    ▼  h_s (predicts σ for y)
    σ
    │
  (y, σ) → GaussianConditional                      ──→ y bits
    │
    ▼  ŷ
    │
    ▼  g_s (4 transposed convs, inverse GDN)
    x̂  (B, 3, H, W)  →  feed into frozen SegNet, PoseNet for loss
```

## Notes on the evaluator contract

`inflate.sh` writes lossless HEVC `.hevc` files into the output dir using the
original video basenames. If the evaluator expects a different format (e.g.
`.raw` uint8 matching `TensorVideoDataset`), pass `--ext raw` in `inflate.sh`
or peek at `evaluate.py` and adjust.

## Going further (after a baseline lands)

* **Conditional coding**: encode frame `t+1` given reconstructed frame `t`.
  Cuts pose distortion sharply on driving footage.
* **Implicit Neural Representation (selfcomp at 0.38)**: overfit a tiny MLP
  per video; ship the MLP weights as the bitstream. Trade train time for rate.
* **Mixed-precision inference** with `torch.cuda.amp` — same quality, half memory.