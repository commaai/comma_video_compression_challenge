<div align="center">
<h1>comma video compression challenge</h1>

<h3>
  <a href="https://comma.ai/leaderboard">Leaderboard</a>
  <span> · </span>
  <a href="https://comma.ai/jobs">comma.ai/jobs</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Discord</a>
  <span> · </span>
  <a href="https://x.com/comma_ai">X</a>
</h3>

</div>

 `./videos/0.mkv` is a 1 minute 35.8 MB dashcam video. Make it as small as possible while preserving semantic content and temporal dynamics.

- semantic content distortion is measured using:
  - a SegNet: average class disagreements between the predictions of a SegNet evaluated on original vs. reconstructed frames
- temporal dynamics distortion is measured using:
  - a PoseNet: MSE of the outputs of a PoseNet evaluated on original vs. reconstructed 2 consecutive frames
- the compression rate is:
  - the size of the compressed archive divided by the size of the original archive
- the final score is computed as (lower is better):
  - score = 100 * segnet_distortion + 25 * rate + √ (10 * posenet_distortion)

<p align="center">
<img height="800" alt="image" src="https://github.com/user-attachments/assets/eac1bf44-3b35-40fd-ab82-4dde4a2f5d07" />
</p>

## prize pool - submit by May, 3rd 2026 11:59pm AOE
- 1st place: [comma four OR $750] + special swag
- 2nd place: [$500] + special swag
- 3rd place: [$250] + special swag

## quickstart
Clone the repo
```
git clone https://github.com/commaai/comma_video_compression_challenge.git && cd comma_video_compression_challenge
```

Install dependencies
```
sudo apt-get update && sudo apt-get install -y git-lfs ffmpeg  # Linux
brew install git-lfs ffmpeg                                    # (or) macOS (with Homebrew)
git lfs install && git lfs pull
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --group cpu                                            # cpu|cu126|cu128|cu130|mps
source .venv/bin/activate
```

Test Dataloaders and Models
```
python frame_utils.py
python modules.py
```

Create a submission dir and copy the fast baseline_fast scripts
```
mkdir -p submissions/my_submission
cp submissions/baseline_fast/{compress.sh,inflate.{sh,py}} submissions/my_submission/
```

Compress
```
bash submissions/my_submission/compress.sh
```

Evaluate
```
bash evaluate.sh --submission-dir ./submissions/my_submission --device cpu  # cpu|cuda|mps
```

If everything worked as expected, this should producce a `report.txt` file with this content:
```
=== Evaluation config ===
  batch_size: 16
  device: cpu
  num_threads: 2
  prefetch_queue_depth: 4
  report: submissions/baseline_fast/report.txt
  seed: 1234
  submission_dir: submissions/baseline_fast
  uncompressed_dir: /home/batman/comma_video_compression_challenge/videos
  video_names_file: /home/batman/comma_video_compression_challenge/public_test_video_names.txt
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.38042614
  Average SegNet Distortion: 0.00946623
  Submission file size: 2,244,900 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.05979147
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 4.39
```

## submission format and rules

A submission is a Pull Request to this repo that includes:

- **a download link to `archive.zip`** — your compressed data.
- **`inflate.sh`** — a bash script that converts the extracted `archive/` into raw video frames.
- **optional**: a compression script that produces `archive.zip` from the original videos, and any other assets you want to include (code, models, etc.)

See [submissions/baseline_fast/](submissions/baseline_fast/) for a working example, and  `./evaluate.sh` for how the evaluation process works.

Open a Pull Request with your submission and follow the template instructions to be evaluated. If your submission includes a working compression script, and is competitive we'll merge it into the repo. Otherwise, only the leaderboard will be updated with your score and a link to your PR.

### evaluation

```bash
bash evaluate.sh --submission-dir ./submissions/baseline_fast --device cpu|cuda|mps
```

The official evaluation has a time limit of 30 minutes. If your inflation script requires a GPU, it will run on a T4 GPU instance (RAM: 26GB, VRAM: 16GB), if it doesn't it will run on a CPU instance (CPU: 4, RAM: 16GB).

### rules

- External libraries and tools can be used and won't count towards compressed size, unless they use large artifacts (neural networks, meshes, point clouds, etc.), in which case those artifacts should be included in the archive and will count towards the compressed size. This applies to the PoseNet and SegNet as well.
- You can use anything for compression, including the models, original uncompressed video, and any other assets you want to include.
- You may include your compression script in the submission, but it's not required.

## leaderboard (lower is better)

| Name     | Score | PR |
| -------- |:-------:| -------- |
| baseline_fast | 4.4     | |
| no_compress | 25.0     | |

## going further

Check out this large grid search over various ffmpeg parameters. Each point in the figure corresponds to a ffmpeg setting. The fastest encoder setting was submitted as the baseline_fast. You can inspect the grid search [here](https://github.com/user-attachments/files/26169452/grid_search_results.csv) and look for patterns.

<p align="center">
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/ee097dbd-9912-4e7f-a24c-834c178d9668"/>
</p>

You can also use [test_videos.zip](https://huggingface.co/datasets/commaai/comma2k19/resolve/main/compression_challenge/test_videos.zip), which is a 2.4 GB archive of 64 driving videos from the comma2k19 dataset, to test your compression strategy on more samples.

The evaluation script and the dataloader are designed to be scalable and can handle different batch sizes, sequence lengths, and video resolutions. You can modify them to fit your needs.
