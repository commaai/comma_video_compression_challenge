# commaH26x_compression_challenge

 `./test_videos/b0c9d2329ad1606b|2018-07-27--06-03-57/10/video.hevc` is a 20 FPS 1 minute long driving video of size 37.5 MB. Make it as small as possible while preserving semantic content and temporal dynamics.

- semantic content distortion is measured using:
  - a SegNet: average class disagreements between the predictions of a SegNet evaluated on original vs. reconstructed frames
- temporal dynamics distortion is measured using:
  - a PoseNet: MSE of the outputs of a PoseNet (x,y,z velocities and roll,pitch,yaw rates) evaluated on original vs. reconstructed 2 consecutive frames
- the compression rate is:
  - the size of the compressed archive divided by the size of the original archive
- the final score is computed as:
  - a weighted average of the different components of the distortion and the rate

```
score = 100 * segnet_distortion + sqrt(10 * posenet_distortion) + 25 * rate
```

<p align="center">
<img height="800" alt="image" src="https://github.com/user-attachments/assets/eac1bf44-3b35-40fd-ab82-4dde4a2f5d07" />
</p>


## quickstart
```
# clone the repo
git clone https://github.com/commaai/commaH26x_compression_challenge.git
cd commaH26x_compression_challenge

# install git-lfs and ffmpeg
sudo apt-get update && sudo apt-get install -y git-lfs ffmpeg                        # Linux
brew install git-lfs ffmpeg                                                          # macOS (with Homebrew)

# git lfs
git lfs install
git lfs pull

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# pick one based on your system(cuda/macOS/cpu): cu126|cu128|cu130|cpu|mps
uv sync --group cu128

# activate
source .venv/bin/activate

# test dataloaders
python frame_utils.py

# test models
python modules.py

# create a submission dir and copy the fast baseline scripts
mkdir -p submissions/my_submission
cp submissions/baseline_fast/compress.sh submissions/my_submission/
cp submissions/baseline_fast/inflate.sh submissions/my_submission/

# naively recompress (creates submissions/my_submission/archive.zip)
bash submissions/my_submission/compress.sh

# evaluate the submission (device: cuda / mps / cpu)
bash evaluate.sh --submission-dir ./submissions/my_submission --device cuda
```

If everything worked as expected, this should producce a `report.txt` file with this content:

```
=== Evaluation config ===
  batch_size: 16
  device: cuda
  num_threads: 2
  prefetch_queue_depth: 4
  report: submissions/baseline_fast/report.txt
  seed: 1234
  submission_dir: submissions/baseline_fast
  uncompressed_dir: /home/batman/commaH26x_compression_challenge/test_videos
  video_names_file: /home/batman/commaH26x_compression_challenge/public_test_video_names.txt
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.38071454
  Average SegNet Distortion: 0.00946292
  Submission file size: 2245157 bytes
  Original uncompressed size: 37533786 bytes
  Compression Rate: 0.05981696
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 4.39
```

## submission format and rules

A submission is a directory containing two assets:

- **a download link to `archive.zip`** — your compressed data. Its size is used to compute the rate term of the score. It will be unzipped into `archive/` by the evaluation script.
- **`inflate.sh`** — a bash script that converts the extracted `archive/` contents into raw video frames.
- **optional**: a compression script that produces `archive.zip` from the original videos, and any other assets you want to include (code, models, etc.)

`inflate.sh` must produce a raw video file at `<output_dir>/<segment_id>/video.raw`. A `.raw` file is a flat binary dump of uint8 RGB frames with shape `(N, H, W, 3)` where N is the number of frames, H and W match the original video dimensions, no header.

See [submissions/baseline/](submissions/baseline/) or [submissions/baseline_fast/](submissions/baseline_fast/) for working examples, and  `./evaluate.sh` for how the evaluation process works.

Open a Pull Request with your submission and follow the template instructions to be evaluated. If your submission includes a working compression script, and is competitive we'll merge it into the repo. Otherwise, only the leaderboard will be updated with your score and a link to your PR.

Note that the evaluation has a time limit of 30 minutes. If your inflation script requires a GPU, it will run on a T4 GPU instance (RAM: 26GB, VRAM: 16GB), if it doesn't it will run on a CPU instance (CPU: 4, RAM: 16GB).

### evaluation

```bash
bash evaluate.sh --submission-dir ./submissions/baseline --device cpu|cuda|mps
```

### rules

- External libraries and tools can be used and won't count towards compressed size, unless they use large artifacts (neural networks, meshes, point clouds, etc.).
- `inflate.sh` should not consume anything outside of the submission directory and the extracted archive.
- You can use anything for compression, including the models and the original uncompressed videos.
- You may include your compression script in the submission, but it's not required.

## leaderboard

| Name     | Score | PR |
| -------- |:-------:| -------- |
| baseline  | 3.0   | |
| baseline_fast | 4.4     | |

## going further

Check out this large grid search over various ffmpeg parameters. Each point in the figure corresponds to a ffmpeg setting, the best scoring setting was submitted as the baseline, and the fastest encoder setting was submitted as the baseline_fast. You can inspect the grid search [here](https://github.com/user-attachments/files/26169452/grid_search_results.csv) and look for patterns.

<p align="center">
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/ee097dbd-9912-4e7f-a24c-834c178d9668"/>
</p>

You can also use [test_videos.zip](https://huggingface.co/datasets/commaai/comma2k19/resolve/main/compression_challenge/test_videos.zip), which is a 2.4 GB archive of 64 driving videos from the comma2k19 dataset, to test your compression strategy on more samples.

The evaluation script and the dataloader are designed to be scalable and can handle different batch sizes, sequence lengths, and video resolutions. You can modify them to fit your needs.
