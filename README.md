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

 `./videos/0.mkv` is a 1 minute 37.5 MB dashcam video. Make it as small as possible while preserving semantic content and temporal dynamics.

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
- 1st place: [comma four OR $1,000] + special swag
- 2nd place: [$500] + special swag
- 3rd place: [$250] + special swag
- Best write-up (visualizations, patterns, etc.): [comma four OR $1,000] + special swag

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

- External libraries and tools can be used and won't count towards compressed size, unless they use large artifacts (neural networks, meshes, point clouds, etc.), in which case those artifacts should be included in the archive and will count towards the compressed size. This applies to the PoseNet and SegNet.
- You can use anything for compression, including the models, original uncompressed video, and any other assets you want to include.
- Submissions are done via public Pull Requests. You may include your compression script in the submission, but it's not required.
- Final ranking will be based on the public leaderboard, no private testing will be performed.

## leaderboard (lower is better)

<!-- TABLE-START -->
<table class="ranked">
 <thead>
  <tr>
   <th>
   </th>
   <th>
    score
   </th>
   <th>
    name
   </th>
   <th>
    link
   </th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>
   </td>
   <td>
    0.33
   </td>
   <td>
    quantizr
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/55" target="_blank">
     #55
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    0.38
   </td>
   <td>
    selfcomp
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/56" target="_blank">
     #56
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    0.60
   </td>
   <td>
    mask2mask
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/53" target="_blank">
     #53
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    1.89
   </td>
   <td>
    neural_inflate
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/49" target="_blank">
     #49
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    1.91
   </td>
   <td>
    svtav1_dilated_ren
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/58" target="_blank">
     #58
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    1.94
   </td>
   <td>
    roi_v2
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/48" target="_blank">
     #48
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    1.95
   </td>
   <td>
    av1_roi_lanczos_unsharp
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/31" target="_blank">
     #31
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    1.98
   </td>
   <td>
    svtav1_av1grain_10bit
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/51" target="_blank">
     #51
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    1.98
   </td>
   <td>
    damir_bearclaw_002
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/30" target="_blank">
     #30
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.01
   </td>
   <td>
    roi_gop300_c34
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/43" target="_blank">
     #43
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.02
   </td>
   <td>
    v4_qp_aq2_roi
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/44" target="_blank">
     #44
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.03
   </td>
   <td>
    av1_crf31_bicubic
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/52" target="_blank">
     #52
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.05
   </td>
   <td>
    svtav1_cheetah
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/24" target="_blank">
     #24
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.07
   </td>
   <td>
    svtav1_45pct_unsharp20_direct
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/27" target="_blank">
     #27
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.08
   </td>
   <td>
    svtav1_gop360_binomial_unsharp
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/26" target="_blank">
     #26
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.08
   </td>
   <td>
    av1_sharp1_adaptive
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/23" target="_blank">
     #23
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.09
   </td>
   <td>
    svtav1_45pct_unsharp
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/20" target="_blank">
     #20
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.16
   </td>
   <td>
    svtav1_spline_fg22
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/37" target="_blank">
     #37
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.20
   </td>
   <td>
    svt_av1_lanczos_fg
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/18" target="_blank">
     #18
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.55
   </td>
   <td>
    h265_g16_512x384_veryslow
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/21" target="_blank">
     #21
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    3.32
   </td>
   <td>
    h265_tuned
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/22" target="_blank">
     #22
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    4.39
   </td>
   <td>
    baseline_fast
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/tree/3e91fd50585789e50a636479ae80f4f877c5e2ac/submissions/baseline_fast" target="_blank">
     #1
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    5.09
   </td>
   <td>
    damir_bearclaw_003
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/pull/39" target="_blank">
     #39
    </a>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    25.0
   </td>
   <td>
    no_compress
   </td>
   <td>
    <a href="https://github.com/commaai/comma_video_compression_challenge/tree/3e91fd50585789e50a636479ae80f4f877c5e2ac/submissions/no_compress" target="_blank">
     #0
    </a>
   </td>
  </tr>
 </tbody>
</table>
<!-- TABLE-END -->

> mirrored from [comma.ai/leaderboard](https://comma.ai/leaderboard)

## going further

Check out this large grid search over various ffmpeg parameters. Each point in the figure corresponds to a ffmpeg setting. The fastest encoder setting was submitted as the baseline_fast. You can inspect the grid search [here](https://github.com/user-attachments/files/26169452/grid_search_results.csv) and look for patterns.

<p align="center">
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/ee097dbd-9912-4e7f-a24c-834c178d9668"/>
</p>

You can also use [test_videos.zip](https://huggingface.co/datasets/commaai/comma2k19/resolve/main/compression_challenge/test_videos.zip), which is a 2.4 GB archive of 64 driving videos from the comma2k19 dataset, to test your compression strategy on more samples.

The evaluation script and the dataloader are designed to be scalable and can handle different batch sizes, sequence lengths, and video resolutions. You can modify them to fit your needs.
