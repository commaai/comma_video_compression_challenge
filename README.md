# comma2k25 compression challenge 🤏

<p align="center">
<img height="300" alt="image" src="https://github.com/user-attachments/assets/f72fae51-96bd-47c6-b4ff-6e17e79220cb" />
</p>

 `./test_videos/b0c9d2329ad1606b|2018-07-27--06-03-57/10/video.hevc` is a 1 minute long driving video of size 37.5 MB. Make it as small as possible while preserving semantic content (evaluated by a segmentation model) and temporal dynamics (evaluated by an egomotion relative pose model).

- distortion:
  - SegNet distortion: average class disagreements between the predictions of a SegNet evaluated on original vs. reconstructed frames
  - PoseNet distortion: MSE of the outputs of a PoseNet (x,y,z velocities and roll,pitch,yaw rates) evaluated on original vs. reconstructed seq_len frames
- rate
  - the size of the compressed archive devided by the size of the original archive
- score: a weighted average of the different components of the distortion and the rate

```
score = 100 * segnet_distortion + sqrt(10 * posenet_distortion) + 25 * rate
```

<p align="center">
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/4ba670ba-d823-4e16-a508-18fed000a525"/>
</p>

## Quickstart
```
# clone the repo
git clone https://github.com/commaai/comma2k25_compression_challenge.git
cd comma2k25_compression_challenge

# git lfs
git lfs install
git lfs pull

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# pick one: cu126/ cu128 / cu130 / cpu
uv sync --group cu128

# activate
source .venv/bin/activate

# test dataloaders
python frame_utils.py

# test models
python modules.py

# naively recompress
bash examples/baseline_fast.sh --in-dir test_videos/ --jobs 1 --video-names-file public_test_video_names.txt --out-dir ./submission/

# evaluate the naive recompression strategy
torchrun --nproc-per-node 1 evaluate.py  --dataloader examples/baseline_dataloader_fast.py --compressed-dir ./submission
```

If everything worked as expected, this should producce a `report.txt` file with this content:

```
=== Evaluation config ===
  batch_size: 16
  compressed_dir: submission
  dataloader: examples/baseline_dataloader_fast.py
  device: None
  num_threads: 2
  prefetch_queue_depth: 4
  report: report.txt
  seed: 1234
  uncompressed_dir: test_videos
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.06430182
  Average SegNet Distortion: 0.00381109
  Submission file size (deflated): 14428760.00000000 bytes
  Original uncompressed size (deflated): 37533786.00000000 bytes
  Compression Rate (deflated): 0.38442059
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 10.79350826
```

## submission format

`DatasetClass` should be a subclass of `torch.utils.data.IterableDataset` defined in a python file, with the following signature and methods

```python

class DatasetClass(torch.utils.data.IterableDataset):
  def __init__(
      self,
      file_names: List[str],
      batch_size: int,
      device: torch.device,
      ...
      ):

  def prepare_data(self):
    # called by all ranks
    # e.g. do something with comma2k19_submission.zip
    ...

  def __iter__(self):
    for file_name in self.file_names:
      # yield name, idx, batch

      # name: segment_id
      # idx: batch index within the segment
      # batch: torch.Tensor with shape (batch_size, seq_len, camera_size[1], camera_size[0], 3), dtype=uint8
      ...

```

## going further

[test_videos.zip](https://huggingface.co/datasets/commaai/comma2k19/resolve/main/compression_challenge/test_videos.zip) is a 2.4 GB archive of 64 driving videos from the comma2k19 dataset.
