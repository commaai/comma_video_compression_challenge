#!/usr/bin/env python
import math, os, mmap, torch, warnings
import torch.nn.functional as F
import numpy as np
from typing import List
from pathlib import Path

HERE = Path(__file__).resolve().parent

seq_len = 2
camera_size = (1164, 874)
camera_fl = 910.
segnet_model_input_size = (512, 384)

def hevc_buffer_mmap(path: str):
  f = open(path, "rb")
  mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  mv = memoryview(mm)
  return mv, (mm, f)

def _hevc_frame_count(path: str) -> int:
  # assumes one slice per frame x265 default
  with open(path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as b:
    frames, i = 0, 0
    find = b.find
    while True:
      j = find(b'\x00\x00\x01', i)
      if j < 0: return frames
      p = j + 3
      if ((b[p] >> 1) & 0x3F) <= 31:   # any VCL slice
        frames += 1
      i = p

def _container_frame_count(path: str) -> int:
  import av
  container = av.open(path)
  stream = container.streams.video[0]
  n = stream.frames
  if n == 0:  # some containers don't report frame count
    n = sum(1 for _ in container.demux(stream) if _.size > 0)
  container.close()
  return n

def frame_count(path: str) -> int:
  if path.endswith('.hevc'):
    return _hevc_frame_count(path)
  return _container_frame_count(path)


@torch.no_grad()
def rgb_to_yuv6(rgb_chw: torch.Tensor) -> torch.Tensor:
  H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
  H2, W2 = H // 2, W // 2
  rgb = rgb_chw[..., : , :2*H2, :2*W2]

  R = rgb[..., 0, :, :]
  G = rgb[..., 1, :, :]
  B = rgb[..., 2, :, :]

  kYR, kYG, kYB = 0.299, 0.587, 0.114
  Y = (R * kYR + G * kYG + B * kYB).clamp_(0.0, 255.0)
  U = ((B - Y) / 1.772 + 128.0).clamp_(0.0, 255.0)
  V = ((R - Y) / 1.402 + 128.0).clamp_(0.0, 255.0)

  U_sub = (
    U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
    U[..., 0::2, 1::2] + U[..., 1::2, 1::2]
  ) * 0.25
  V_sub = (
    V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
    V[..., 0::2, 1::2] + V[..., 1::2, 1::2]
  ) * 0.25

  y00 = Y[..., 0::2, 0::2]
  y10 = Y[..., 1::2, 0::2]
  y01 = Y[..., 0::2, 1::2]
  y11 = Y[..., 1::2, 1::2]
  return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)

class VideoDataset(torch.utils.data.IterableDataset):
  def __init__(self, file_names: List[str], data_dir: Path, batch_size: int, device: torch.device, format: str = None, num_threads: int = 2, seed: int = 123, prefetch_queue_depth: int = 4):
    super().__init__()
    if format is not None:
      file_names = [str(Path(fn).with_suffix('.' + format.lstrip('.'))) for fn in file_names]
    self.all_file_names = file_names
    self.batch_size = batch_size
    self.device = device
    self.data_dir = data_dir
    self.num_threads = num_threads
    self.seed = seed
    self.prefetch_queue_depth = prefetch_queue_depth
    self.rank, self.world_size = self._get_dist_info()
    self.file_names = self.all_file_names[self.rank::self.world_size]
    self.paths = [str(data_dir / fn) for fn in self.file_names]

  @property
  def device_id(self):
    return self.device.index

  @staticmethod
  def _get_dist_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
      return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))

  def prepare_data(self):
    assert all((self.data_dir / fn).exists() for fn in self.file_names)
    print(f"{type(self).__name__} on rank {self.rank} with {len(self.paths)} files.")

class DaliVideoDataset(VideoDataset):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    assert self.device.type == 'cuda', f"DaliVideoDataset requires cuda, got {self.device}"
    import nvidia.dali.fn as fn
    from nvidia.dali import pipeline_def
    warnings.filterwarnings("ignore", message=r"Please set `reader_name`.*", category=Warning, module=r"nvidia\.dali\.plugin\.base_iterator")

    @pipeline_def
    def _pipe():
      vid = fn.experimental.inputs.video(
        name="inbuf",
        sequence_length=seq_len,
        device="mixed",
        no_copy=True,
        blocking=False,
        last_sequence_policy="pad",
      )
      return vid
    self._pipe_def = _pipe

  def __iter__(self):
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy

    for path in self.paths:
      mv, (mm, f) = hevc_buffer_mmap(path)
      frames_per_file = frame_count(path)
      num_sequences = frames_per_file // seq_len
      it_size = math.ceil(num_sequences / self.batch_size)
      pipe = self._pipe_def(batch_size=self.batch_size, num_threads=self.num_threads, device_id=self.device_id, prefetch_queue_depth=self.prefetch_queue_depth)
      pipe.build()
      pipe.feed_input("inbuf", [mv])
      it = DALIGenericIterator([pipe], output_map=["video"], auto_reset=False, last_batch_policy=LastBatchPolicy.PARTIAL, last_batch_padded=False, prepare_first_batch=False)

      idx = 0
      while idx < it_size:
        data = next(it)
        vid = data[0]["video"]
        idx += 1
        yield path, idx, vid

      torch.cuda.synchronize()
      it.reset()
      del it, pipe
      mv.release()
      mm.close()
      f.close()

def yuv420_to_rgb(frame) -> torch.Tensor:
  """
  yuv420 to rgb matching nvdec output.
  uses bilinear chroma upsampling + BT.601 limited range.
  """
  H, W = frame.height, frame.width
  y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(H, frame.planes[0].line_size)[:, :W]
  u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(H//2, frame.planes[1].line_size)[:, :W//2]
  v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(H//2, frame.planes[2].line_size)[:, :W//2]

  y_t = torch.from_numpy(y.copy()).float()
  u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
  v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)

  u_up = F.interpolate(u_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()
  v_up = F.interpolate(v_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()

  yf = (y_t - 16.0) * (255.0 / 219.0)
  uf = (u_up - 128.0) * (255.0 / 224.0)
  vf = (v_up - 128.0) * (255.0 / 224.0)

  r = (yf + 1.402 * vf).clamp(0, 255)
  g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
  b = (yf + 1.772 * uf).clamp(0, 255)
  return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)

class AVVideoDataset(VideoDataset):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    assert self.device.type != 'cuda', f"AVVideoDataset not for cuda, use DaliVideoDataset instead"

  def __iter__(self):
    import av
    for path in self.paths:
      fmt = 'hevc' if path.endswith('.hevc') else None
      container = av.open(path, format=fmt)
      stream = container.streams.video[0]
      batch_buf = []
      seq_buf = []
      idx = 0

      for frame in container.decode(stream):
        arr = yuv420_to_rgb(frame)  # (H, W, 3) uint8, matches nvdec
        seq_buf.append(arr)
        if len(seq_buf) == seq_len:
          batch_buf.append(torch.stack(seq_buf))  # (seq_len, H, W, 3)
          seq_buf = []
          if len(batch_buf) == self.batch_size:
            idx += 1
            yield path, idx, torch.stack(batch_buf)  # (B, seq_len, H, W, 3)
            batch_buf = []

      # partial batch (matches DALI LastBatchPolicy.PARTIAL)
      if batch_buf:
        idx += 1
        yield path, idx, torch.stack(batch_buf)

      container.close()

class TensorVideoDataset(VideoDataset):
  """Loads raw uint8 tensor files (.raw) via mmap and batches into (B, S, H, W, C)."""
  def __init__(self,*args, **kwargs):
    super().__init__(format='raw', *args, **kwargs)

  def __iter__(self):
    import numpy as np
    W, H = camera_size
    C = 3
    frame_bytes = H * W * C
    for path in self.paths:
      file_size = os.path.getsize(path)
      N = file_size // frame_bytes
      mm = np.memmap(path, dtype=np.uint8, mode='r', shape=(N, H, W, C))
      frames = torch.from_numpy(mm)  # (N, H, W, C), shares mmap memory

      batch_buf = []
      seq_buf = []
      idx = 0

      for i in range(frames.shape[0]):
        seq_buf.append(frames[i])
        if len(seq_buf) == seq_len:
          batch_buf.append(torch.stack(seq_buf))  # (seq_len, H, W, C)
          seq_buf = []
          if len(batch_buf) == self.batch_size:
            idx += 1
            yield path, idx, torch.stack(batch_buf)  # (B, seq_len, H, W, C)
            batch_buf = []

      # partial batch
      if batch_buf:
        idx += 1
        yield path, idx, torch.stack(batch_buf)

      del frames, mm

if __name__ == "__main__":
  batch_size = 13
  if torch.cuda.is_available():
    device = torch.device('cuda')
  elif torch.backends.mps.is_available():
    device = torch.device('mps')
  else:
    device = torch.device('cpu')
  files = (HERE / 'public_test_video_names.txt').read_text().splitlines()
  uncompressed_data_dir = Path('./videos/')
  DsClaas = DaliVideoDataset if device.type == 'cuda' else AVVideoDataset
  ds = DsClaas(files, data_dir=uncompressed_data_dir, batch_size=batch_size, device=device)
  ds.prepare_data()
  for i, (path, idx, batch) in enumerate(ds):
    assert list(batch.shape)[1:] == [seq_len, camera_size[1], camera_size[0], 3], f"unexpected batch shape: {batch.shape}"
    print(i, batch.shape)
