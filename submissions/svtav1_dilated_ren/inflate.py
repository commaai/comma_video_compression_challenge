#!/usr/bin/env python
import av, bz2, io, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from frame_utils import camera_size, yuv420_to_rgb


class REN(nn.Module):
  """20K-param residual enhancement network with dilated convs (dilation 1/2/4)
  for a wider receptive field at the same parameter count as a plain 4-conv REN."""
  def __init__(self, c=32):
    super().__init__()
    self.c0 = nn.Conv2d(3, c, 3, padding=1)
    self.c1 = nn.Conv2d(c, c, 3, padding=2, dilation=2)
    self.c2 = nn.Conv2d(c, c, 3, padding=4, dilation=4)
    self.c3 = nn.Conv2d(c, 3, 3, padding=1)

  def forward(self, x):
    h = F.relu(self.c0(x))
    h = F.relu(self.c1(h))
    h = F.relu(self.c2(h))
    return x + self.c3(h)


def load_ren(ren_path: Path, device: torch.device) -> REN:
  sd = torch.load(io.BytesIO(bz2.decompress(ren_path.read_bytes())), map_location=device)
  for k in list(sd.keys()):
    if k.endswith('.scale'): continue
    scale_k = k + '.scale'
    if scale_k in sd:
      sd[k] = sd[k].float() * sd[scale_k]
      del sd[scale_k]
  model = REN().to(device).eval()
  model.load_state_dict(sd)
  return model


def decode_and_inflate_to_file(video_path: str, dst: str):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # ren.bz2 lives alongside the mkv inside the extracted archive directory
  ren = load_ren(Path(video_path).parent / 'ren.bz2', device)

  target_w, target_h = camera_size
  model_w, model_h = 512, 384  # SegNet/PoseNet internal input resolution

  container = av.open(video_path)
  stream = container.streams.video[0]
  n = 0
  with open(dst, 'wb') as f, torch.no_grad():
    for frame in container.decode(stream):
      t = yuv420_to_rgb(frame)
      x = t.permute(2, 0, 1).unsqueeze(0).float().to(device)
      if x.shape[-2:] != (model_h, model_w):
        x = F.interpolate(x, size=(model_h, model_w), mode='bicubic', align_corners=False)
      x = (ren(x / 255.0) * 255.0).clamp(0, 255)
      x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
      out = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8).cpu().numpy()
      f.write(out.tobytes())
      n += 1
  container.close()
  return n


if __name__ == '__main__':
  src, dst = sys.argv[1], sys.argv[2]
  n = decode_and_inflate_to_file(src, dst)
  print(f'saved {n} frames')
