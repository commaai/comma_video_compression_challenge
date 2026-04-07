#!/usr/bin/env python
import av, torch, numpy as np
import torch.nn.functional as F
from PIL import Image
from frame_utils import camera_size, yuv420_to_rgb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 9-tap binomial unsharp kernel (Pascal row 8 / 65536)
_r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL = (torch.outer(_r, _r) / (_r.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)
STRENGTH = 0.40


def decode_and_resize_to_file(video_path: str, dst: str):
  target_w, target_h = camera_size
  container = av.open(video_path)
  stream = container.streams.video[0]
  n = 0
  with open(dst, 'wb') as f:
    for frame in container.decode(stream):
      t = yuv420_to_rgb(frame)
      H, W, _ = t.shape
      if H != target_h or W != target_w:
        pil = Image.fromarray(t.numpy())
        pil = pil.resize((target_w, target_h), Image.LANCZOS)
        x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'), KERNEL, padding=0, groups=3)
        x = x + STRENGTH * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
      f.write(t.contiguous().numpy().tobytes())
      n += 1
  container.close()
  return n


if __name__ == "__main__":
  import sys
  src, dst = sys.argv[1], sys.argv[2]
  n = decode_and_resize_to_file(src, dst)
  print(f"saved {n} frames")
