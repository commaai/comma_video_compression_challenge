#!/usr/bin/env python
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import av
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from frame_utils import camera_size, yuv420_to_rgb
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return x + self.net(x)


class TinyAR(nn.Module):
    def __init__(self, ch=32, n_res=4):
        super().__init__()
        self.head = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(n_res)])
        self.tail = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, x):
        # x: float tensor in [0, 255]
        h = torch.relu(self.head(x))
        h = self.body(h)
        return (x + self.tail(h)).clamp(0, 255)


def load_quantized_state_dict(path: Path) -> dict[str, torch.Tensor]:
  raw = load_file(path, device="cpu")
  out = {}
  for name, tensor in raw.items():
    if name.endswith("__scale") or name.endswith("__bits"):
      continue
    scale_name = f"{name}__scale"
    bits_name = f"{name}__bits"
    if scale_name in raw and bits_name in raw:
      bits = int(raw[bits_name].item())
      if bits != 8:
        raise ValueError(f"unsupported quant bits for {name}: {bits}")
      out[name] = tensor.float() * raw[scale_name].float()[0]
    else:
      out[name] = tensor.float()
  return out


def decode_apply_quant_ar_to_file(video_path: str, dst: str, archive_dir: str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  target_w, target_h = camera_size
  archive_path = Path(archive_dir)
  cfg = json.loads((archive_path / "config.json").read_text())
  model = TinyAR(ch=int(cfg["channels"]), n_res=int(cfg["n_res"])).to(device)
  model.load_state_dict(load_quantized_state_dict(archive_path / "ar_model_q.safetensors"))
  model.eval()

  fmt = "hevc" if video_path.endswith(".hevc") else None
  container = av.open(video_path, format=fmt)
  stream = container.streams.video[0]
  n = 0
  with open(dst, "wb") as f:
    for frame in container.decode(stream):
      t = yuv420_to_rgb(frame)
      h, w, _ = t.shape
      if h != target_h or w != target_w:
        x = t.permute(2, 0, 1).unsqueeze(0).float()
        x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0)
      else:
        t = t.float()
      with torch.no_grad():
        inp = t.permute(2, 0, 1).unsqueeze(0).to(device)
        out = model(inp)
        t = out.squeeze(0).permute(1, 2, 0).clamp(0, 255).round()
      f.write(t.to(torch.uint8).contiguous().cpu().numpy().tobytes())
      n += 1
  container.close()
  return n


if __name__ == "__main__":
  video_path = sys.argv[1]
  dst = sys.argv[2]
  archive_dir = sys.argv[3] if len(sys.argv) > 3 else str(Path(video_path).parent)
  n = decode_apply_quant_ar_to_file(video_path, dst, archive_dir)
  print(f"saved {n} frames")
