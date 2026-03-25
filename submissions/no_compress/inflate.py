#!/usr/bin/env python
import av
from frame_utils import yuv420_to_rgb


def decode_to_file(video_path: str, dst: str):
  container = av.open(video_path)
  stream = container.streams.video[0]
  n = 0
  with open(dst, 'wb') as f:
    for frame in container.decode(stream):
      t = yuv420_to_rgb(frame)  # (H, W, 3)
      f.write(t.contiguous().numpy().tobytes())
      n += 1
  container.close()
  return n


if __name__ == "__main__":
  import sys
  src, dst = sys.argv[1], sys.argv[2]
  n = decode_to_file(src, dst)
  print(f"saved {n} frames")
