#!/usr/bin/env python
"""Decode videos/0.mkv to work/gt.raw exactly like the official CPU eval path (PyAV + yuv420_to_rgb)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import av
from frame_utils import yuv420_to_rgb

def main():
    src = ROOT / 'videos' / '0.mkv'
    dst = Path(__file__).resolve().parent / 'gt.raw'
    container = av.open(str(src))
    stream = container.streams.video[0]
    n = 0
    with open(dst, 'wb') as f:
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame)  # (H, W, 3) uint8
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    print(f"decoded {n} frames -> {dst} ({dst.stat().st_size:,} bytes)")

if __name__ == '__main__':
    main()
