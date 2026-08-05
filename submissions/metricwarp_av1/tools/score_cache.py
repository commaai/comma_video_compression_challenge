#!/usr/bin/env python
"""Score an A-sampled float16 cache (true full-res chain metric inputs)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from fastscore import score_frames

cache = np.load(sys.argv[1], mmap_mode='r')
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 8
size = int(sys.argv[3]) if len(sys.argv) > 3 else 0
fr = np.asarray(cache, dtype=np.float32)
r = score_frames(fr, stride=stride, archive_size=size)
print(f"pairs {r['n_pairs']} pose {r['pose_dist']:.8f} (t {r['pose_term']:.4f}) "
      f"seg {r['seg_dist']:.8f} (t {r['seg_term']:.4f}) rate_t {r['rate_term']:.4f} SCORE {r['score']:.5f}")
