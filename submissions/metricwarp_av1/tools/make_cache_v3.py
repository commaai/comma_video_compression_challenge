#!/usr/bin/env python
"""Build pose-search cache v3: odds = rounded+segfixed (uint8 domain), evens = original floats."""
import sys
import numpy as np
src, segfix_npz, dst = sys.argv[1], sys.argv[2], sys.argv[3]
cache = np.load(src, mmap_mode='r')
d = np.load(segfix_npz)
blob = d['blob'].tobytes()
D = d['dirtable']
AMPS = [6, 12, 18]
acts = {}
p = 0
while p < len(blob):
    k = blob[p] | (blob[p+1] << 8); cnt = blob[p+2]; p += 3
    a = []
    for _ in range(cnt):
        t = blob[p] | (blob[p+1] << 8); di, ai = blob[p+2], blob[p+3]; p += 4
        a.append((t, np.round(D[di] * AMPS[ai]).astype(np.int16)))
    acts[k] = a
out = np.empty_like(np.asarray(cache))
out[0::2] = cache[0::2]  # evens unchanged (float)
for k in range(600):
    odd = np.clip(np.round(np.asarray(cache[2*k+1], dtype=np.float32)), 0, 255).astype(np.int16)
    for (t, delta) in acts.get(k, []):
        ty, tx = divmod(int(t), 32)
        odd[ty*16:(ty+1)*16, tx*16:(tx+1)*16] = np.clip(odd[ty*16:(ty+1)*16, tx*16:(tx+1)*16] + delta, 0, 255)
    out[2*k+1] = odd.astype(np.float16)
np.save(dst, out)
print(f"wrote {dst} ({len(acts)} frames edited)")
