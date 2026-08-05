#!/usr/bin/env python
"""Concatenate segfix chunk npz blobs."""
import sys
import numpy as np
out = sys.argv[1]
blobs, D = [], None
t0 = t1 = 0
for f in sys.argv[2:]:
    d = np.load(f)
    blobs.append(d['blob'].tobytes())
    D = d['dirtable']
    s = d['stats']; t0 += int(s[0]); t1 += int(s[1])
blob = b''.join(blobs)
np.savez(out, blob=np.frombuffer(blob, dtype=np.uint8), dirtable=D, start=0, end=600, stats=np.array([t0, t1]))
print(f"merged blob {len(blob)}B; flips {t0}->{t1} ({(t0-t1)/max(t0,1)*100:.1f}% fixed)")
