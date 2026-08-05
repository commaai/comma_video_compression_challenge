#!/usr/bin/env python
"""Merge chunked correction npz files into one."""
import sys
import numpy as np
out = sys.argv[1]
P = np.zeros((600, 6), dtype=np.float32)
M = np.zeros((600, 2), dtype=np.float32)
for f in sys.argv[2:]:
    d = np.load(f)
    s, e = int(d['start']), int(d['end'])
    P[s:e] = d['params'][s:e]
    M[s:e] = d['mse'][s:e]
np.savez(out, params=P, mse=M, start=0, end=600)
a, b = M[:, 0].mean(), M[:, 1].mean()
print(f"merged: base {a:.6f} (term {(10*a)**0.5:.4f}) -> corrected {b:.6f} (term {(10*b)**0.5:.4f})")
hard = (M[:, 1] > 0.002).sum()
print(f"pairs above 0.002: {hard}")
