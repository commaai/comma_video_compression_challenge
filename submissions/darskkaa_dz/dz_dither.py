import itertools
import numpy as np
import torch
from torch.nn import functional as F
CH, CW, EH, EW = (874, 1164, 384, 512)

def _taps(n_in, n_out):
    scale = np.float32(n_in) / np.float32(n_out)
    src = scale * (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) - np.float32(0.5)
    src = np.maximum(src, np.float32(0.0))
    i0 = src.astype(np.int64)
    fr = (src - i0.astype(np.float32)).astype(np.float32)
    return (torch.from_numpy(i0), torch.from_numpy(np.minimum(i0 + 1, n_in - 1)), torch.from_numpy(fr))

class Dither:

    def __init__(self, dev, near=2, wide=16, tol=0.004):
        self.dev, self.tol = (dev, tol)
        y0, y1, fy = _taps(CH, EH)
        x0, x1, fx = _taps(CW, EW)
        assert (y0[1:] > y1[:-1]).all() and (x0[1:] > x1[:-1]).all(), 'blocks overlap'
        w = torch.stack([(1 - fy)[:, None] * (1 - fx)[None], (1 - fy)[:, None] * fx[None], fy[:, None] * (1 - fx)[None], fy[:, None] * fx[None]], -1)
        ws, order = w.sort(dim=-1, descending=True)
        self.ws, self.order = (ws.to(dev), order.to(dev))
        rng = range(-near, near + 1)
        self.near = torch.tensor(list(itertools.product(rng, rng, rng)), dtype=torch.float32, device=dev)
        rng = range(-wide, wide + 1)
        self.wide = torch.tensor([(a, b, 0) for a in rng for b in rng], dtype=torch.float32, device=dev)
        self.near_sums = (self.ws[..., :3] @ self.near.T).contiguous()
        self.rows = torch.stack([y0[:, None].expand(EH, EW), y0[:, None].expand(EH, EW), y1[:, None].expand(EH, EW), y1[:, None].expand(EH, EW)], -1).to(dev)
        self.cols = torch.stack([x0[None, :].expand(EH, EW), x1[None, :].expand(EH, EW), x0[None, :].expand(EH, EW), x1[None, :].expand(EH, EW)], -1).to(dev)

    @staticmethod
    def _pick(t, sums, combos):
        rem = t[..., None] - sums
        lo = combos.clamp_max(0.0).min(1).values
        hi = combos.clamp_min(0.0).max(1).values
        n = torch.minimum(torch.maximum(rem.round(), -lo), 255.0 - hi)
        err, idx = (rem - n).abs().min(-1)
        n = n.gather(-1, idx[..., None])
        return (err, torch.cat([n + combos[idx], n], -1))

    @torch.no_grad()
    def full(self, T):
        T = T.clamp(0.0, 255.0)
        B = T.shape[0]
        out = F.interpolate(T, size=(CH, CW), mode='bicubic', align_corners=False).clamp(0.0, 255.0).round()
        vals = torch.empty(B, 3, EH, EW, 4, device=self.dev)
        for b in range(B):
            for c in range(3):
                err, v = self._pick(T[b, c], self.near_sums, self.near)
                m = err > self.tol
                if m.any():
                    e2, v2 = self._pick(T[b, c][m], self.ws[m][:, :3] @ self.wide.T, self.wide)
                    better = e2 < err[m]
                    vm = v[m]
                    vm[better] = v2[better]
                    v[m] = vm
                vals[b, c] = v
        taps = torch.empty_like(vals)
        taps.scatter_(-1, self.order.expand(B, 3, EH, EW, 4), vals)
        for k in range(4):
            out[:, :, self.rows[..., k], self.cols[..., k]] = taps[..., k]
        return out.to(torch.uint8)
