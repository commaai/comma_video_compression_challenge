import struct
import numpy as np
import torch
from torch.nn import functional as F
from dz_bitcoder import Decoder, Models, put_int
EH, EW = (384, 512)
FX, CX, V0 = (400.0, 256.0, 174.0)
UNIT = 20.0
KNOBS = ('tz', 'roll', 'far dx', 'far dy', 'road dx', 'road dy', 'road shear x', 'road shear y', 'far zoom')

def pixel_grid(dev):
    v, u = torch.meshgrid(torch.arange(EH, device=dev) + 0.5, torch.arange(EW, device=dev) + 0.5, indexing='ij')
    return (u, v)

def hood_of(tok):
    hood = tok == 4
    first = torch.where(hood.any(1), hood.float().argmax(1), torch.full_like(hood[:, 0], EH, dtype=torch.long)).float()
    return (hood[:, None].float(), first - 1.5)

def road_mask(tok):
    m = (tok <= 1).float()[:, None]
    for _ in range(2):
        m = F.avg_pool2d(F.pad(m, (2, 2, 2, 2), mode='replicate'), 5, 1)
    return m[:, 0]

def layer_flow(mr, x):
    u, v = pixel_grid(mr.device)
    du, dv = ((u - CX)[None], (v - V0)[None])
    ms = 1.0 - mr
    shear = dv.clamp_min(0.0) / 50.0

    def k(i):
        return x[:, i, None, None]
    fu = -k(1) * dv / 100.0 + ms * (k(2) + k(8) * du / 100.0) + mr * (k(4) + k(6) * shear)
    fv = k(1) * du / 100.0 + ms * (k(3) + k(8) * dv / 100.0) + mr * (k(5) + k(7) * shear)
    return UNIT * torch.stack([fu, fv], -1)

def soften(A, passes=1):
    for _ in range(passes):
        A = F.avg_pool2d(F.pad(A, (1, 1, 1, 1), mode='replicate'), 3, 1)
    return A

def target(A, hood, top, tz, flow, passes=1):
    u, v = pixel_grid(A.device)
    rho = ((v - V0) / FX).clamp_min(0.0)
    z = (1.0 + tz[:, None, None] * rho[None]).clamp_min(0.001)
    um = (u[None] - CX) / z + CX
    vm = (v[None] - V0) / z + V0
    if flow is not None:
        um = um + flow[..., 0]
        vm = vm + flow[..., 1]
    vm = torch.minimum(vm, top[:, None, :])
    grid = torch.stack([um / EW * 2 - 1, vm / EH * 2 - 1], -1)
    s = F.grid_sample(soften(A, passes), grid, mode='bilinear', padding_mode='border', align_corners=False)
    return hood * A + (1.0 - hood) * s

def slave(seen, tok, x):
    hood, top = hood_of(tok)
    return target(seen, hood, top, x[:, 0], layer_flow(road_mask(tok), x))
RHO_NUM = (0, 1, 2, 3, 4)
HEAD = struct.Struct('<Hff')

def _knob_pass(coder, models, q, rho, encode):
    N, K = q.shape
    for t in range(N):
        for k in range(K):
            pred = rho[k] * (int(q[t - 1, k]) if t else 0) >> 2
            if encode:
                put_int(coder, models, ('k', k), int(q[t, k]) - pred, True)
            else:
                q[t, k] = pred + put_int(coder, models, ('k', k), 0, False)

def decode_blob(blob):
    N, tz_step, step = HEAD.unpack_from(blob)
    K = len(KNOBS)
    rho = list(blob[HEAD.size:HEAD.size + K])
    q = np.zeros((N, K), dtype=np.int64)
    _knob_pass(Decoder(blob[HEAD.size + K:]), Models(), q, rho, False)
    x = q.astype(np.float32)
    x[:, 0] *= np.float32(tz_step)
    x[:, 1:] *= np.float32(step)
    return x
