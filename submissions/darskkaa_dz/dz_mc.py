import torch
ROAD, LANE = (0, 1)

def composite(prev, tau_q, v0_q, u0_q, dx=0, dy=0):
    H, W = prev.shape
    prev = prev.long()
    v = torch.arange(H, device=prev.device, dtype=torch.int64)[:, None]
    u = torch.arange(W, device=prev.device, dtype=torch.int64)[None, :]
    a = 2 * v + 1 - int(v0_q)
    d = 16000 + int(tau_q) * a.clamp_min(0)
    vp = torch.div(int(v0_q) * d + 16000 * a, 2 * d, rounding_mode='floor') - int(dy)
    up = torch.div(2 * int(u0_q) * d + (2 * u + 1 - 2 * int(u0_q)) * 16000, 2 * d, rounding_mode='floor') - int(dx)
    src = prev[vp.clamp(0, H - 1).expand(H, W), up.clamp(0, W - 1)]
    ground = prev <= LANE
    lane = (src == LANE) & (a > 0)
    out = prev.clone()
    out[ground & lane] = LANE
    out[ground & ~lane & (prev == LANE)] = ROAD
    return out
