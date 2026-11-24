import struct
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dz_bitcoder import Decoder, Models
H, W, NC = (384, 512, 5)

class Block(nn.Module):

    def __init__(self, width, dilation, frame_dim, groups):
        super().__init__()
        self.dw = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation, groups=width)
        self.pw = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(groups, width)
        self.film = nn.Linear(frame_dim, 2 * width) if frame_dim else None

    def forward(self, x, frame):
        r = self.norm(self.pw(self.dw(x)))
        if self.film is not None:
            s, b = self.film(frame).chunk(2, dim=1)
            r = r * (1.0 + s[:, :, None, None]) + b[:, :, None, None]
        return x + F.gelu(r)

class Renderer(nn.Module):

    def __init__(self, width=48, dilations=(1, 1, 2, 4), frame_dim=0, frames=600, head_kernel=3):
        super().__init__()
        self.width, self.dilations, self.frame_dim = (width, tuple(dilations), frame_dim)
        self.mix = nn.Conv2d(NC + 4, width, 1)
        self.frame_embed = nn.Embedding(frames, frame_dim) if frame_dim else None
        self.blocks = nn.ModuleList([Block(width, d, frame_dim, max(1, width // 8)) for d in dilations])
        self.head = nn.Conv2d(width, 3, head_kernel, padding=head_kernel // 2)
        yy, xx = torch.meshgrid(torch.linspace(-1.0, 1.0, H), torch.linspace(-1.0, 1.0, W), indexing='ij')
        self.register_buffer('coords', torch.stack([xx, yy, xx.square(), yy.square()])[None], persistent=False)

    def forward(self, tokens, index):
        x = torch.cat([F.one_hot(tokens, NC).permute(0, 3, 1, 2).to(self.coords.dtype), self.coords.expand(len(tokens), -1, -1, -1)], 1)
        x = self.mix(x)
        frame = self.frame_embed(index) if self.frame_embed is not None else None
        for block in self.blocks:
            x = block(x, frame)
        return torch.sigmoid(self.head(F.gelu(x))) * 255.0

def shapes_of(model):
    return [(name, tuple(p.shape)) for name, p in model.named_parameters()]

def _mag(c, m, ctx, mag):
    n = (mag + 1).bit_length() - 1
    k = 0
    while c.bit(m[ctx, 'u', k], 1 if k < n else 0):
        k += 1
    rest, out = (mag + 1 - (1 << k), 0)
    for i in range(k - 1, -1, -1):
        out = out << 1 | c.bit(m[ctx[0], 'm', k, i], rest >> i & 1)
    return (1 << k) + out - 1

def _int(c, m, ctx, v):
    if not c.bit(m[ctx, 'z'], 1 if v else 0):
        return 0
    neg = c.bit(m[ctx[0], 's'], 1 if v < 0 else 0)
    mag = _mag(c, m, ctx, abs(v) - 1) + 1
    return -mag if neg else mag

def _kind(name):
    parts = name.split('.')
    return parts[-2] + '.' + parts[-1] if parts[0] == 'blocks' else name

def _pass(c, shapes, q):
    enc = q is not None
    m = Models()
    out = {}
    for name, shape in shapes:
        kind = _kind(name)
        rows = shape[0] if len(shape) > 1 else 1
        vals = q[name].reshape(rows, -1).astype(np.int64) if enc else np.zeros((rows, int(np.prod(shape)) // rows), np.int64)
        for r in range(rows):
            acc = 1.0
            for j in range(vals.shape[1]):
                v = _int(c, m, (kind, min(int(acc * 2).bit_length(), 8)), int(vals[r, j]) if enc else 0)
                vals[r, j] = v
                acc = acc * 0.75 + abs(v) * 0.25
        out[name] = vals.reshape(shape)
    return out

def decode(blob):
    width, hk, fdim, nd = struct.unpack_from('<BBBB', blob)
    model = Renderer(width, tuple(blob[4:4 + nd]), fdim, head_kernel=hk)
    shapes = shapes_of(model)
    o = 4 + nd
    steps = np.frombuffer(blob[o:o + 2 * len(shapes)], dtype='<f2').astype(np.float32)
    q = _pass(Decoder(blob[o + 2 * len(shapes):]), shapes, None)
    state = {name: torch.from_numpy(q[name].astype(np.float32) * steps[i]) for i, (name, _) in enumerate(shapes)}
    model.load_state_dict(state, strict=True)
    return model.eval()
