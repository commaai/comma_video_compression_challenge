"""One more temporal context for the token model (extension record 0x40): the previous token map, re-projected."""
import importlib
import struct

import numpy as np
import torch
from torch.nn import functional as F

PARAMS = 5
PROB_BITS = 12
ONE = 1 << PROB_BITS


def composite(prev, tau_q, v0_q, u0_q, dx=0, dy=0):
    H, W = prev.shape
    prev = prev.long()
    v = torch.arange(H, device=prev.device, dtype=torch.int64)[:, None]
    u = torch.arange(W, device=prev.device, dtype=torch.int64)[None, :]
    a = 2 * v + 1 - int(v0_q)
    d = 16000 + int(tau_q) * a.clamp_min(0)
    vp = torch.div(int(v0_q) * d + 16000 * a, 2 * d, rounding_mode="floor") - int(dy)
    up = torch.div(2 * int(u0_q) * d + (2 * u + 1 - 2 * int(u0_q)) * 16000, 2 * d, rounding_mode="floor") - int(dx)
    src = prev[vp.clamp(0, H - 1).expand(H, W), up.clamp(0, W - 1)]
    ground = prev <= 1
    lane = (src == 1) & (a > 0)
    out = prev.clone()
    out[ground & lane] = 1
    out[ground & ~lane & (prev == 1)] = 0
    return out


def attach(model, codes, device):
    hpac_integer = importlib.import_module("hpac_integer")
    conv = hpac_integer.IntegerConv2d(model.num_classes, model.ch, 3, padding=1, weight_bound=127,
                                      use_weight_scales=True, exponent_min=-6)
    weight, bias, exponent = codes
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(np.asarray(weight, dtype=np.float32)))
        conv.bias.copy_(torch.from_numpy(np.asarray(bias, dtype=np.float32)))
        conv.exponent.copy_(torch.from_numpy(np.asarray(exponent, dtype=np.float32)))
    model.conv_mc = conv.to(device)
    return model


def add_branch(model, context, composite_map):
    hpac_integer = importlib.import_module("hpac_integer")
    shift, past, scale, spm = context
    batch, height, width = composite_map.shape
    one_hot = F.one_hot(composite_map, num_classes=model.num_classes).permute(0, 3, 1, 2).float()
    full = model._from_patches(past, batch, height // model.P, width // model.P) + model.conv_mc(one_hot)
    past = model._to_patches(hpac_integer.requantize(full, 0, -model.activation_bound, model.activation_bound))
    return shift, past, scale, spm


class Bit:
    __slots__ = ("n0", "n1")
    LIMIT = 1023

    def __init__(self):
        self.n0 = 0
        self.n1 = 0

    def p0(self):
        p = ((2 * self.n0 + 1) << PROB_BITS) // (2 * (self.n0 + self.n1) + 2)
        return min(max(p, 8), ONE - 8)

    def update(self, bit):
        if bit:
            self.n1 += 1
        else:
            self.n0 += 1
        if self.n0 + self.n1 > self.LIMIT:
            self.n0 = (self.n0 + 1) >> 1
            self.n1 = (self.n1 + 1) >> 1


class Models(dict):
    def __missing__(self, key):
        m = self[key] = Bit()
        return m


class Decoder:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.range = 0xFFFFFFFF
        self.code = 0
        for _ in range(4):
            self.code = (self.code << 8) | self._byte()

    def _byte(self):
        b = self.data[self.pos] if self.pos < len(self.data) else 0
        self.pos += 1
        return b

    def bit(self, model):
        bound = (self.range >> PROB_BITS) * model.p0()
        if self.code < bound:
            self.range = bound
            bit = 0
        else:
            self.code -= bound
            self.range -= bound
            bit = 1
        model.update(bit)
        while self.range < (1 << 24):
            self.range = (self.range << 8) & 0xFFFFFFFF
            self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF
        return bit


def _uint(coder, models, ctx):
    k = 0
    while coder.bit(models[(ctx, "u", k)]):
        k += 1
    out = 0
    for i in range(k - 1, -1, -1):
        out = (out << 1) | coder.bit(models[(ctx, "m", k, i)] if i >= k - 2 else models[("raw",)])
    return (1 << k) + out - 1


def _int(coder, models, ctx):
    if not coder.bit(models[(ctx, "z")]):
        return 0
    negative = coder.bit(models[(ctx, "s")])
    magnitude = _uint(coder, models, ctx) + 1
    return -magnitude if negative else magnitude


def decode_record(blob):
    channels, frames, rho = struct.unpack_from("<BHB", blob)
    base = np.frombuffer(blob[4:4 + 2 * PARAMS], dtype="<i2").astype(np.int64)
    weight = np.zeros((channels, 5, 3, 3), np.int64)
    bias = np.zeros(channels, np.int64)
    exponent = np.zeros(channels, np.int64)
    params = np.zeros((frames, PARAMS), np.int64)
    coder, models = Decoder(blob[4 + 2 * PARAMS:]), Models()
    for c in range(channels):
        exponent[c] = _int(coder, models, ("e",))
        bias[c] = _int(coder, models, ("b",))
        for i in range(5):
            for k in range(9):
                weight[c, i, k // 3, k % 3] = _int(coder, models, ("w", i, min(abs(int(exponent[c])), 6)))
    for t in range(1, frames):
        for k in range(PARAMS):
            pred = (rho * int(params[t - 1, k])) >> 2 if t > 1 else 0
            params[t, k] = pred + _int(coder, models, ("p", k))
    params[1:] += base
    return weight, bias, exponent, params
