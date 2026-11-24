"""HPAC2 = #150's IntegerHPAC + a t-2 temporal branch (conv_past2 on one-hot(warp(tokens[t-2]))) folded into `past`.
Shared by training (hpac_train.py), the encoder (encode_prev2.py) and our decoder, so all three compute bit-identical
contexts.  The t-2 branch is serialized as EXT record 0x30 (see pack_record / unpack_record).
"""
import struct
import brotli
import numpy as np
import torch
from torch.nn import functional as F
from container.context_warp import apply_warp

EXT_TAG_PREV2 = 0x30


def make_hpac2_class(HI):
    class HPAC2(HI.IntegerHPAC):
        def __init__(self, k2=3, **kw):
            super().__init__(**kw)
            ck = dict(weight_bound=kw['weight_bound'], use_weight_scales=kw['use_weight_scales'], exponent_min=kw['weight_exponent_min'])
            self.k2 = k2
            self.conv_past2 = HI.IntegerConv2d(kw['num_classes'], kw['channels'], k2, padding=k2 // 2, **ck)
            with torch.no_grad():
                self.conv_past2.weight.zero_(); self.conv_past2.bias.zero_()
            self._HI = HI

        def prepare_frame_context(self, idx, previous_raw, previous2_raw=None):
            shift, past, scale, spm = super().prepare_frame_context(idx, previous_raw)
            if previous2_raw is None:
                return shift, past, scale, spm
            p1 = F.one_hot(previous_raw, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            p2 = F.one_hot(previous2_raw, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            full = self._HI.requantize(self.conv_past(p1) + self.conv_past2(p2), 0, -self.activation_bound, self.activation_bound)
            return shift, self._to_patches(full), scale, spm
    return HPAC2


def prev2_prepare(model, idx, frame, decoded, warp_ids):
    """decoded: [N,H,W] long (only frames < `frame` are read). Same construction as hpac_train.prev_frames."""
    z = torch.zeros((1,) + tuple(decoded.shape[1:]), dtype=torch.long, device=decoded.device)
    p1 = apply_warp(decoded[frame - 1][None], warp_ids[frame]) if frame >= 1 else z
    p2 = apply_warp(decoded[frame - 2][None], warp_ids[frame]) if frame >= 2 else z
    return model.prepare_frame_context(idx, p1, p2)


def pack_record(conv):
    """conv_past2 -> EXT record bytes: [0x30][u8 k2][u32 len][brotli(exp i8 x64 | bias i16 x64 | weight i8 x64*5*k2*k2)]"""
    w, b, e = conv.codes()
    k2 = w.shape[-1]
    body = (e.detach().round().cpu().numpy().astype(np.int8).tobytes() + b.detach().round().cpu().numpy().astype('<i2').tobytes()
            + w.detach().round().cpu().numpy().astype(np.int8).tobytes())
    comp = brotli.compress(body, quality=11)
    return bytes([EXT_TAG_PREV2, k2]) + struct.pack('<I', len(comp)) + comp


def unpack_record(rec, conv):
    assert rec[0] == EXT_TAG_PREV2
    k2 = rec[1]; n = struct.unpack_from('<I', rec, 2)[0]; body = brotli.decompress(rec[6:6 + n])
    c_out = conv.weight.shape[0]; c_in = conv.weight.shape[1]
    e = np.frombuffer(body[:c_out], np.int8); b = np.frombuffer(body[c_out:3 * c_out], '<i2')
    w = np.frombuffer(body[3 * c_out:], np.int8).reshape(c_out, c_in, k2, k2)
    with torch.no_grad():
        conv.exponent.copy_(torch.from_numpy(e.astype(np.float32))); conv.bias.copy_(torch.from_numpy(b.astype(np.float32)))
        conv.weight.copy_(torch.from_numpy(w.astype(np.float32)))
    return 6 + n


# ---------------------------------------------------------------------------------------------------------------------
# Shared-weights t-2 branch (EXT tag 0x31): reuses conv_past's integer weights on one-hot(t-2) with per-output-channel
# integer gain m, exponent e and bias b only:  past = requantize(conv(p1;Wp)*2^ep + bp + conv(p2; Wp*m)*2^e + b, 0)
EXT_TAG_SHARE = 0x31


def make_hpacs_class(HI):
    import torch.nn as nn

    class HPACS(HI.IntegerHPAC):
        def __init__(self, **kw):
            super().__init__(**kw)
            c = kw['channels']
            self.m2 = nn.Parameter(torch.zeros(c)); self.b2 = nn.Parameter(torch.zeros(c)); self.e2 = nn.Parameter(torch.full((c,), -4.0))
            self._HI = HI

        def prepare_frame_context(self, idx, previous_raw, previous2_raw=None):
            shift, past, scale, spm = super().prepare_frame_context(idx, previous_raw)
            if previous2_raw is None:
                return shift, past, scale, spm
            HI_ = self._HI
            p1 = F.one_hot(previous_raw, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            p2 = F.one_hot(previous2_raw, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            w, _, _ = self.conv_past.codes()
            m = HI_.ste_round(self.m2.clamp(-127, 127)); e = HI_.ste_round(self.e2.clamp(-8, 0)); b = HI_.ste_round(self.b2.clamp(-32768, 32767))
            t2 = F.conv2d(p2, w * m.view(-1, 1, 1, 1), None, padding=1) * torch.pow(2.0, e).view(1, -1, 1, 1) + b.view(1, -1, 1, 1)
            full = HI_.requantize(self.conv_past(p1) + t2, 0, -self.activation_bound, self.activation_bound)
            return shift, self._to_patches(full), scale, spm
    return HPACS


def pack_record_share(model):
    body = (model.m2.detach().round().clamp(-127, 127).cpu().numpy().astype(np.int8).tobytes()
            + model.e2.detach().round().clamp(-8, 0).cpu().numpy().astype(np.int8).tobytes()
            + model.b2.detach().round().clamp(-32768, 32767).cpu().numpy().astype('<i2').tobytes())
    comp = brotli.compress(body, quality=11)
    return bytes([EXT_TAG_SHARE, 0]) + struct.pack('<I', len(comp)) + comp


def unpack_record_share(rec, model):
    assert rec[0] == EXT_TAG_SHARE
    n = struct.unpack_from('<I', rec, 2)[0]; body = brotli.decompress(rec[6:6 + n]); c = model.m2.numel()
    m = np.frombuffer(body[:c], np.int8); e = np.frombuffer(body[c:2 * c], np.int8); b = np.frombuffer(body[2 * c:4 * c], '<i2')
    with torch.no_grad():
        model.m2.copy_(torch.from_numpy(m.astype(np.float32))); model.e2.copy_(torch.from_numpy(e.astype(np.float32)))
        model.b2.copy_(torch.from_numpy(b.astype(np.float32)))
    return 6 + n
