"""Parse / re-encode the shipped SM3R(mixed) + LAY1 semantic-renderer section.

Mirrors external/pr141 ddm_mp2_semantic_receiver (decoder) and compress.py segment_semantic (layout).
A renderer is represented as `RendererQ`: per-tensor (depth, scales fp16, int codes) + fp16 vectors + FiLM row masks.
"""
from __future__ import annotations
import struct, sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "external/pr141")); sys.path.insert(0, str(REPO / "external/pr141/cpr1"))
import ddm_mp2_semantic_receiver as sch  # noqa: E402
from runtime.segment_layout import pack_segments, unpack_segments  # noqa: E402
import inflate as renderer  # noqa: E402  (external/pr141/cpr1/inflate.py)

TEMPLATE = renderer.SemanticTokenRenderer(96).state_dict()
QNAMES = sch._quantized_names(TEMPLATE)
PRUNE = sch.ROW_PRUNE_NAMES


@dataclass
class RendererQ:
    keep_percent: int
    depth: dict            # qname -> bits
    scales: dict           # qname -> fp16 np (kept rows only for pruned)
    codes: dict            # qname -> int8 np [rows_kept*cols] (kept rows only for pruned)
    rowmask: dict          # pruned qname -> bool[rows]
    fp16: dict             # vector name -> fp16 np

    def copy(self):
        c = lambda d: {k: v.copy() for k, v in d.items()}
        return RendererQ(self.keep_percent, dict(self.depth), c(self.scales), c(self.codes), c(self.rowmask), c(self.fp16))

    def state_dict(self):
        out = {}
        for name, tmpl in TEMPLATE.items():
            if tmpl.ndim < 2:
                out[name] = torch.from_numpy(self.fp16[name].astype(np.float32)).reshape(tmpl.shape); continue
            emb = name.endswith("embed.weight")
            sc = torch.from_numpy(self.scales[name].astype(np.float32))
            if name in PRUNE:
                rows = tmpl.shape[0]; cols = tmpl.numel() // rows
                k = int(self.rowmask[name].sum())
                c = torch.from_numpy(self.codes[name].astype(np.float32)).reshape(k, cols) * sc[:, None]
                d = torch.zeros(rows, cols); d[torch.from_numpy(self.rowmask[name])] = c
                out[name] = d.reshape(tmpl.shape)
            else:
                shape = [1] * tmpl.ndim; shape[-1 if emb else 0] = sc.numel()
                out[name] = torch.from_numpy(self.codes[name].astype(np.float32)).reshape(tmpl.shape) * sc.reshape(shape)
        return out


def _rd_bits(blob, count, bits):
    v, rest = sch._unpack_signed_bits(blob, count, bits); return v.numpy(), rest


def parse(section: bytes) -> RendererQ:
    blob = unpack_segments(section)
    assert blob[:4] == b"SM3R" and blob[5] == 6, "expected SM3R mixed"
    keep = blob[6]
    rem = memoryview(blob)[10:]
    depths, rem = sch._decode_depth_nibbles(rem, len(QNAMES), "SM3R")
    q = RendererQ(keep, dict(zip(QNAMES, depths)), {}, {}, {}, {})
    for name, t in TEMPLATE.items():
        if t.ndim < 2:
            a = np.frombuffer(rem[: t.numel() * 2], dtype="<f2").copy(); q.fp16[name] = a; rem = rem[t.numel() * 2:]; continue
        bits = q.depth[name]
        if name in PRUNE:
            rows = t.shape[0]; cols = t.numel() // rows
            mb = (rows + 7) // 8
            m = np.unpackbits(np.frombuffer(rem[:mb], dtype=np.uint8), bitorder="little")[:rows].astype(bool); rem = rem[mb:]
            k = int(m.sum()); q.rowmask[name] = m
            sc = np.frombuffer(rem[: k * 2], dtype="<f2").copy(); rem = rem[k * 2:]
            c, rem = _rd_bits(rem, k * cols, bits)
        else:
            n = sch._scale_count(name, t)
            sc = np.frombuffer(rem[: n * 2], dtype="<f2").copy(); rem = rem[n * 2:]
            c, rem = _rd_bits(rem, t.numel(), bits)
        q.scales[name] = sc; q.codes[name] = c
    assert len(rem) == 0
    return q


def _pack_bits(codes: np.ndarray, bits: int) -> bytes:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    assert codes.min() >= lo and codes.max() <= hi, (codes.min(), codes.max(), bits)
    u = (codes.astype(np.int16) & ((1 << bits) - 1)).astype(np.uint16)
    b = ((u[:, None] >> np.arange(bits, dtype=np.uint16)[None]) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(b, bitorder="little").tobytes()


def encode(q: RendererQ):
    """Returns (section_bytes_LAY1, raw_blob, segment_sizes, per_tensor_bytes)."""
    names = QNAMES
    header = b"SM3R" + bytes([1, 6, q.keep_percent, 0])
    mask = sum(1 << i for i, n in enumerate(names) if n in PRUNE)
    header += struct.pack("<H", mask)
    d = [q.depth[n] for n in names]
    if len(d) % 2: d.append(0)
    header += bytes(d[i] | (d[i + 1] << 4) for i in range(0, len(d), 2))
    metadata = bytearray(header); segs = []; per = {}
    for name, t in TEMPLATE.items():
        if t.ndim < 2:
            metadata += q.fp16[name].astype("<f2").tobytes(); continue
        m = b""
        if name in PRUNE:
            m += np.packbits(q.rowmask[name].astype(np.uint8), bitorder="little").tobytes()
        m += q.scales[name].astype("<f2").tobytes()
        metadata += m
        code = _pack_bits(q.codes[name].reshape(-1), q.depth[name])
        segs.extend((bytes(metadata), code)); metadata = bytearray()
        per[name] = dict(meta=len(m), codes=len(code), bits=q.depth[name], n=int(q.codes[name].size))
    if metadata: segs.append(bytes(metadata))
    blob = b"".join(segs)
    packed = pack_segments(segs)
    assert unpack_segments(packed) == blob
    return packed, blob, [len(s) for s in segs], per


# ---------------------------------------------------------------------------------------------- quantisation helpers
def qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def quantize_from_float(sd: dict, ref: RendererQ, depth: dict | None = None) -> RendererQ:
    """Float state dict -> RendererQ using the shipped scheme: symmetric, scale = absmax/qmax per row (per column for embeddings),
    scale stored fp16 and codes computed against the fp16 scale.  Row masks / keep_percent follow `ref`; depths follow `depth` or ref."""
    depth = dict(ref.depth if depth is None else depth)
    q = RendererQ(ref.keep_percent, depth, {}, {}, {k: v.copy() for k, v in ref.rowmask.items()}, {})
    for name, t in TEMPLATE.items():
        w = sd[name].detach().cpu().float()
        if t.ndim < 2:
            q.fp16[name] = w.numpy().reshape(-1).astype("<f2"); continue
        emb = name.endswith("embed.weight")
        qm = qmax(depth[name])
        if name in PRUNE:
            rows = t.shape[0]; w2 = w.reshape(rows, -1)[torch.from_numpy(ref.rowmask[name])]
            ax = w2.abs().amax(1)
        elif emb:
            w2 = w; ax = w.abs().amax(0)
        else:
            w2 = w.reshape(t.shape[0], -1); ax = w2.abs().amax(1)
        scale = (ax / qm).clamp_min(1e-8).numpy().astype("<f2")
        s32 = torch.from_numpy(scale.astype(np.float32)).clamp_min(1e-8)
        c = torch.round(w2 / (s32[None] if emb else s32[:, None])).clamp(-qm, qm)
        q.scales[name] = scale; q.codes[name] = c.numpy().astype(np.int8).reshape(-1)
    return q


def n_changed_codes(a: RendererQ, b: RendererQ) -> dict:
    ch = tot = 0; per = {}
    for n in QNAMES:
        if a.depth[n] != b.depth[n] or a.codes[n].shape != b.codes[n].shape:
            x = int(a.codes[n].size); per[n] = x; ch += x; tot += x; continue
        d = int((a.codes[n] != b.codes[n]).sum()); per[n] = d; ch += d; tot += int(a.codes[n].size)
    return dict(changed=ch, total=tot, pct=100 * ch / tot, per_tensor=per)
