"""Flat FP4 serializer for Ship 4 model.pt.br polish sweep.

Replaces torch.save legacy format (23.3 KB envelope, 25.9% of payload)
with a compact [manifest_json | concat_tensor_bytes] layout, brotli-compressed.

Compatible with Ship 4 state_dict:
  - __codebook__ (fp32 [8])
  - quantized: dict of per-module records (conv2d / embedding / linear), with
    weight_kind in {fp4_packed, fp16} + optional bias_fp16
  - dense_fp16: residual dict of un-quantized norm/conv weights

Round-trip is bit-identical: no fp16 round-off, no lossy steps. Only the
serialization envelope changes.
"""
from __future__ import annotations

import json
import struct
from typing import Any

import brotli
import numpy as np
import torch

FLAT_MAGIC = b"QFP4"
FLAT_VERSION = 1

_DTYPE_MAP: dict[str, tuple[np.dtype, int]] = {
    "f32": (np.dtype("<f4"), 4),
    "f16": (np.dtype("<f2"), 2),
    "u8": (np.dtype("<u1"), 1),
    "i64": (np.dtype("<i8"), 8),
}


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().cpu().numpy().tobytes()


def _torch_dtype_to_name(dt: torch.dtype) -> str:
    if dt == torch.float16:
        return "f16"
    if dt == torch.float32:
        return "f32"
    if dt == torch.uint8:
        return "u8"
    if dt == torch.int64:
        return "i64"
    raise ValueError(f"unsupported dtype: {dt}")


def encode_flat(sd: dict[str, Any]) -> bytes:
    """Encode Ship-4-style FP4 state dict into flat brotli-compressed bytes."""
    manifest: list[dict[str, Any]] = []
    payload = bytearray()

    cb = sd["__codebook__"]
    manifest.append({"t": "cb", "o": len(payload), "l": cb.numel() * 4, "s": list(cb.shape)})
    payload += _tensor_bytes(cb.float())

    for name, rec in sd["quantized"].items():
        e: dict[str, Any] = {"t": "q", "n": name, "k": rec["type"], "ws": rec["weight_shape"]}
        if rec["weight_kind"] == "fp4_packed":
            pw = rec["packed_weight"]
            sc = rec["scales_fp16"]
            e["wk"] = "fp4"
            e["wn"] = rec["weight_numel"]
            e["pw"] = [len(payload), pw.numel()]
            payload += _tensor_bytes(pw)
            e["sc"] = [len(payload), sc.numel() * 2]
            payload += _tensor_bytes(sc.half())
        else:
            w = rec["weight_fp16"]
            e["wk"] = "fp16"
            e["w"] = [len(payload), w.numel() * 2]
            payload += _tensor_bytes(w.half())
        if rec.get("bias_fp16") is not None:
            b = rec["bias_fp16"]
            e["b"] = [len(payload), b.numel() * 2]
            payload += _tensor_bytes(b.half())
        if rec["type"] == "conv2d":
            e["cv"] = [rec["stride"], rec["padding"], rec["dilation"], rec["groups"]]
        manifest.append(e)

    for name, t in sd["dense_fp16"].items():
        dt_name = _torch_dtype_to_name(t.dtype)
        elem = _DTYPE_MAP[dt_name][1]
        manifest.append({
            "t": "d",
            "n": name,
            "d": dt_name,
            "s": list(t.shape),
            "o": len(payload),
            "l": t.numel() * elem,
        })
        payload += _tensor_bytes(t)

    mj = json.dumps(manifest, separators=(",", ":")).encode()
    header = FLAT_MAGIC + struct.pack("<BI", FLAT_VERSION, len(mj))
    raw = header + mj + bytes(payload)
    return brotli.compress(raw, quality=11)


def decode_flat(blob: bytes) -> dict[str, torch.Tensor]:
    """Decode flat blob into a load_state_dict-compatible dict."""
    raw = brotli.decompress(blob)
    assert raw[:4] == FLAT_MAGIC, f"bad magic {raw[:4]!r}"
    version = raw[4]
    assert version == FLAT_VERSION, f"unsupported flat version {version}"
    mj_len = struct.unpack("<I", raw[5:9])[0]
    manifest = json.loads(raw[9:9 + mj_len].decode())
    pay = memoryview(raw)[9 + mj_len:]

    codebook: torch.Tensor | None = None
    q_entries: list[dict[str, Any]] = []
    d_entries: list[dict[str, Any]] = []
    for e in manifest:
        if e["t"] == "cb":
            arr = np.frombuffer(pay[e["o"]:e["o"] + e["l"]], dtype="<f4").reshape(e["s"])
            codebook = torch.from_numpy(arr.copy())
        elif e["t"] == "q":
            q_entries.append(e)
        elif e["t"] == "d":
            d_entries.append(e)
        else:
            raise ValueError(f"unknown entry type {e['t']}")

    assert codebook is not None, "missing __codebook__"

    state_dict: dict[str, torch.Tensor] = {}
    for e in q_entries:
        name = e["n"]
        ws = e["ws"]
        if e["wk"] == "fp4":
            pw_o, pw_l = e["pw"]
            sc_o, sc_l = e["sc"]
            packed = np.frombuffer(pay[pw_o:pw_o + pw_l], dtype="<u1").copy()
            scales = np.frombuffer(pay[sc_o:sc_o + sc_l], dtype="<f2").copy()
            nib = _unpack_nibbles(packed, packed.shape[0] * 2)
            w = _dequant_fp4(nib, scales, ws, codebook)
        else:
            w_o, w_l = e["w"]
            w16 = np.frombuffer(pay[w_o:w_o + w_l], dtype="<f2").copy().reshape(ws)
            w = torch.from_numpy(w16).float()
        state_dict[f"{name}.weight"] = w
        if "b" in e:
            b_o, b_l = e["b"]
            b16 = np.frombuffer(pay[b_o:b_o + b_l], dtype="<f2").copy()
            state_dict[f"{name}.bias"] = torch.from_numpy(b16).float()

    for e in d_entries:
        dt = _DTYPE_MAP[e["d"]][0]
        arr = np.frombuffer(pay[e["o"]:e["o"] + e["l"]], dtype=dt).copy().reshape(e["s"])
        t = torch.from_numpy(arr)
        state_dict[e["n"]] = t.float() if t.is_floating_point() else t

    return state_dict


def _unpack_nibbles(packed: np.ndarray, count: int) -> np.ndarray:
    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2] = hi
    out[1::2] = lo
    return out[:count]


def _dequant_fp4(nibbles: np.ndarray, scales: np.ndarray, shape: list[int], codebook: torch.Tensor) -> torch.Tensor:
    flat_n = int(np.prod(shape))
    block_size = nibbles.size // scales.size
    nib = nibbles.reshape(-1, block_size).astype(np.int64)
    signs = (nib >> 3).astype(np.int64)
    mag = (nib & 0x7).astype(np.int64)
    levels = codebook.numpy().astype(np.float32)
    q = levels[mag]
    q = np.where(signs.astype(bool), -q, q)
    dq = q * scales.astype(np.float32)[:, None]
    return torch.from_numpy(dq.reshape(-1)[:flat_n].reshape(shape).copy()).float()
