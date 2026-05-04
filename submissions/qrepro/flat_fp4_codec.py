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
    return decode_flat_raw(raw)


def decode_flat_raw(raw: bytes) -> dict[str, torch.Tensor]:
    """Decode an already-decompressed flat blob."""
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


def decode_payload_only_for_model(blob: bytes, model: torch.nn.Module, block_size: int = 32) -> dict[str, torch.Tensor]:
    """Decode a manifest-free payload using the fixed qpose model structure.

    This mirrors ``encode_flat``'s tensor byte order for this architecture, but
    omits the JSON manifest from the archive. The decoder can recover shapes and
    dense state keys from the instantiated model.
    """
    pay = memoryview(brotli.decompress(blob))
    offset = 0
    codebook_arr = np.frombuffer(pay[offset:offset + 32], dtype="<f4").copy()
    codebook = torch.from_numpy(codebook_arr)
    offset += 32

    state_dict: dict[str, torch.Tensor] = {}
    covered: set[str] = set()
    omitted_zero_init = {
        "frame2_head.block1.film_proj.weight",
        "frame2_head.block1.film_proj.bias",
    }
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name not in {"QConv2d", "QEmbedding"}:
            continue
        weight_shape = list(module.weight.shape)
        weight_numel = int(module.weight.numel())
        # The public qpose export keeps embeddings and final RGB heads in fp16;
        # all other QConv2d weights are FP4 block-quantized.
        fp16_weight = cls_name == "QEmbedding" or name.endswith(".head")
        if fp16_weight:
            nbytes = weight_numel * 2
            arr = np.frombuffer(pay[offset:offset + nbytes], dtype="<f2").copy().reshape(weight_shape)
            state_dict[f"{name}.weight"] = torch.from_numpy(arr).float()
            offset += nbytes
        else:
            nblocks = (weight_numel + block_size - 1) // block_size
            packed_len = nblocks * block_size // 2
            scales_len = nblocks * 2
            packed = np.frombuffer(pay[offset:offset + packed_len], dtype="<u1").copy()
            offset += packed_len
            scales = np.frombuffer(pay[offset:offset + scales_len], dtype="<f2").copy()
            offset += scales_len
            nib = _unpack_nibbles(packed, packed.shape[0] * 2)
            state_dict[f"{name}.weight"] = _dequant_fp4(nib, scales, weight_shape, codebook)
        covered.add(f"{name}.weight")

        if cls_name == "QConv2d" and module.bias is not None:
            nbytes = int(module.bias.numel()) * 2
            arr = np.frombuffer(pay[offset:offset + nbytes], dtype="<f2").copy()
            state_dict[f"{name}.bias"] = torch.from_numpy(arr).float()
            offset += nbytes
            covered.add(f"{name}.bias")

    for name, tensor in model.state_dict().items():
        if name in covered:
            continue
        if name in omitted_zero_init:
            continue
        if torch.is_floating_point(tensor):
            nbytes = tensor.numel() * 2
            arr = np.frombuffer(pay[offset:offset + nbytes], dtype="<f2").copy().reshape(tuple(tensor.shape))
            state_dict[name] = torch.from_numpy(arr).float()
            offset += nbytes
        else:
            nbytes = tensor.numel() * 8
            arr = np.frombuffer(pay[offset:offset + nbytes], dtype="<i8").copy().reshape(tuple(tensor.shape))
            state_dict[name] = torch.from_numpy(arr)
            offset += nbytes

    if offset != len(pay):
        raise ValueError(f"payload-only model decode consumed {offset} bytes, payload has {len(pay)}")
    return state_dict


def decode_qrow_payload_for_model(blob: bytes, model: torch.nn.Module, block_size: int = 32) -> dict[str, torch.Tensor]:
    """Decode QFPL plus row-wise uint8 storage for selected frame1 FiLM rows."""
    pay = memoryview(brotli.decompress(blob))
    offset = 0
    codebook_arr = np.frombuffer(pay[offset:offset + 32], dtype="<f4").copy()
    codebook = torch.from_numpy(codebook_arr)
    offset += 32

    state_dict: dict[str, torch.Tensor] = {}
    covered: set[str] = set()
    omitted_zero_init = {
        "frame2_head.block1.film_proj.weight",
        "frame2_head.block1.film_proj.bias",
    }
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name not in {"QConv2d", "QEmbedding"}:
            continue
        weight_shape = list(module.weight.shape)
        weight_numel = int(module.weight.numel())
        fp16_weight = cls_name == "QEmbedding" or name.endswith(".head")
        if fp16_weight:
            nbytes = weight_numel * 2
            arr = np.frombuffer(pay[offset:offset + nbytes], dtype="<f2").copy().reshape(weight_shape)
            state_dict[f"{name}.weight"] = torch.from_numpy(arr).float()
            offset += nbytes
        else:
            nblocks = (weight_numel + block_size - 1) // block_size
            packed_len = nblocks * block_size // 2
            scales_len = nblocks * 2
            packed = np.frombuffer(pay[offset:offset + packed_len], dtype="<u1").copy()
            offset += packed_len
            scales = np.frombuffer(pay[offset:offset + scales_len], dtype="<f2").copy()
            offset += scales_len
            nib = _unpack_nibbles(packed, packed.shape[0] * 2)
            state_dict[f"{name}.weight"] = _dequant_fp4(nib, scales, weight_shape, codebook)
        covered.add(f"{name}.weight")

        if cls_name == "QConv2d" and module.bias is not None:
            nbytes = int(module.bias.numel()) * 2
            arr = np.frombuffer(pay[offset:offset + nbytes], dtype="<f2").copy()
            state_dict[f"{name}.bias"] = torch.from_numpy(arr).float()
            offset += nbytes
            covered.add(f"{name}.bias")

    special = "frame1_head.block1.film_proj.weight"
    for name, tensor in model.state_dict().items():
        if name in covered:
            continue
        if name in omitted_zero_init:
            continue
        if name == special:
            rows, cols = tuple(tensor.shape)
            mask_nbytes = (rows + 7) // 8
            mask_bytes = np.frombuffer(pay[offset:offset + mask_nbytes], dtype=np.uint8).copy()
            offset += mask_nbytes
            bits = np.unpackbits(mask_bytes, bitorder="little")[:rows].astype(bool)
            n_q = int(bits.sum())
            n_fp = rows - n_q
            mins = np.frombuffer(pay[offset:offset + n_q * 2], dtype="<f2").astype(np.float32)
            offset += n_q * 2
            scales = np.frombuffer(pay[offset:offset + n_q * 2], dtype="<f2").astype(np.float32)
            offset += n_q * 2
            q = np.frombuffer(pay[offset:offset + n_q * cols], dtype=np.uint8).astype(np.float32).reshape(n_q, cols)
            offset += n_q * cols
            fp = np.frombuffer(pay[offset:offset + n_fp * cols * 2], dtype="<f2").copy().reshape(n_fp, cols)
            offset += n_fp * cols * 2
            arr = np.empty((rows, cols), dtype=np.float16)
            arr[bits] = (q * scales[:, None] + mins[:, None]).astype(np.float16)
            arr[~bits] = fp
            state_dict[name] = torch.from_numpy(arr).float()
        elif torch.is_floating_point(tensor):
            nbytes = tensor.numel() * 2
            arr = np.frombuffer(pay[offset:offset + nbytes], dtype="<f2").copy().reshape(tuple(tensor.shape))
            state_dict[name] = torch.from_numpy(arr).float()
            offset += nbytes
        else:
            nbytes = tensor.numel() * 8
            arr = np.frombuffer(pay[offset:offset + nbytes], dtype="<i8").copy().reshape(tuple(tensor.shape))
            state_dict[name] = torch.from_numpy(arr)
            offset += nbytes

    if offset != len(pay):
        raise ValueError(f"qrow model decode consumed {offset} bytes, payload has {len(pay)}")
    return state_dict


def decode_qrow_grouped_payload_for_model(blob: bytes, model: torch.nn.Module, block_size: int = 32) -> dict[str, torch.Tensor]:
    """Decode QFQ2: the same tensors as QFQ1, grouped by storage kind.

    Grouping homogeneous byte streams gives Brotli slightly better context than
    interleaving packed weights, scales, and fp16 tensors module-by-module.
    """
    pay = memoryview(brotli.decompress(blob))
    offset = 0
    codebook_arr = np.frombuffer(pay[offset:offset + 32], dtype="<f4").copy()
    codebook = torch.from_numpy(codebook_arr)
    offset += 32

    state_dict: dict[str, torch.Tensor] = {}
    covered: set[str] = set()
    packed_modules: list[tuple[str, list[int], int, int]] = []
    fp16_modules: list[tuple[str, list[int], int]] = []
    bias_modules: list[tuple[str, int]] = []
    omitted_zero_init = {
        "frame2_head.block1.film_proj.weight",
        "frame2_head.block1.film_proj.bias",
    }

    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name not in {"QConv2d", "QEmbedding"}:
            continue
        weight_shape = list(module.weight.shape)
        weight_numel = int(module.weight.numel())
        fp16_weight = cls_name == "QEmbedding" or name.endswith(".head")
        if fp16_weight:
            fp16_modules.append((name, weight_shape, weight_numel))
        else:
            nblocks = (weight_numel + block_size - 1) // block_size
            packed_len = nblocks * block_size // 2
            scales_len = nblocks * 2
            packed_modules.append((name, weight_shape, packed_len, scales_len))
        covered.add(f"{name}.weight")

        if cls_name == "QConv2d" and module.bias is not None:
            bias_modules.append((name, int(module.bias.numel())))
            covered.add(f"{name}.bias")

    packed_bytes: dict[str, np.ndarray] = {}
    for name, _shape, packed_len, _scales_len in packed_modules:
        packed_bytes[name] = np.frombuffer(pay[offset:offset + packed_len], dtype="<u1").copy()
        offset += packed_len

    scale_bytes: dict[str, np.ndarray] = {}
    for name, _shape, _packed_len, scales_len in packed_modules:
        scale_bytes[name] = np.frombuffer(pay[offset:offset + scales_len], dtype="<f2").copy()
        offset += scales_len

    bias_bytes: dict[str, np.ndarray] = {}
    for name, n_bias in bias_modules:
        nbytes = n_bias * 2
        bias_bytes[name] = np.frombuffer(pay[offset:offset + nbytes], dtype="<f2").copy()
        offset += nbytes

    fp16_weight_bytes: dict[str, np.ndarray] = {}
    for name, shape, weight_numel in fp16_modules:
        nbytes = weight_numel * 2
        fp16_weight_bytes[name] = np.frombuffer(pay[offset:offset + nbytes], dtype="<f2").copy().reshape(shape)
        offset += nbytes

    for name, shape, _packed_len, _scales_len in packed_modules:
        packed = packed_bytes[name]
        scales = scale_bytes[name]
        nib = _unpack_nibbles(packed, packed.shape[0] * 2)
        state_dict[f"{name}.weight"] = _dequant_fp4(nib, scales, shape, codebook)
    for name, arr in fp16_weight_bytes.items():
        state_dict[f"{name}.weight"] = torch.from_numpy(arr).float()
    for name, arr in bias_bytes.items():
        state_dict[f"{name}.bias"] = torch.from_numpy(arr).float()

    special = "frame1_head.block1.film_proj.weight"
    special_shape: tuple[int, int] | None = None
    dense_float: list[tuple[str, tuple[int, ...], int]] = []
    dense_int: list[tuple[str, tuple[int, ...], int]] = []
    for name, tensor in model.state_dict().items():
        if name in covered:
            continue
        if name in omitted_zero_init:
            continue
        if name == special:
            special_shape = tuple(tensor.shape)
        elif torch.is_floating_point(tensor):
            dense_float.append((name, tuple(tensor.shape), int(tensor.numel())))
        else:
            dense_int.append((name, tuple(tensor.shape), int(tensor.numel())))

    for name, shape, numel in dense_float:
        nbytes = numel * 2
        arr = np.frombuffer(pay[offset:offset + nbytes], dtype="<f2").copy().reshape(shape)
        state_dict[name] = torch.from_numpy(arr).float()
        offset += nbytes

    for name, shape, numel in dense_int:
        nbytes = numel * 8
        arr = np.frombuffer(pay[offset:offset + nbytes], dtype="<i8").copy().reshape(shape)
        state_dict[name] = torch.from_numpy(arr)
        offset += nbytes

    if special_shape is not None:
        rows, cols = special_shape
        mask_nbytes = (rows + 7) // 8
        mask_bytes = np.frombuffer(pay[offset:offset + mask_nbytes], dtype=np.uint8).copy()
        offset += mask_nbytes
        bits = np.unpackbits(mask_bytes, bitorder="little")[:rows].astype(bool)
        n_q = int(bits.sum())
        n_fp = rows - n_q
        mins = np.frombuffer(pay[offset:offset + n_q * 2], dtype="<f2").astype(np.float32)
        offset += n_q * 2
        scales = np.frombuffer(pay[offset:offset + n_q * 2], dtype="<f2").astype(np.float32)
        offset += n_q * 2
        q = np.frombuffer(pay[offset:offset + n_q * cols], dtype=np.uint8).astype(np.float32).reshape(n_q, cols)
        offset += n_q * cols
        fp = np.frombuffer(pay[offset:offset + n_fp * cols * 2], dtype="<f2").copy().reshape(n_fp, cols)
        offset += n_fp * cols * 2
        arr = np.empty((rows, cols), dtype=np.float16)
        arr[bits] = (q * scales[:, None] + mins[:, None]).astype(np.float16)
        arr[~bits] = fp
        state_dict[special] = torch.from_numpy(arr).float()

    if offset != len(pay):
        raise ValueError(f"qrow grouped model decode consumed {offset} bytes, payload has {len(pay)}")
    return state_dict


def decode_qrow_grouped3_payload_for_model(blob: bytes, model: torch.nn.Module, block_size: int = 32) -> dict[str, torch.Tensor]:
    """Decode QFQ3: packed weights, one fp16 byte-plane group, then raw tensors."""
    pay = memoryview(brotli.decompress(blob))
    offset = 0

    state_dict: dict[str, torch.Tensor] = {}
    covered: set[str] = set()
    packed_modules: list[tuple[str, list[int], int, int]] = []
    fp16_modules: list[tuple[str, list[int], int]] = []
    bias_modules: list[tuple[str, int]] = []
    omitted_zero_init = {
        "frame2_head.block1.film_proj.weight",
        "frame2_head.block1.film_proj.bias",
    }

    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name not in {"QConv2d", "QEmbedding"}:
            continue
        weight_shape = list(module.weight.shape)
        weight_numel = int(module.weight.numel())
        fp16_weight = cls_name == "QEmbedding" or name.endswith(".head")
        if fp16_weight:
            fp16_modules.append((name, weight_shape, weight_numel))
        else:
            nblocks = (weight_numel + block_size - 1) // block_size
            packed_modules.append((name, weight_shape, nblocks * block_size // 2, nblocks * 2))
        covered.add(f"{name}.weight")

        if cls_name == "QConv2d" and module.bias is not None:
            bias_modules.append((name, int(module.bias.numel())))
            covered.add(f"{name}.bias")

    packed_bytes: dict[str, np.ndarray] = {}
    for name, _shape, packed_len, _scales_len in packed_modules:
        packed_bytes[name] = np.frombuffer(pay[offset:offset + packed_len], dtype="<u1").copy()
        offset += packed_len

    special = "frame1_head.block1.film_proj.weight"
    special_shape: tuple[int, int] | None = None
    dense_float: list[tuple[str, tuple[int, ...], int]] = []
    dense_int: list[tuple[str, tuple[int, ...], int]] = []
    for name, tensor in model.state_dict().items():
        if name in covered:
            continue
        if name in omitted_zero_init:
            continue
        if name == special:
            special_shape = tuple(tensor.shape)
        elif torch.is_floating_point(tensor):
            dense_float.append((name, tuple(tensor.shape), int(tensor.numel())))
        else:
            dense_int.append((name, tuple(tensor.shape), int(tensor.numel())))
    if special_shape is None:
        raise ValueError("QFQ3 requires frame1 FiLM qrow tensor")

    rows, cols = special_shape
    mask_nbytes = (rows + 7) // 8
    known_fp16 = (
        sum(scales_len for _name, _shape, _packed_len, scales_len in packed_modules)
        + sum(n_bias * 2 for _name, n_bias in bias_modules)
        + sum(weight_numel * 2 for _name, _shape, weight_numel in fp16_modules)
        + sum(numel * 2 for _name, _shape, numel in dense_float)
    )
    known_raw = 32 + sum(numel * 8 for _name, _shape, numel in dense_int) + mask_nbytes
    remaining = len(pay) - offset
    const_total = known_fp16 + known_raw + rows * cols * 2
    denom = cols - 4
    if denom <= 0 or (const_total - remaining) % denom:
        raise ValueError("invalid QFQ3 grouped lengths")
    n_q = (const_total - remaining) // denom
    if n_q < 0 or n_q > rows:
        raise ValueError("invalid QFQ3 qrow count")
    n_fp = rows - n_q
    fp16_len = known_fp16 + n_q * 4 + n_fp * cols * 2
    raw_len = known_raw + n_q * cols
    if fp16_len + raw_len != remaining:
        raise ValueError("invalid QFQ3 length split")

    fp16_planed = np.frombuffer(pay[offset:offset + fp16_len], dtype=np.uint8).copy()
    offset += fp16_len
    half = (fp16_len + 1) // 2
    fp16_bytes = np.empty(fp16_len, dtype=np.uint8)
    fp16_bytes[0::2] = fp16_planed[:half]
    fp16_bytes[1::2] = fp16_planed[half:]
    fp16_pay = memoryview(fp16_bytes)
    fp16_offset = 0

    raw_offset = offset
    codebook_arr = np.frombuffer(pay[raw_offset:raw_offset + 32], dtype="<f4").copy()
    codebook = torch.from_numpy(codebook_arr)
    raw_offset += 32

    dense_int_bytes: dict[str, np.ndarray] = {}
    for name, shape, numel in dense_int:
        nbytes = numel * 8
        dense_int_bytes[name] = np.frombuffer(pay[raw_offset:raw_offset + nbytes], dtype="<i8").copy().reshape(shape)
        raw_offset += nbytes

    mask_bytes = np.frombuffer(pay[raw_offset:raw_offset + mask_nbytes], dtype=np.uint8).copy()
    raw_offset += mask_nbytes
    bits = np.unpackbits(mask_bytes, bitorder="little")[:rows].astype(bool)
    if int(bits.sum()) != n_q:
        raise ValueError("QFQ3 qrow mask count mismatch")
    qrow_q = np.frombuffer(pay[raw_offset:raw_offset + n_q * cols], dtype=np.uint8).astype(np.float32).reshape(n_q, cols)
    raw_offset += n_q * cols
    if raw_offset != len(pay):
        raise ValueError("QFQ3 raw group length mismatch")

    scale_bytes: dict[str, np.ndarray] = {}
    for name, _shape, _packed_len, scales_len in packed_modules:
        scale_bytes[name] = np.frombuffer(fp16_pay[fp16_offset:fp16_offset + scales_len], dtype="<f2").copy()
        fp16_offset += scales_len

    bias_bytes: dict[str, np.ndarray] = {}
    for name, n_bias in bias_modules:
        nbytes = n_bias * 2
        bias_bytes[name] = np.frombuffer(fp16_pay[fp16_offset:fp16_offset + nbytes], dtype="<f2").copy()
        fp16_offset += nbytes

    fp16_weight_bytes: dict[str, np.ndarray] = {}
    for name, shape, weight_numel in fp16_modules:
        nbytes = weight_numel * 2
        fp16_weight_bytes[name] = np.frombuffer(fp16_pay[fp16_offset:fp16_offset + nbytes], dtype="<f2").copy().reshape(shape)
        fp16_offset += nbytes

    dense_float_bytes: dict[str, np.ndarray] = {}
    for name, shape, numel in dense_float:
        nbytes = numel * 2
        dense_float_bytes[name] = np.frombuffer(fp16_pay[fp16_offset:fp16_offset + nbytes], dtype="<f2").copy().reshape(shape)
        fp16_offset += nbytes

    qrow_min = np.frombuffer(fp16_pay[fp16_offset:fp16_offset + n_q * 2], dtype="<f2").astype(np.float32)
    fp16_offset += n_q * 2
    qrow_scale = np.frombuffer(fp16_pay[fp16_offset:fp16_offset + n_q * 2], dtype="<f2").astype(np.float32)
    fp16_offset += n_q * 2
    qrow_fp = np.frombuffer(fp16_pay[fp16_offset:fp16_offset + n_fp * cols * 2], dtype="<f2").copy().reshape(n_fp, cols)
    fp16_offset += n_fp * cols * 2
    if fp16_offset != len(fp16_pay):
        raise ValueError("QFQ3 fp16 group length mismatch")

    for name, shape, _packed_len, _scales_len in packed_modules:
        packed = packed_bytes[name]
        scales = scale_bytes[name]
        nib = _unpack_nibbles(packed, packed.shape[0] * 2)
        state_dict[f"{name}.weight"] = _dequant_fp4(nib, scales, shape, codebook)
    for name, arr in fp16_weight_bytes.items():
        state_dict[f"{name}.weight"] = torch.from_numpy(arr).float()
    for name, arr in bias_bytes.items():
        state_dict[f"{name}.bias"] = torch.from_numpy(arr).float()
    for name, arr in dense_float_bytes.items():
        state_dict[name] = torch.from_numpy(arr).float()
    for name, arr in dense_int_bytes.items():
        state_dict[name] = torch.from_numpy(arr)

    qrow = np.empty((rows, cols), dtype=np.float16)
    qrow[bits] = (qrow_q * qrow_scale[:, None] + qrow_min[:, None]).astype(np.float16)
    qrow[~bits] = qrow_fp
    state_dict[special] = torch.from_numpy(qrow).float()

    return state_dict


def decode_qrow_grouped4_payload_for_model(blob: bytes, model: torch.nn.Module, block_size: int = 32) -> dict[str, torch.Tensor]:
    """Decode QFQ4: QFQ3 tensors with better Brotli-local byte ordering."""
    pay = memoryview(brotli.decompress(blob))
    offset = 0

    state_dict: dict[str, torch.Tensor] = {}
    covered: set[str] = set()
    packed_modules: list[tuple[str, list[int], int, int]] = []
    fp16_modules: list[tuple[str, list[int], int]] = []
    bias_modules: list[tuple[str, int]] = []
    omitted_zero_init = {
        "frame2_head.block1.film_proj.weight",
        "frame2_head.block1.film_proj.bias",
    }

    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name not in {"QConv2d", "QEmbedding"}:
            continue
        weight_shape = list(module.weight.shape)
        weight_numel = int(module.weight.numel())
        fp16_weight = cls_name == "QEmbedding" or name.endswith(".head")
        if fp16_weight:
            fp16_modules.append((name, weight_shape, weight_numel))
        else:
            nblocks = (weight_numel + block_size - 1) // block_size
            packed_modules.append((name, weight_shape, nblocks * block_size // 2, nblocks * 2))
        covered.add(f"{name}.weight")

        if cls_name == "QConv2d" and module.bias is not None:
            bias_modules.append((name, int(module.bias.numel())))
            covered.add(f"{name}.bias")

    packed_order = sorted(packed_modules, key=lambda item: (".pw" in item[0], item[0]))
    packed_bytes: dict[str, np.ndarray] = {}
    for name, _shape, packed_len, _scales_len in packed_order:
        packed_bytes[name] = np.frombuffer(pay[offset:offset + packed_len], dtype="<u1").copy()
        offset += packed_len

    special = "frame1_head.block1.film_proj.weight"
    special_shape: tuple[int, int] | None = None
    dense_float: list[tuple[str, tuple[int, ...], int]] = []
    dense_int: list[tuple[str, tuple[int, ...], int]] = []
    for name, tensor in model.state_dict().items():
        if name in covered:
            continue
        if name in omitted_zero_init:
            continue
        if name == special:
            special_shape = tuple(tensor.shape)
        elif torch.is_floating_point(tensor):
            dense_float.append((name, tuple(tensor.shape), int(tensor.numel())))
        else:
            dense_int.append((name, tuple(tensor.shape), int(tensor.numel())))
    if special_shape is None:
        raise ValueError("QFQ4 requires frame1 FiLM qrow tensor")

    rows, cols = special_shape
    mask_nbytes = (rows + 7) // 8
    known_fp16 = (
        sum(scales_len for _name, _shape, _packed_len, scales_len in packed_modules)
        + sum(n_bias * 2 for _name, n_bias in bias_modules)
        + sum(weight_numel * 2 for _name, _shape, weight_numel in fp16_modules)
        + sum(numel * 2 for _name, _shape, numel in dense_float)
    )
    known_raw = 32 + sum(numel * 8 for _name, _shape, numel in dense_int) + mask_nbytes
    remaining = len(pay) - offset
    const_total = known_fp16 + known_raw + rows * cols * 2
    denom = cols - 4
    if denom <= 0 or (const_total - remaining) % denom:
        raise ValueError("invalid QFQ4 grouped lengths")
    n_q = (const_total - remaining) // denom
    if n_q < 0 or n_q > rows:
        raise ValueError("invalid QFQ4 qrow count")
    n_fp = rows - n_q
    fp16_len = known_fp16 + n_q * 4 + n_fp * cols * 2
    raw_len = known_raw + n_q * cols
    if fp16_len + raw_len != remaining:
        raise ValueError("invalid QFQ4 length split")

    fp16_planed = np.frombuffer(pay[offset:offset + fp16_len], dtype=np.uint8).copy()
    offset += fp16_len
    high_count = fp16_len // 2
    fp16_bytes = np.empty(fp16_len, dtype=np.uint8)
    fp16_bytes[1::2] = fp16_planed[:high_count]
    fp16_bytes[0::2] = fp16_planed[high_count:]
    fp16_pay = memoryview(fp16_bytes)
    fp16_offset = 0

    raw_offset = offset
    qrow_q = np.frombuffer(pay[raw_offset:raw_offset + n_q * cols], dtype=np.uint8).astype(np.float32).reshape(n_q, cols)
    raw_offset += n_q * cols
    mask_bytes = np.frombuffer(pay[raw_offset:raw_offset + mask_nbytes], dtype=np.uint8).copy()
    raw_offset += mask_nbytes
    dense_int_bytes: dict[str, np.ndarray] = {}
    for name, shape, numel in reversed(dense_int):
        nbytes = numel * 8
        dense_int_bytes[name] = np.frombuffer(pay[raw_offset:raw_offset + nbytes], dtype="<i8").copy().reshape(shape)
        raw_offset += nbytes
    codebook_arr = np.frombuffer(pay[raw_offset:raw_offset + 32], dtype="<f4").copy()
    codebook = torch.from_numpy(codebook_arr)
    raw_offset += 32
    if raw_offset != len(pay):
        raise ValueError("QFQ4 raw group length mismatch")

    bits = np.unpackbits(mask_bytes, bitorder="little")[:rows].astype(bool)
    if int(bits.sum()) != n_q:
        raise ValueError("QFQ4 qrow mask count mismatch")

    scale_bytes: dict[str, np.ndarray] = {}
    for name, _shape, _packed_len, scales_len in packed_order:
        scale_bytes[name] = np.frombuffer(fp16_pay[fp16_offset:fp16_offset + scales_len], dtype="<f2").copy()
        fp16_offset += scales_len

    dense_float_bytes: dict[str, np.ndarray] = {}
    for name, shape, numel in dense_float:
        nbytes = numel * 2
        dense_float_bytes[name] = np.frombuffer(fp16_pay[fp16_offset:fp16_offset + nbytes], dtype="<f2").copy().reshape(shape)
        fp16_offset += nbytes

    bias_bytes: dict[str, np.ndarray] = {}
    for name, n_bias in bias_modules:
        nbytes = n_bias * 2
        bias_bytes[name] = np.frombuffer(fp16_pay[fp16_offset:fp16_offset + nbytes], dtype="<f2").copy()
        fp16_offset += nbytes

    fp16_weight_bytes: dict[str, np.ndarray] = {}
    for name, shape, weight_numel in fp16_modules:
        nbytes = weight_numel * 2
        fp16_weight_bytes[name] = np.frombuffer(fp16_pay[fp16_offset:fp16_offset + nbytes], dtype="<f2").copy().reshape(shape)
        fp16_offset += nbytes

    qrow_min = np.frombuffer(fp16_pay[fp16_offset:fp16_offset + n_q * 2], dtype="<f2").astype(np.float32)
    fp16_offset += n_q * 2
    qrow_scale = np.frombuffer(fp16_pay[fp16_offset:fp16_offset + n_q * 2], dtype="<f2").astype(np.float32)
    fp16_offset += n_q * 2
    qrow_fp = np.frombuffer(fp16_pay[fp16_offset:fp16_offset + n_fp * cols * 2], dtype="<f2").copy().reshape(n_fp, cols)
    fp16_offset += n_fp * cols * 2
    if fp16_offset != len(fp16_pay):
        raise ValueError("QFQ4 fp16 group length mismatch")

    for name, shape, _packed_len, _scales_len in packed_modules:
        packed = packed_bytes[name]
        scales = scale_bytes[name]
        nib = _unpack_nibbles(packed, packed.shape[0] * 2)
        state_dict[f"{name}.weight"] = _dequant_fp4(nib, scales, shape, codebook)
    for name, arr in fp16_weight_bytes.items():
        state_dict[f"{name}.weight"] = torch.from_numpy(arr).float()
    for name, arr in bias_bytes.items():
        state_dict[f"{name}.bias"] = torch.from_numpy(arr).float()
    for name, arr in dense_float_bytes.items():
        state_dict[name] = torch.from_numpy(arr).float()
    for name, arr in dense_int_bytes.items():
        state_dict[name] = torch.from_numpy(arr)

    qrow = np.empty((rows, cols), dtype=np.float16)
    qrow[bits] = (qrow_q * qrow_scale[:, None] + qrow_min[:, None]).astype(np.float16)
    qrow[~bits] = qrow_fp
    state_dict[special] = torch.from_numpy(qrow).float()

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
