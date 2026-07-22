"""Fixed-schema loader for the deployed exact-integer entropy model."""

from __future__ import annotations

import numpy as np
import torch

from hpac_integer import IntegerConv2d, IntegerLinear


SELF_COMPRESSED_MAGIC = b"IHS1"
COMPRESSIBLE_TYPES = (IntegerConv2d, IntegerLinear)


def _unpack_nibbles(raw: memoryview, count: int):
    byte_count = (count + 1) // 2
    packed = np.frombuffer(raw[:byte_count], dtype=np.uint8)
    values = np.empty(byte_count * 2, dtype=np.uint8)
    values[0::2] = packed & 0xF
    values[1::2] = packed >> 4
    return values[:count].copy(), raw[byte_count:]


def _weight_rows(module, weight):
    if isinstance(module, IntegerConv2d):
        mask = module.mask.to(torch.bool).expand_as(weight)
        return [weight[index][mask[index]] for index in range(weight.shape[0])]
    return [weight[index].reshape(-1) for index in range(weight.shape[0])]


def _restore_weight_row(module, parameter, index, values):
    values = torch.from_numpy(values.astype(np.float32))
    if isinstance(module, IntegerConv2d):
        mask = module.mask.to(torch.bool).expand_as(parameter)[index]
        parameter[index].zero_()
        parameter[index][mask] = values
    else:
        parameter[index].copy_(values.reshape(parameter[index].shape))


def _deserialize_self_compressed(model, raw: bytes) -> None:
    view = memoryview(raw)[len(SELF_COMPRESSED_MAGIC):]
    modules = [
        module for module in model.modules()
        if isinstance(module, COMPRESSIBLE_TYPES)
    ]
    channel_count = sum(module.weight.shape[0] for module in modules)
    depths, view = _unpack_nibbles(view, channel_count)

    total_weight_bits = 0
    depth_offset = 0
    for module in modules:
        module_depths = depths[depth_offset:depth_offset + module.weight.shape[0]]
        row_counts = [row.numel() for row in _weight_rows(module, module.weight)]
        total_weight_bits += sum(
            int(bits) * count for bits, count in zip(module_depths, row_counts)
        )
        depth_offset += module.weight.shape[0]
    weight_bytes = (total_weight_bits + 7) // 8
    packed = np.frombuffer(view[:weight_bytes], dtype=np.uint8)
    weight_bits = np.unpackbits(packed, bitorder="little")[:total_weight_bits]
    view = view[weight_bytes:]

    bit_offset = 0
    depth_offset = 0
    with torch.no_grad():
        for module in modules:
            parameter = module.weight
            module_depths = depths[
                depth_offset:depth_offset + parameter.shape[0]
            ]
            for index, (bits, template) in enumerate(zip(
                module_depths, _weight_rows(module, parameter)
            )):
                count = template.numel()
                bits = int(bits)
                if bits:
                    count_bits = count * bits
                    rows = weight_bits[
                        bit_offset:bit_offset + count_bits
                    ].reshape(count, bits).astype(np.int16)
                    unsigned = (
                        rows * (1 << np.arange(bits, dtype=np.int16))
                    ).sum(axis=1, dtype=np.int16)
                    sign = 1 << (bits - 1)
                    values = np.where(
                        unsigned >= sign, unsigned - (1 << bits), unsigned
                    ).astype(np.int16)
                    bit_offset += count_bits
                else:
                    values = np.zeros(count, dtype=np.int16)
                _restore_weight_row(module, parameter, index, values)
            depth_offset += parameter.shape[0]

        module_by_name = dict(model.named_modules())
        for name, parameter in model.named_parameters():
            module_name, field = name.rsplit(".", 1)
            module = module_by_name[module_name]
            if field == "weight" and isinstance(module, COMPRESSIBLE_TYPES):
                continue
            dtype = np.dtype("<i2" if field == "bias" else "i1")
            byte_count = parameter.numel() * dtype.itemsize
            value = np.frombuffer(
                view[:byte_count], dtype=dtype, count=parameter.numel()
            ).copy().reshape(parameter.shape)
            parameter.copy_(torch.from_numpy(value.astype(np.float32)))
            view = view[byte_count:]
    if bit_offset != total_weight_bits or len(view):
        raise ValueError("self-compressed integer model has trailing data")


def deserialize_integer_model(model, raw: bytes) -> None:
    if raw.startswith(SELF_COMPRESSED_MAGIC):
        _deserialize_self_compressed(model, raw)
        return
    modules = dict(model.named_modules())
    offset = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            module_name, field = name.rsplit(".", 1)
            module = modules[module_name]
            masked_weight = field == "weight" and isinstance(module, IntegerConv2d)
            count = (
                int(module.mask.to(torch.bool).expand_as(parameter).sum().item())
                if masked_weight else parameter.numel()
            )
            dtype = np.dtype("<i2" if field == "bias" else "i1")
            byte_count = count * dtype.itemsize
            value = np.frombuffer(
                raw, dtype=dtype, count=count, offset=offset
            ).copy()
            if masked_weight:
                restored = torch.zeros_like(parameter)
                mask = module.mask.to(torch.bool).expand_as(parameter)
                restored[mask] = torch.from_numpy(value.astype(np.float32))
            else:
                restored = torch.from_numpy(
                    value.reshape(parameter.shape).astype(np.float32)
                )
            parameter.copy_(restored)
            offset += byte_count
    if offset != len(raw):
        raise ValueError(f"integer model has {len(raw) - offset} trailing bytes")
