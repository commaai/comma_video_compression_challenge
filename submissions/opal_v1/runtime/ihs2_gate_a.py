"""Decode the IHS2-v3 representation used by F26."""

from __future__ import annotations

import numpy as np

from .ihs2 import (
    FLAG_EXPONENTS_3BIT,
    FLAG_TIGHT_ROWS,
    FRAME_FORMATS,
    FRAME_RAW,
    IHS1_MAGIC,
    IHS2_MAGIC,
    IHS2Error,
    IHS2Layout,
    _decode_frame,
    _pack_nibbles,
    _read_signed_rows,
    _unpack_nibbles,
    _unpack_unsigned,
    _write_signed_rows,
    minimum_signed_depth,
)

VERSION = 3
FLAG_PACK_BIASES = 1 << 5
_KNOWN_FLAGS = 0x3F


def _bias_fields(layout: IHS2Layout):
    return tuple(field for field in layout.tail_fields if field.is_bias)


def _other_fields(layout: IHS2Layout):
    return tuple(
        field
        for field in layout.tail_fields
        if field.name != "frame_embed.weight"
        and not field.is_exponent
        and not field.is_bias
    )


def _pack_biases(raw_fields: tuple[bytes, ...], layout: IHS2Layout) -> tuple[bytes, bytes]:
    fields = _bias_fields(layout)
    if len(raw_fields) != len(fields):
        raise IHS2Error("IHS2 v3 bias field count mismatch")
    widths = []
    for field, raw in zip(fields, raw_fields, strict=True):
        if len(raw) != field.byte_count:
            raise IHS2Error("IHS2 v3 bias byte count mismatch")
        values = np.frombuffer(raw, dtype="<i2")
        width = 0 if not np.any(values) else minimum_signed_depth(values)
        if width > 15:
            raise IHS2Error("IHS2 v3 bias exceeds its signed-width field")
        widths.append(width)

    bit_count = sum(
        field.count * width for field, width in zip(fields, widths, strict=True)
    )
    output = bytearray((bit_count + 7) // 8)
    offset = 0
    for field, raw, width in zip(fields, raw_fields, widths, strict=True):
        values = np.frombuffer(raw, dtype="<i2")
        if width == 0:
            if np.any(values):
                raise IHS2Error("implicit zero bias contains a non-zero value")
            continue
        minimum = -(1 << (width - 1))
        maximum = (1 << (width - 1)) - 1
        if np.any(values < minimum) or np.any(values > maximum):
            raise IHS2Error("bias value is outside its stored signed width")
        for value in values:
            byte, shift = divmod(offset, 8)
            packed = (int(value) & ((1 << width) - 1)) << shift
            output[byte] |= packed & 0xFF
            if shift + width > 8:
                output[byte + 1] |= (packed >> 8) & 0xFF
            if shift + width > 16:
                output[byte + 2] |= (packed >> 16) & 0xFF
            offset += width
    return _pack_nibbles(np.asarray(widths, dtype=np.uint8)), bytes(output)


def _unpack_biases(
    widths_raw: bytes, packed: bytes, layout: IHS2Layout
) -> tuple[bytes, ...]:
    fields = _bias_fields(layout)
    widths = _unpack_nibbles(widths_raw, len(fields))
    if np.any(widths > 15):
        raise IHS2Error("IHS2 v3 bias width is outside 0..15")
    bit_count = sum(
        field.count * int(width)
        for field, width in zip(fields, widths, strict=True)
    )
    if len(packed) != (bit_count + 7) // 8:
        raise IHS2Error("truncated IHS2 v3 bias stream")
    if bit_count % 8 and packed[-1] >> (bit_count % 8):
        raise IHS2Error("non-zero IHS2 v3 bias padding")

    decoded = []
    offset = 0
    for field, width_value in zip(fields, widths, strict=True):
        width = int(width_value)
        if width == 0:
            decoded.append(np.zeros(field.count, dtype="<i2").tobytes())
            continue
        values = np.empty(field.count, dtype=np.int16)
        sign = 1 << (width - 1)
        for index in range(field.count):
            byte, shift = divmod(offset, 8)
            word = packed[byte]
            if byte + 1 < len(packed):
                word |= packed[byte + 1] << 8
            if byte + 2 < len(packed):
                word |= packed[byte + 2] << 16
            code = (word >> shift) & ((1 << width) - 1)
            values[index] = code - (1 << width) if code & sign else code
            offset += width
        decoded.append(values.astype("<i2").tobytes())

    canonical_widths, canonical_payload = _pack_biases(tuple(decoded), layout)
    if canonical_widths != widths_raw or canonical_payload != packed:
        raise IHS2Error("non-canonical IHS2 v3 bias metadata")
    return tuple(decoded)


def _join_non_exponent(
    biases: tuple[bytes, ...], other: tuple[bytes, ...], layout: IHS2Layout
) -> bytes:
    output = bytearray()
    bias_index = 0
    other_index = 0
    for field in layout.tail_fields:
        if field.name == "frame_embed.weight" or field.is_exponent:
            continue
        if field.is_bias:
            output.extend(biases[bias_index])
            bias_index += 1
        else:
            output.extend(other[other_index])
            other_index += 1
    if bias_index != len(biases) or other_index != len(other):
        raise IHS2Error("IHS2 v3 tail field accounting mismatch")
    return bytes(output)


def decode_v3(blob: bytes, layout: IHS2Layout) -> bytes:
    """Restore the canonical IHS1 bytes from a strict IHS2-v3 payload."""
    if len(blob) < 6 or blob[:4] != IHS2_MAGIC or blob[4] != VERSION:
        raise IHS2Error("invalid IHS2 v3 header")
    flags = blob[5]
    if flags & ~_KNOWN_FLAGS:
        raise IHS2Error("IHS2 v3 reserved flags are non-zero")
    frame_format = flags & 7
    if frame_format not in FRAME_FORMATS:
        raise IHS2Error("IHS2 v3 reserved frame format")
    pack_exponents = bool(flags & FLAG_EXPONENTS_3BIT)
    tighten_rows = bool(flags & FLAG_TIGHT_ROWS)
    pack_biases = bool(flags & FLAG_PACK_BIASES)

    offset = 6
    end = offset + layout.depth_bytes
    if len(blob) < end:
        raise IHS2Error("truncated IHS2 v3 original depths")
    original_depths = _unpack_nibbles(blob[offset:end], layout.depth_count)
    offset = end
    if tighten_rows:
        end = offset + layout.depth_bytes
        if len(blob) < end:
            raise IHS2Error("truncated IHS2 v3 tightened depths")
        stored_depths = _unpack_nibbles(blob[offset:end], layout.depth_count)
        offset = end
        if np.any(stored_depths > original_depths):
            raise IHS2Error("IHS2 v3 tightened depth exceeds original depth")
    else:
        stored_depths = original_depths

    weight_bits = sum(
        int(depth) * count
        for depth, count in zip(stored_depths, layout.row_counts, strict=True)
    )
    weight_bytes = (weight_bits + 7) // 8
    end = offset + weight_bytes
    if len(blob) < end:
        raise IHS2Error("truncated IHS2 v3 weights")
    rows = _read_signed_rows(blob[offset:end], stored_depths, layout)
    offset = end
    if tighten_rows:
        canonical_depths = np.asarray(
            [minimum_signed_depth(row) for row in rows], dtype=np.uint8
        )
        if not np.array_equal(canonical_depths, stored_depths):
            raise IHS2Error("non-canonical IHS2 v3 tightened depth")

    frame_bytes = 600 * 8 if frame_format == FRAME_RAW else 600 * 8 // 2
    end = offset + frame_bytes
    if len(blob) < end:
        raise IHS2Error("truncated IHS2 v3 frame field")
    frame = _decode_frame(blob[offset:end], frame_format)
    offset = end

    bias_fields = _bias_fields(layout)
    if pack_biases:
        width_bytes = (len(bias_fields) + 1) // 2
        end = offset + width_bytes
        if len(blob) < end:
            raise IHS2Error("truncated IHS2 v3 bias widths")
        widths_raw = blob[offset:end]
        offset = end
        widths = _unpack_nibbles(widths_raw, len(bias_fields))
        bias_bits = sum(
            field.count * int(width)
            for field, width in zip(bias_fields, widths, strict=True)
        )
        bias_bytes = (bias_bits + 7) // 8
        end = offset + bias_bytes
        if len(blob) < end:
            raise IHS2Error("truncated IHS2 v3 packed biases")
        bias_values = _unpack_biases(widths_raw, blob[offset:end], layout)
        offset = end
    else:
        bias_bytes = sum(field.byte_count for field in bias_fields)
        end = offset + bias_bytes
        if len(blob) < end:
            raise IHS2Error("truncated IHS2 v3 raw biases")
        bias_values = []
        cursor = offset
        for field in bias_fields:
            bias_values.append(blob[cursor : cursor + field.byte_count])
            cursor += field.byte_count
        bias_values = tuple(bias_values)
        offset = end

    other_fields = _other_fields(layout)
    other_bytes = sum(field.byte_count for field in other_fields)
    end = offset + other_bytes
    if len(blob) < end:
        raise IHS2Error("truncated IHS2 v3 raw tail")
    other_values = []
    cursor = offset
    for field in other_fields:
        other_values.append(blob[cursor : cursor + field.byte_count])
        cursor += field.byte_count
    offset = end

    exponent_bytes = (
        (layout.exponent_count * 3 + 7) // 8
        if pack_exponents
        else layout.exponent_count
    )
    if len(blob) != offset + exponent_bytes:
        raise IHS2Error("IHS2 v3 truncated or trailing exponent section")
    if pack_exponents:
        codes = _unpack_unsigned(blob[offset:], layout.exponent_count, 3)
        if np.any(codes == 7):
            raise IHS2Error("IHS2 reserved exponent code 7")
        exponents = (codes - 6).astype(np.int8)
    else:
        exponents = np.frombuffer(blob[offset:], dtype=np.int8).copy()
    if np.any(exponents < -6) or np.any(exponents > 0):
        raise IHS2Error("IHS2 v3 exponent outside exact alphabet")

    non_exponent = _join_non_exponent(
        tuple(bias_values), tuple(other_values), layout
    )
    output = bytearray(IHS1_MAGIC)
    output.extend(_pack_nibbles(original_depths))
    output.extend(_write_signed_rows(rows, original_depths, layout))
    non_exponent_offset = 0
    exponent_offset = 0
    for field in layout.tail_fields:
        if field.name == "frame_embed.weight":
            output.extend(frame)
        elif field.is_exponent:
            end = exponent_offset + field.count
            output.extend(exponents[exponent_offset:end].astype(np.int8).tobytes())
            exponent_offset = end
        else:
            end = non_exponent_offset + field.byte_count
            output.extend(non_exponent[non_exponent_offset:end])
            non_exponent_offset = end
    if (
        non_exponent_offset != len(non_exponent)
        or exponent_offset != layout.exponent_count
    ):
        raise IHS2Error("IHS2 v3 tail reconstruction accounting mismatch")
    return bytes(output)
