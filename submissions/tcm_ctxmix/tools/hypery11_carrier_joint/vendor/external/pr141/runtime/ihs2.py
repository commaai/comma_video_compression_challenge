"""Exact fixed-schema IHS2 representation for the deployed CPR1 IntegerHPAC.

IHS2 is a storage representation, not a new model.  It reconstructs the
byte-identical deployed IHS1 blob before the pinned CPR1 loader is called.
The only constants in this module describe the already-deployed architecture;
all model values, original depths, and reduced depths are carried in IHS2.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    """One fixed field in the deployed integer HPAC state."""

    name: str
    count: int
    dtype: str
    shape: tuple[int, ...]

    @property
    def itemsize(self) -> int:
        return np.dtype(self.dtype).itemsize

    @property
    def byte_count(self) -> int:
        return self.count * self.itemsize

    @property
    def is_bias(self) -> bool:
        return self.name.endswith(".bias")

    @property
    def is_exponent(self) -> bool:
        return self.name.endswith(".exponent")

IHS1_MAGIC = b"IHS1"
IHS2_MAGIC = b"IHS2"
# The three low bits select the frame field.  Zero intentionally means the
# original raw int8 frame field, so isolated exponent/depth candidates need no
# special side format.
FRAME_RAW = 0
FRAME_E0L0 = 1
FRAME_E0L1 = 2
FRAME_E1L0 = 3
FRAME_E1L1 = 4
FRAME_E2L0 = 5
FRAME_E2L1 = 6
FRAME_FORMATS = {
    FRAME_RAW: "raw_i8",
    FRAME_E0L0: "E0L0_twos_complement_frame_major",
    FRAME_E0L1: "E0L1_twos_complement_dimension_major",
    FRAME_E1L0: "E1L0_biased_frame_major",
    FRAME_E1L1: "E1L1_biased_dimension_major",
    FRAME_E2L0: "E2L0_zigzag_frame_major",
    FRAME_E2L1: "E2L1_zigzag_dimension_major",
}

FLAG_EXPONENTS_3BIT = 1 << 3
FLAG_TIGHT_ROWS = 1 << 4


class IHS2Error(ValueError):
    """An IHS1/IHS2 representation is invalid or cannot be reconstructed."""


@dataclass(frozen=True)
class IHS2Layout:
    """Known CPR1 HPAC structure; none of these entries are learned values."""

    row_counts: tuple[int, ...]
    module_ranges: tuple[tuple[str, int, int], ...]
    tail_fields: tuple[FieldSpec, ...]

    @property
    def depth_count(self) -> int:
        return len(self.row_counts)

    @property
    def depth_bytes(self) -> int:
        return (self.depth_count + 1) // 2

    @property
    def frame_field(self) -> FieldSpec:
        fields = [
            field for field in self.tail_fields if field.name == "frame_embed.weight"
        ]
        if len(fields) != 1 or fields[0].count != 600 * 8 or fields[0].dtype != "i1":
            raise IHS2Error("IHS2 requires one 600x8 int8 frame embedding")
        return fields[0]

    @property
    def exponent_count(self) -> int:
        return sum(field.count for field in self.tail_fields if field.is_exponent)

    @property
    def non_exponent_tail_bytes(self) -> int:
        return sum(
            field.byte_count
            for field in self.tail_fields
            if not field.is_exponent and field.name != "frame_embed.weight"
        )

    @property
    def raw_tail_bytes(self) -> int:
        return sum(field.byte_count for field in self.tail_fields)


def layout_from_model(model) -> IHS2Layout:
    """Derive the fixed deployed layout from an unmodified CPR1 model object."""
    integer = importlib.import_module("hpac_integer")
    compressed_types = (integer.IntegerConv2d, integer.IntegerLinear)
    modules = dict(model.named_modules())
    row_counts: list[int] = []
    module_ranges: list[tuple[str, int, int]] = []
    for name, module in model.named_modules():
        if not isinstance(module, compressed_types):
            continue
        start = len(row_counts)
        if isinstance(module, integer.IntegerConv2d):
            mask = module.mask.to(bool).expand_as(module.weight)
            row_counts.extend(
                int(mask[index].sum().item()) for index in range(module.weight.shape[0])
            )
        else:
            row_counts.extend(
                int(module.weight[index].numel())
                for index in range(module.weight.shape[0])
            )
        module_ranges.append((name, start, len(row_counts)))
    fields: list[FieldSpec] = []
    for name, parameter in model.named_parameters():
        module_name, field = name.rsplit(".", 1)
        module = modules[module_name]
        if field == "weight" and isinstance(module, compressed_types):
            continue
        fields.append(
            FieldSpec(
                name,
                parameter.numel(),
                "<i2" if field == "bias" else "i1",
                tuple(parameter.shape),
            )
        )
    layout = IHS2Layout(tuple(row_counts), tuple(module_ranges), tuple(fields))
    # Fail before writing an artifact if the pinned architecture changed.
    if layout.depth_count != 517 or layout.exponent_count != 517:
        raise IHS2Error("unexpected CPR1 HPAC structural schema")
    if layout.frame_field.count != 4_800:
        raise IHS2Error("unexpected CPR1 HPAC frame embedding")
    return layout


def layout_from_runtime(runtime) -> IHS2Layout:
    """Build a value-free CPR1 model shell and inspect its pinned structure."""
    model = runtime.IntegerHPAC(
        num_pairs=runtime.N,
        num_classes=runtime.NUM_CLASSES,
        patch=runtime.HPAC_PATCH,
        delta=runtime.HPAC_DELTA,
        channels=runtime.HPAC_CHANNELS,
        frame_dim=runtime.HPAC_FILM_DIM,
        norm_mode="none",
        activation="relu",
        use_frame_scale=True,
        weight_bound=127,
        activation_bound=127,
        use_weight_scales=True,
        weight_exponent_min=-6,
        use_spm=True,
        use_norm_gates=False,
    ).eval()
    return layout_from_model(model)


def _pack_nibbles(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    if np.any(values > 15):
        raise IHS2Error("nibble value outside 0..15")
    padded = np.pad(values, (0, values.size % 2), constant_values=0)
    return (padded[0::2] | (padded[1::2] << 4)).tobytes()


def _unpack_nibbles(raw: bytes, count: int) -> np.ndarray:
    if len(raw) != (count + 1) // 2:
        raise IHS2Error("truncated nibble field")
    if count % 2 and raw[-1] >> 4:
        raise IHS2Error("non-zero odd-nibble padding")
    source = np.frombuffer(raw, dtype=np.uint8)
    values = np.empty(source.size * 2, dtype=np.uint8)
    values[0::2] = source & 0x0F
    values[1::2] = source >> 4
    return values[:count]


def _unpack_unsigned(raw: bytes, count: int, bits: int) -> np.ndarray:
    expected = (count * bits + 7) // 8
    if len(raw) != expected:
        raise IHS2Error("truncated packed integer field")
    if count * bits % 8 and raw[-1] >> (count * bits % 8):
        raise IHS2Error("non-zero packed-field padding")
    values = np.empty(count, dtype=np.int16)
    for index in range(count):
        offset = index * bits
        byte, shift = divmod(offset, 8)
        word = raw[byte]
        if byte + 1 < len(raw):
            word |= raw[byte + 1] << 8
        if byte + 2 < len(raw):
            word |= raw[byte + 2] << 16
        values[index] = (word >> shift) & ((1 << bits) - 1)
    return values


def _read_signed_rows(
    raw: bytes, depths: np.ndarray, layout: IHS2Layout
) -> tuple[np.ndarray, ...]:
    total_bits = int(
        sum(
            int(depth) * count
            for depth, count in zip(depths, layout.row_counts, strict=True)
        )
    )
    expected = (total_bits + 7) // 8
    if len(raw) != expected:
        raise IHS2Error("invalid IHS1 weight bitstream length")
    if total_bits % 8 and raw[-1] >> (total_bits % 8):
        raise IHS2Error("IHS1 non-zero weight padding")
    rows: list[np.ndarray] = []
    offset = 0
    for depth, count in zip(depths, layout.row_counts, strict=True):
        depth = int(depth)
        if depth == 0:
            rows.append(np.zeros(count, dtype=np.int16))
            continue
        values = np.empty(count, dtype=np.int16)
        sign = 1 << (depth - 1)
        for index in range(count):
            byte, shift = divmod(offset, 8)
            word = raw[byte]
            if byte + 1 < len(raw):
                word |= raw[byte + 1] << 8
            if byte + 2 < len(raw):
                word |= raw[byte + 2] << 16
            unsigned = (word >> shift) & ((1 << depth) - 1)
            values[index] = unsigned - (1 << depth) if unsigned & sign else unsigned
            offset += depth
        rows.append(values)
    if offset != total_bits:
        raise IHS2Error("IHS1 row bit accounting mismatch")
    return tuple(rows)


def _write_signed_rows(
    rows: tuple[np.ndarray, ...], depths: np.ndarray, layout: IHS2Layout
) -> bytes:
    if len(rows) != layout.depth_count or len(depths) != layout.depth_count:
        raise IHS2Error("row/depth count mismatch")
    total_bits = int(
        sum(
            int(depth) * count
            for depth, count in zip(depths, layout.row_counts, strict=True)
        )
    )
    output = bytearray((total_bits + 7) // 8)
    offset = 0
    for row, depth, count in zip(rows, depths, layout.row_counts, strict=True):
        values = np.asarray(row, dtype=np.int16).reshape(-1)
        depth = int(depth)
        if values.size != count:
            raise IHS2Error("row parameter count mismatch")
        if depth == 0:
            if np.any(values):
                raise IHS2Error("zero-bit row contains a non-zero value")
            continue
        minimum, maximum = -(1 << (depth - 1)), (1 << (depth - 1)) - 1
        if np.any(values < minimum) or np.any(values > maximum):
            raise IHS2Error("row value outside declared signed depth")
        for value in values:
            byte, shift = divmod(offset, 8)
            word = int(value) & ((1 << depth) - 1)
            packed = word << shift
            output[byte] |= packed & 0xFF
            if shift + depth > 8:
                output[byte + 1] |= (packed >> 8) & 0xFF
            if shift + depth > 16:
                output[byte + 2] |= (packed >> 16) & 0xFF
            offset += depth
    if offset != total_bits:
        raise IHS2Error("row packing bit accounting mismatch")
    return bytes(output)


def minimum_signed_depth(values: np.ndarray) -> int:
    """Canonical exact signed width, with an explicit zero-bit all-zero row."""
    values = np.asarray(values, dtype=np.int16).reshape(-1)
    if not values.size or not np.any(values):
        return 0
    minimum, maximum = int(values.min()), int(values.max())
    for bits in range(1, 16):
        if minimum >= -(1 << (bits - 1)) and maximum <= (1 << (bits - 1)) - 1:
            return bits
    raise IHS2Error("row cannot be represented in deployed signed depth range")


def _decode_frame(raw: bytes, frame_format: int) -> bytes:
    if frame_format == FRAME_RAW:
        if len(raw) != 600 * 8:
            raise IHS2Error("invalid raw frame embedding length")
        return raw
    if frame_format not in FRAME_FORMATS:
        raise IHS2Error("invalid frame format")
    codes = _unpack_nibbles(raw, 600 * 8).astype(np.int16)
    if frame_format in (FRAME_E0L0, FRAME_E0L1):
        values = np.where(codes >= 8, codes - 16, codes)
    elif frame_format in (FRAME_E1L0, FRAME_E1L1):
        values = codes - 8
    else:
        values = np.where(codes & 1, -(codes // 2) - 1, codes // 2)
    if np.any(values < -8) or np.any(values > 7):
        raise IHS2Error("invalid decoded int4 frame code")
    matrix = (
        values.reshape(8, 600).T
        if frame_format in (FRAME_E0L1, FRAME_E1L1, FRAME_E2L1)
        else values.reshape(600, 8)
    )
    return matrix.astype(np.int8).tobytes()


def decode_ihs2(blob: bytes, layout: IHS2Layout) -> bytes:
    """Decode the IHS2-v3 representation used by the promoted archive."""
    if not blob.startswith(IHS2_MAGIC + b"\x03"):
        raise IHS2Error("F26 requires an IHS2-v3 model")
    from .ihs2_gate_a import decode_v3

    return decode_v3(blob, layout)


def materialize_ihs1(blob: bytes, runtime) -> bytes:
    """Select the unambiguous stored representation for the production path."""
    if blob.startswith(IHS1_MAGIC):
        return blob
    if blob.startswith(IHS2_MAGIC):
        return decode_ihs2(blob, layout_from_runtime(runtime))
    raise IHS2Error("unknown HPAC representation magic")
