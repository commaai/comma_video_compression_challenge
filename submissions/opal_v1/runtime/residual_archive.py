"""Parse the fixed F24S container inherited byte-for-byte from F26."""

from __future__ import annotations

import hashlib
import lzma
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bits import BitPackingError, packed_length, unpack_signed
from .carrier_repack import pack_frame0_selector_carrier
from .entropy.coefficient_ar1_codec import (
    CAP1_MAGIC,
    CoefficientAr1CodecError,
    decode_cap1,
)
from .entropy.rc64 import NativeDecoder
from .entropy.renderer_weight_codec import (
    RendererWeightCodecError,
    decode_f12_wans_body,
    decode_wans1,
)
from .frame0_selector import decode_selector
from .hpac_inference import configure_cuda_reproducibility, optimize_sparse_evaluator
from .ihs2 import materialize_ihs1

NUM_CLASSES = 5
FIXED_SCHEMA = "fixed_boundary_int6"
FIXED_MAGIC = b"RCF1"
FIXED_STATES = 25
FIXED_BITS = 6
F24S_MAGIC = b"F24S"
IHS2_HEADER = b"IHS2\x03\x31"
IHS2_BYTES = 16_599
IHS2_BODY_BYTES = IHS2_BYTES - len(IHS2_HEADER)
WANS_BODY_BYTES = 36_040
WANS_STREAM_ORDER = (1, 15, 4, 0, 11, 5, 2, 9, 3, 6, 10, 12, 14, 7, 8, 13)
CAP1_PREFIX = CAP1_MAGIC + bytes((1, 0, 0, 0))
SPARSE_SELECTOR_PREFIX = b"F0E1\x01"
CAP_FIELDS = ("predictor", "scales", "lengths", "ks", "basis", "rice")
STORED_CAP_FIELDS = ("scales", "predictor", "lengths", "ks", "basis", "rice")
LZMA_FILTERS = [
    {
        "id": lzma.FILTER_LZMA2,
        "dict_size": 1 << 16,
        "lc": 0,
        "lp": 1,
        "pb": 0,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 273,
        "mf": lzma.MF_BT4,
        "depth": 0,
    }
]


class ResidualArchiveError(ValueError):
    """The fixed F24S/OPAL archive is malformed or non-canonical."""


@dataclass(frozen=True)
class QuantizedTable:
    """The one fixed residual table carried by F26."""

    name: str
    bits: int
    codes: np.ndarray
    scale: float
    values: np.ndarray


@dataclass(frozen=True)
class ResidualArchiveParts:
    semantic_blob: bytes
    carrier_blob: bytes
    hpac_blob: bytes
    token_stream: bytes
    table: QuantizedTable
    schema: str
    residual_payload: bytes
    compressed_models: bytes
    token_codec: str = "rc64"


def _cap1_body_bytes(raw: bytes) -> int:
    if len(raw) < 6:
        raise ResidualArchiveError("truncated CAP1 bit counts")
    basis_bits = int.from_bytes(raw[:3], "little")
    residual_bits = int.from_bytes(raw[3:6], "little")
    if not basis_bits or not residual_bits:
        raise ResidualArchiveError("invalid CAP1 bit counts")
    return (
        6
        + 36
        + 96
        + 32
        + 12
        + (basis_bits + 7) // 8
        + (residual_bits + 7) // 8
    )


def _restore_cap1(raw: bytes) -> bytes:
    size = _cap1_body_bytes(raw)
    if len(raw) != size:
        raise ResidualArchiveError("invalid stored CAP1 body length")
    basis_bits = int.from_bytes(raw[:3], "little")
    residual_bits = int.from_bytes(raw[3:6], "little")
    sizes = {
        "predictor": 36,
        "scales": 96,
        "lengths": 32,
        "ks": 12,
        "basis": (basis_bits + 7) // 8,
        "rice": (residual_bits + 7) // 8,
    }
    fields = {}
    offset = 6
    for name in STORED_CAP_FIELDS:
        end = offset + sizes[name]
        fields[name] = raw[offset:end]
        offset = end
    if offset != len(raw):
        raise ResidualArchiveError("stored CAP1 field accounting mismatch")
    return CAP1_PREFIX + raw[:6] + b"".join(fields[name] for name in CAP_FIELDS)


def _decode_models(models: bytes) -> tuple[bytes, bytes, bytes]:
    if not models.startswith(F24S_MAGIC):
        raise ResidualArchiveError("F26 requires the fixed F24S model container")
    minimum = 4 + IHS2_BODY_BYTES + WANS_BODY_BYTES + 7
    if len(models) < minimum:
        raise ResidualArchiveError("truncated F24S model payload")
    offset = 4
    hpac = IHS2_HEADER + models[offset : offset + IHS2_BODY_BYTES]
    offset += IHS2_BODY_BYTES
    semantic_end = offset + WANS_BODY_BYTES
    try:
        semantic = decode_f12_wans_body(
            models[offset:semantic_end],
            WANS_STREAM_ORDER,
        )
        decode_wans1(semantic)
    except RendererWeightCodecError as error:
        raise ResidualArchiveError(f"invalid F24S renderer weights: {error}") from error
    offset = semantic_end
    cap1_bytes = _cap1_body_bytes(models[offset:])
    cap1_end = offset + cap1_bytes
    if cap1_end >= len(models):
        raise ResidualArchiveError("truncated F24S carrier or selector")
    cap1 = _restore_cap1(models[offset:cap1_end])
    selector = SPARSE_SELECTOR_PREFIX + models[cap1_end:]
    try:
        decode_cap1(cap1, frames=600, dimensions=12)
        decode_selector(selector)
        carrier = pack_frame0_selector_carrier(cap1, selector)
    except (CoefficientAr1CodecError, ValueError) as error:
        raise ResidualArchiveError(f"invalid F24S carrier: {error}") from error
    return semantic, carrier, hpac


def _decode_fixed_table(raw: bytes) -> QuantizedTable:
    expected = len(FIXED_MAGIC) + 2 + packed_length(
        FIXED_STATES * NUM_CLASSES,
        FIXED_BITS,
    )
    if len(raw) != expected or not raw.startswith(FIXED_MAGIC):
        raise ResidualArchiveError("invalid fixed residual table")
    scale = float(np.frombuffer(raw[4:6], dtype="<f2")[0])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ResidualArchiveError("invalid fixed residual scale")
    try:
        codes = np.asarray(
            unpack_signed(raw[6:], FIXED_STATES * NUM_CLASSES, FIXED_BITS),
            dtype=np.int8,
        ).reshape(FIXED_STATES, NUM_CLASSES)
    except BitPackingError as error:
        raise ResidualArchiveError(f"invalid fixed residual codes: {error}") from error
    return QuantizedTable(
        name="boundary_predicted",
        bits=FIXED_BITS,
        codes=codes,
        scale=scale,
        values=codes.astype(np.float32) * scale,
    )


def read_residual_archive(archive_path: Path) -> ResidualArchiveParts:
    """Strictly parse the single F24S representation accepted by F26."""
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"]:
            raise ResidualArchiveError("archive must contain exactly member p")
        outer = archive.read("p")
    try:
        decoder = lzma.LZMADecompressor(
            format=lzma.FORMAT_RAW,
            filters=LZMA_FILTERS,
        )
        models = decoder.decompress(outer)
    except lzma.LZMAError as error:
        raise ResidualArchiveError("invalid raw-LZMA model section") from error
    section = decoder.unused_data
    if not decoder.eof or not section:
        raise ResidualArchiveError("truncated F24S model section")
    compressed = outer[: len(outer) - len(section)]
    semantic, carrier, hpac = _decode_models(models)

    fixed_size = len(FIXED_MAGIC) + 2 + packed_length(
        FIXED_STATES * NUM_CLASSES,
        FIXED_BITS,
    )
    compact_size = fixed_size - len(FIXED_MAGIC)
    if len(section) <= compact_size:
        raise ResidualArchiveError("truncated F24S residual or token section")
    residual = FIXED_MAGIC + section[:compact_size]
    table = _decode_fixed_table(residual)
    tokens = section[compact_size:]
    if not tokens:
        raise ResidualArchiveError("empty F24S RC64 token stream")
    return ResidualArchiveParts(
        semantic_blob=semantic,
        carrier_blob=carrier,
        hpac_blob=hpac,
        token_stream=tokens,
        table=table,
        schema=FIXED_SCHEMA,
        residual_payload=residual,
        compressed_models=compressed,
    )


def _sparse_class(code_dir: Path):
    import sys

    sys.path.insert(0, str(code_dir))
    try:
        from hpac_integer_sparse import SparseIntegerHPAC

        return SparseIntegerHPAC
    finally:
        sys.path.pop(0)


def _boundary_buckets(previous: np.ndarray, max_distance: int = 4) -> np.ndarray:
    if previous.ndim != 2:
        raise ResidualArchiveError("boundary source must be one token frame")
    edge = np.zeros(previous.shape, dtype=bool)
    edge[1:] |= previous[1:] != previous[:-1]
    edge[:-1] |= previous[:-1] != previous[1:]
    edge[:, 1:] |= previous[:, 1:] != previous[:, :-1]
    edge[:, :-1] |= previous[:, :-1] != previous[:, 1:]
    result = np.full(previous.shape, max_distance, dtype=np.uint8)
    active = edge.copy()
    result[active] = 0
    for distance in range(1, max_distance):
        grown = active.copy()
        grown[1:] |= active[:-1]
        grown[:-1] |= active[1:]
        grown[:, 1:] |= active[:, :-1]
        grown[:, :-1] |= active[:, 1:]
        active = grown
        result[(result == max_distance) & active] = distance
    return result


def _probability_table(logits: np.ndarray, precision: int) -> np.ndarray:
    quantized = np.clip(
        np.rint(np.asarray(logits, dtype=np.float32) * precision),
        -32768,
        32767,
    ).astype(np.int16)
    values = quantized.astype(np.float32) / precision
    values = values.astype(np.float64)
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def decode_production_tokens(
    parts: ResidualArchiveParts,
    runtime,
    code_dir: Path,
    device,
) -> tuple[object, dict[str, object]]:
    """Decode the OPAL-adapted RC64 stream around F26's fixed probability model."""
    import torch

    started = time.time()
    configure_cuda_reproducibility()
    base_hpac = materialize_ihs1(parts.hpac_blob, runtime)
    model = runtime.load_hpac(base_hpac, device)
    masks = runtime.group_masks(device)
    sparse = _sparse_class(code_dir)(model, runtime.EVAL_H, runtime.EVAL_W)
    library = os.environ.get("CPR1_RC64_LIBRARY")
    if not library:
        raise ResidualArchiveError("RC64 decoding requires CPR1_RC64_LIBRARY")
    decoder = NativeDecoder(Path(library), parts.token_stream)

    group_plans = []
    for mask in masks:
        mask_array = mask.detach().cpu().numpy()
        flat_positions = np.flatnonzero(mask_array.reshape(-1))
        group_plans.append(
            (
                torch.from_numpy(flat_positions).to(device),
                flat_positions,
            )
        )

    corrected_digest = hashlib.sha256()
    cdf_digest = hashlib.sha256()
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        previous = torch.zeros(
            (1, runtime.EVAL_H, runtime.EVAL_W),
            dtype=torch.long,
            device=device,
        )
        tokens = torch.empty(
            (runtime.N, runtime.EVAL_H, runtime.EVAL_W),
            dtype=torch.uint8,
        )
        for frame in range(runtime.N):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(
                    device="cpu", dtype=torch.uint8
                ).numpy()
                boundary = _boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(
                    runtime.EVAL_H * runtime.EVAL_W,
                    4,
                    dtype=np.uint8,
                )
            for group, (device_positions, flat_positions) in enumerate(group_plans):
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = (
                    boundary[flat_positions].astype(np.int64) * NUM_CLASSES
                    + predicted
                )
                corrected = base_logits + parts.table.values[feature]
                corrected_digest.update(
                    np.ascontiguousarray(corrected, dtype="<f4").tobytes()
                )
                probability = _probability_table(
                    corrected,
                    runtime.HPAC_LOGIT_PRECISION,
                )
                cdf_digest.update(
                    np.ascontiguousarray(probability, dtype="<f4").tobytes()
                )
                symbols = decoder.decode(probability).astype(np.int64)
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(
                    device
                )
            tokens[frame] = current[0].to(device="cpu", dtype=torch.uint8)
            previous = current
    elapsed = time.time() - started
    del model
    return tokens, {
        "corrected_quantized_logit_sha256": corrected_digest.hexdigest(),
        "corrected_cdf_input_sha256": cdf_digest.hexdigest(),
        "decoded_token_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        "decode_runtime_seconds": elapsed,
        "adapter_sha256": "",
        "token_codec": "rc64",
        "decoder_bit_position": decoder.bit_position,
    }
