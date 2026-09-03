"""Parse and decode the fixed F24S archive format used by F26."""

from __future__ import annotations

import hashlib
import lzma
import os
import struct
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import brotli
except ImportError:  # Local parse-back may select the pinned CLI instead.
    brotli = None

from .bits import BitPackingError, packed_length, unpack_signed
from .carrier_repack import pack_frame0_selector_carrier
from .compensation_overlay import (
    MAGIC as COMPENSATION_MAGIC,
    split_selector_compensation,
)
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
SPLIT_MODEL_HEADER = struct.Struct("<HHH")
HP4_MODEL_HEADER = struct.Struct("<4sBBBBHHHHH")
HP4_MAGIC = b"HP4M"
# DDM_RX1_IHS1_CONTAINER_V1
RX1_MAGIC = b"RX1M"
RX1_MODEL_HEADER = struct.Struct("<4sBBBBHHH")
RX1_CODEC_XZ = 1
RX1_CODEC_BROTLI = 2
HP4_FRAME_OFFSET_BODY = 13_373
HP4_FRAME_BYTES = 2_400
SPLIT_MODEL_SECTION_BYTES = (16_593, 36_040)
PACKED_CAP1_SECTION_BYTES = 22_183
CANONICAL_CAP1_SECTION_BYTES = 22_223


class ResidualArchiveError(ValueError):
    """The promoted F26 archive is malformed or non-canonical."""


def _decompress_brotli(stream: bytes) -> bytes:
    if brotli is not None:
        try:
            return brotli.decompress(stream)
        except brotli.error as error:
            raise ResidualArchiveError("invalid split Brotli model stream") from error
    binary = os.environ.get("CP135_BROTLI_CLI", "")
    if not binary:
        raise ResidualArchiveError(
            "split Brotli model requires Brotli==1.2.0 or CP135_BROTLI_CLI"
        )
    completed = subprocess.run(
        [binary, "-d", "-c"],
        input=stream,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise ResidualArchiveError("split Brotli CLI rejected a model stream")
    return completed.stdout


def _unpack_unsigned(raw: bytes, count: int, bits: int) -> np.ndarray:
    if len(raw) != (count * bits + 7) // 8:
        raise ResidualArchiveError("packed CAP1 field has the wrong length")
    if count * bits % 8 and raw[-1] >> (count * bits % 8):
        raise ResidualArchiveError("packed CAP1 field has nonzero padding")
    output = np.empty(count, dtype=np.int16)
    for index in range(count):
        offset = index * bits
        byte, shift = divmod(offset, 8)
        word = raw[byte]
        if byte + 1 < len(raw):
            word |= raw[byte + 1] << 8
        output[index] = (word >> shift) & ((1 << bits) - 1)
    return output


def _restore_packed_cap1_metadata(packed: bytes) -> bytes:
    if len(packed) < 142:
        raise ResidualArchiveError("packed CAP1 section is truncated")
    factor_base = packed[102]
    factors = factor_base + _unpack_unsigned(packed[103:114], 12, 7)
    bias_codes = _unpack_unsigned(packed[114:123], 12, 6)
    biases = np.where(bias_codes >= 32, bias_codes - 64, bias_codes).astype(np.int8)
    lengths = _unpack_unsigned(packed[123:139], 32, 4).astype(np.uint8)
    k_base = packed[139]
    ks = (k_base + _unpack_unsigned(packed[140:142], 12, 1)).astype(np.uint8)
    if (
        np.any(factors > 512)
        or np.any(biases < -16)
        or np.any(biases > 16)
        or np.any(ks >= 12)
    ):
        raise ResidualArchiveError("packed CAP1 metadata exceeds canonical domains")
    result = (
        packed[:102]
        + factors.astype("<i2").tobytes()
        + biases.tobytes()
        + lengths.tobytes()
        + ks.tobytes()
        + packed[142:]
    )
    if len(result) != len(packed) + 40:
        raise ResidualArchiveError("packed CAP1 inverse produced the wrong length")
    return result


def _decode_rx1_models(outer: bytes):
    """Restore RX1's counted IHS1 probability object and frozen MC36 renderer."""
    if len(outer) < RX1_MODEL_HEADER.size or not outer.startswith(RX1_MAGIC):
        return None
    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, carrier_bytes = RX1_MODEL_HEADER.unpack_from(outer)
    model_end = RX1_MODEL_HEADER.size + hpac_bytes + semantic_bytes + carrier_bytes
    if magic != RX1_MAGIC or version != 1 or codec not in (RX1_CODEC_XZ, RX1_CODEC_BROTLI):
        raise ResidualArchiveError("unsupported RX1 model")
    # DDM_SZ1_SEMANTIC_METADATA_SPLIT_V1: reserved bit 0 is the semantic split
    # flag.  Unknown bits still refuse, so the check stays fail-closed and an
    # archive with reserved == 0 takes exactly the path it takes today.
    if table_mode not in (0, 1) or (reserved & ~SZ1_RESERVED_KNOWN_BITS) or min(hpac_bytes, semantic_bytes, carrier_bytes) <= 0:
        raise ResidualArchiveError("invalid RX1 model metadata")
    if model_end + 96 >= len(outer):
        raise ResidualArchiveError("truncated RX1 model")
    offset = RX1_MODEL_HEADER.size
    hpac_stream = outer[offset : offset + hpac_bytes]
    offset += hpac_bytes
    semantic_stream = outer[offset : offset + semantic_bytes]
    offset += semantic_bytes
    carrier_stream = outer[offset : offset + carrier_bytes]
    try:
        hpac = lzma.decompress(hpac_stream, format=lzma.FORMAT_XZ) if codec == RX1_CODEC_XZ else _decompress_brotli(hpac_stream)
    except lzma.LZMAError as error:
        raise ResidualArchiveError("invalid RX1 IHS1 stream") from error
    if not hpac.startswith(b"IHS1"):
        raise ResidualArchiveError("RX1 HPAC is not canonical IHS1")
    semantic_body = _decompress_brotli(semantic_stream)
    carrier_body = _decompress_brotli(carrier_stream)
    # DDM_CK2_WHOLE_SECTION_PLANE2_V1: restore the carrier body BEFORE the packed-CAP1
    # framing arithmetic reads its u24 bit counts, so those offsets see today's bytes.
    if reserved & CK2_RESERVED_CARRIER_PLANE2:
        carrier_body = _ck2_uninterleave_planes(carrier_body)
    # DDM_RR5_ARITH_BASIS_V1: restore the shipped Huffman basis stream from
    # the rider's adaptive-arithmetic form BEFORE the packed-CAP1 framing
    # arithmetic reads its u24 bit counts.  Pure entropy recode: the restored
    # body is byte-identical to the pre-rider body, so every stage below is
    # unchanged.  Generic algorithm, zero transmitted bytes.
    if reserved & RR5_RESERVED_ARITH_BASIS:
        from .rr5_arith_basis import restore_carrier_body as restore_rr5_carrier_body

        carrier_body = restore_rr5_carrier_body(carrier_body)
    # DDM_DX2_CABAC_COEFFICIENTS_V1: the coefficient rider is disjoint from
    # RR5's basis rider. Restore it after RR5 and before packed-CAP1 framing
    # reads the changed residual bit count. Integer-only, device-free decoder.
    if reserved & DX2_RESERVED_CABAC_COEFFICIENTS:
        from .dx2_cabac_coefficients import restore_carrier_body as restore_dx2_carrier_body

        carrier_body = restore_dx2_carrier_body(carrier_body)
    # DDM_RX1_HP4_CARRIER_INVERSE_V1
    # DDM_SA2_VARIABLE_PACKED_CAP1_V1 - the packed CAP1 portion's length is
    # DERIVED from its own u24 bit counts instead of matched against a pinned
    # constant, so a re-solved carrier lattice (whose Rice residual stream has a
    # different length) parses.  Pure framing arithmetic: no video-derived data.
    _packed_portion = 102 + 40 + (
        (int.from_bytes(carrier_body[0:3], "little") + 7) // 8
    ) + ((int.from_bytes(carrier_body[3:6], "little") + 7) // 8)
    if len(carrier_body) >= _packed_portion + 9:
        carrier_body = (
            _restore_packed_cap1_metadata(carrier_body[:_packed_portion])
            + carrier_body[_packed_portion:]
        )
    elif len(carrier_body) != CANONICAL_CAP1_SECTION_BYTES:
        raise ResidualArchiveError("RX1 carrier representation differs")
    # DDM_SZ1_SEMANTIC_METADATA_SPLIT_V1
    # reserved bit 0: the fp16 metadata region is byte-split (high plane, then low
    # plane).  Absent (reserved == 0) is byte-identity: the code path below is
    # unchanged.  The un-split is an exact byte permutation over a frozen region and
    # restores the F12 body byte-for-byte before any parsing, so WANS_BODY_BYTES and
    # every downstream check see exactly the bytes they see today.
    if reserved & SZ1_RESERVED_SEMANTIC_SPLIT:
        semantic_body = _sz1_unsplit_semantic(semantic_body)
    # DDM_CK2_WHOLE_SECTION_PLANE2_V1: whole-section un-split, no fitted constants.
    if reserved & CK2_RESERVED_SEMANTIC_PLANE2:
        semantic_body = _ck2_uninterleave_planes(semantic_body)
    tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R"))
    if tagged_semantic:
        semantic = semantic_body
    else:
        if len(semantic_body) != WANS_BODY_BYTES:
            raise ResidualArchiveError("RX1 semantic section length differs")
        try:
            semantic = decode_f12_wans_body(semantic_body, WANS_STREAM_ORDER)
            decode_wans1(semantic)
        except RendererWeightCodecError as error:
            raise ResidualArchiveError("invalid RX1 renderer weights") from error
    cap1_bytes = _cap1_body_bytes(carrier_body)
    if cap1_bytes >= len(carrier_body):
        raise ResidualArchiveError("truncated RX1 carrier")
    cap1 = _restore_cap1(carrier_body[:cap1_bytes])
    selector_tail = SPARSE_SELECTOR_PREFIX + carrier_body[cap1_bytes:]
    try:
        selector, compensation = split_selector_compensation(selector_tail)
        decode_cap1(cap1, frames=600, dimensions=12)
        decode_selector(selector)
        carrier = pack_frame0_selector_carrier(cap1, selector)
    except (CoefficientAr1CodecError, ValueError) as error:
        raise ResidualArchiveError("invalid RX1 carrier") from error
    return semantic, carrier, hpac, compensation, outer[model_end:], outer[:model_end]


def _decode_hp4_models(outer: bytes) -> tuple[bytes, bytes] | None:
    """Restore the exact F24S body from HP4 order0+Brotli-q11 fields."""
    if len(outer) < HP4_MODEL_HEADER.size or not outer.startswith(HP4_MAGIC):
        return None
    magic, version, predictor, codec, flags, *lengths = HP4_MODEL_HEADER.unpack_from(outer)
    model_end = HP4_MODEL_HEADER.size + sum(lengths)
    if (magic, version, predictor, codec, flags) != (HP4_MAGIC, 1, 0, 2, 0):
        raise ResidualArchiveError("unsupported HP4 model")
    if model_end + 96 >= len(outer):
        raise ResidualArchiveError("truncated HP4 model")
    fields = []
    offset = HP4_MODEL_HEADER.size
    for length in lengths:
        fields.append(outer[offset : offset + length])
        offset += length
    remainder = _decompress_brotli(fields[0])
    if fields[1]:
        raise ResidualArchiveError("HP4 order0 predictor parameters must be empty")
    embedding = _decompress_brotli(fields[2])
    if len(embedding) != HP4_FRAME_BYTES:
        raise ResidualArchiveError("HP4 embedding length differs")
    hpac = remainder[:HP4_FRAME_OFFSET_BODY] + embedding + remainder[HP4_FRAME_OFFSET_BODY:]
    semantic = _decompress_brotli(fields[3])
    carrier = _decompress_brotli(fields[4])
    if len(hpac) != IHS2_BODY_BYTES or len(semantic) != WANS_BODY_BYTES:
        raise ResidualArchiveError("HP4 restored fixed section length differs")
    if len(carrier) == PACKED_CAP1_SECTION_BYTES:
        carrier = _restore_packed_cap1_metadata(carrier)
    elif (
        len(carrier) > PACKED_CAP1_SECTION_BYTES
        and carrier[PACKED_CAP1_SECTION_BYTES:].startswith(COMPENSATION_MAGIC)
    ):
        carrier = (
            _restore_packed_cap1_metadata(carrier[:PACKED_CAP1_SECTION_BYTES])
            + carrier[PACKED_CAP1_SECTION_BYTES:]
        )
    elif len(carrier) != CANONICAL_CAP1_SECTION_BYTES:
        raise ResidualArchiveError("HP4 restored carrier length differs")
    return b"F24S" + hpac + semantic + carrier, outer[model_end:]


def _decode_split_models(outer: bytes) -> tuple[bytes, bytes] | None:
    """Decode CP135's three fixed F24S physical sections when selected."""
    if len(outer) < SPLIT_MODEL_HEADER.size:
        return None
    lengths = SPLIT_MODEL_HEADER.unpack_from(outer)
    model_end = SPLIT_MODEL_HEADER.size + sum(lengths)
    if min(lengths) <= 0 or model_end + 96 >= len(outer):
        return None
    offset = SPLIT_MODEL_HEADER.size
    streams = []
    for length in lengths:
        streams.append(outer[offset : offset + length])
        offset += length
    try:
        sections = tuple(_decompress_brotli(stream) for stream in streams)
    except ResidualArchiveError:
        return None
    if tuple(map(len, sections[:2])) != SPLIT_MODEL_SECTION_BYTES:
        return None
    carrier = sections[2]
    if len(carrier) == PACKED_CAP1_SECTION_BYTES:
        carrier = _restore_packed_cap1_metadata(carrier)
    elif (
        len(carrier) > PACKED_CAP1_SECTION_BYTES
        and carrier[PACKED_CAP1_SECTION_BYTES:].startswith(COMPENSATION_MAGIC)
    ):
        carrier = (
            _restore_packed_cap1_metadata(carrier[:PACKED_CAP1_SECTION_BYTES])
            + carrier[PACKED_CAP1_SECTION_BYTES:]
        )
    elif len(carrier) != CANONICAL_CAP1_SECTION_BYTES:
        return None
    return b"F24S" + sections[0] + sections[1] + carrier, outer[model_end:]


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
    compensation_blob: bytes | None = None
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
    selector_tail = SPARSE_SELECTOR_PREFIX + models[cap1_end:]
    try:
        selector, compensation = split_selector_compensation(selector_tail)
        decode_cap1(cap1, frames=600, dimensions=12)
        decode_selector(selector)
        carrier = pack_frame0_selector_carrier(cap1, selector)
    except (CoefficientAr1CodecError, ValueError) as error:
        raise ResidualArchiveError(f"invalid F24S carrier: {error}") from error
    return semantic, carrier, hpac, compensation


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
    rx1 = _decode_rx1_models(outer)
    if rx1 is not None:
        semantic, carrier, hpac, compensation, section, compressed = rx1
    else:
        split = _decode_hp4_models(outer)
        if split is None:
            split = _decode_split_models(outer)
        if split is not None:
            models, section = split
            compressed = outer[: len(outer) - len(section)]
        else:
            try:
                decoder = lzma.LZMADecompressor(
                    format=lzma.FORMAT_RAW,
                    filters=LZMA_FILTERS,
                )
                models = decoder.decompress(outer)
            except lzma.LZMAError as error:
                raise ResidualArchiveError("invalid F24S model section") from error
            section = decoder.unused_data
            if not decoder.eof or not section:
                raise ResidualArchiveError("truncated F24S model section")
            compressed = outer[: len(outer) - len(section)]
        semantic, carrier, hpac, compensation = _decode_models(models)

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
        compensation_blob=compensation,
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


# --- ddm_rr8 NATIVE FREE CORRECTOR SELECTION --------------------------------------------
# The corrector is 71.7% of the shipped token stage on a contest T4 (ddm_cd1, MEASURED), so
# it is lowered to C.  The C reproduces the frozen SHIPPED_CONFIG only, and refuses to bind
# if that config has drifted -- a refusal costs the speedup, a silent mismatch costs the
# submission (ddm_rr2 scored S = 27.83 on a desynchronised decoder and it read as a model
# failure rather than what it was).
def _rr8_corrector_kind(corrector) -> str:
    """Name the corrector that ACTUALLY ran, into the token report.

    Without this the identity gate is VACUOUS.  If the native build fails, the selector
    falls back to the shipped Python corrector -- correctly -- and a full-field identity run
    then compares the shipped decoder against itself and PASSES, having proved nothing about
    the port.  Absence of a fallback message on stderr is not evidence either.  This field is
    what lets the gate fail, and it is what tells a paid T4 row whether it measured the port
    or the fallback.
    """
    return type(corrector).__name__


def _rr8_select_corrector(plane: int):
    """The C corrector when the runtime built one; otherwise the shipped Python corrector.

    No try/except: ``load_native_corrector`` returns None when no library was built (the
    fail-closed path inflate.sh takes on a hostile toolchain) and RAISES when one was named
    but is missing, has the wrong ABI, or was compiled against a different configuration.
    Swallowing that second class would run a corrector nobody chose.
    """
    from .free_corrector import FreeCorrector
    from .native_free_corrector import load_native_corrector

    native = load_native_corrector(plane)
    if native is not None:
        return native
    return FreeCorrector(plane)


def decode_production_tokens(
    parts: ResidualArchiveParts,
    runtime,
    code_dir: Path,
    device,
) -> tuple[object, dict[str, object]]:
    """Decode the F26 RC64 stream with the fixed boundary residual model."""
    import torch

    from .hpac_inference import configure_cuda_reproducibility, optimize_sparse_evaluator
    from .free_corrector import FreeCorrector

    started = time.time()
    if device.type == "cuda":
        configure_cuda_reproducibility()
    base_hpac = materialize_ihs1(parts.hpac_blob, runtime)
    model = runtime.load_hpac(base_hpac, device)
    masks = runtime.group_masks(device)
    sparse = _sparse_class(code_dir)(model, runtime.EVAL_H, runtime.EVAL_W)
    corrector = _rr8_select_corrector(runtime.EVAL_H * runtime.EVAL_W)
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
            corrector.begin_frame(boundary)
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
                state = corrector.group_state(
                    probability,
                    predicted,
                    flat_positions,
                )
                symbols = decoder.decode(
                    corrector.coding_row(state)
                ).astype(np.int64)
                corrector.observe(state, symbols)
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(
                    device
                )
            tokens[frame] = current[0].to(device="cpu", dtype=torch.uint8)
            corrector.end_frame(tokens[frame].numpy().reshape(-1))
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
        "free_corrector": _rr8_corrector_kind(corrector),
    }


# DDM_SZ1_SEMANTIC_METADATA_SPLIT_V1 -- free receiver code, zero transmitted bytes.
SZ1_RESERVED_SEMANTIC_SPLIT = 0x01
# DDM_CK2_WHOLE_SECTION_PLANE2_V1
# Bits 1 and 2 mark a 2-plane byte de-interleave over the WHOLE decompressed section
# body.  Unlike bit 0 this carries no fitted offset/length: the transform is a pure
# permutation of the entire buffer, so it re-fits nothing when the body layout changes.
CK2_RESERVED_SEMANTIC_PLANE2 = 0x02
CK2_RESERVED_CARRIER_PLANE2 = 0x04
# DDM_RR5_ARITH_BASIS_V1: the basis rider's reserved bit.
RR5_RESERVED_ARITH_BASIS = 0x08
# DDM_DX2_CABAC_COEFFICIENTS_V1: disjoint coefficient-stream rider.
DX2_RESERVED_CABAC_COEFFICIENTS = 0x10
SZ1_RESERVED_KNOWN_BITS = 0x1F


def _ck2_uninterleave_planes(body: bytes) -> bytes:
    """Exact inverse of a whole-body 2-plane de-interleave.

    The encoder emits ``body[0::2] + body[1::2]`` over the even-length prefix and leaves
    any odd trailing byte in place.  Restoring is the same statement read backwards, so
    the transform is total: it round-trips every input length, including 0 and 1.
    """
    span = len(body) & ~1
    half = span // 2
    restored = np.empty(span, dtype=np.uint8)
    planes = np.frombuffer(body[:span], dtype=np.uint8)
    restored[0::2] = planes[:half]
    restored[1::2] = planes[half:]
    return restored.tobytes() + body[span:]
SZ1_SPLIT_OFFSET = 49
SZ1_SPLIT_LENGTH = 8284


def _sz1_unsplit_semantic(body: bytes) -> bytes:
    """Restore interleaved fp16 metadata from its two byte planes (exact inverse)."""
    start = SZ1_SPLIT_OFFSET
    end = start + SZ1_SPLIT_LENGTH
    if len(body) < end:
        raise ResidualArchiveError("RX1 semantic body too short for the metadata split")
    planes = np.frombuffer(body[start:end], dtype=np.uint8)
    half = SZ1_SPLIT_LENGTH // 2
    region = np.empty(SZ1_SPLIT_LENGTH, dtype=np.uint8)
    region[1::2] = planes[:half]
    region[0::2] = planes[half:]
    return body[:start] + region.tobytes() + body[end:]
