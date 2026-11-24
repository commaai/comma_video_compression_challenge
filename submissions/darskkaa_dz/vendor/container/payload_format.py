"""minimal archive format extension.

p_new = compressed_models || compact_residual || warped_token_stream
        || warp_signal || uint16_le(len(warp_signal))

PR141's `compressed_models` and `compact_residual` stay byte-for-byte
unchanged (context warping only ever changes `token_stream`).
Only `token_stream` is replaced, and a trailer carries the warp-choice
signal. The 2-byte little-endian length field lets a wrapper locate and
strip the trailer from the END of member "p" BEFORE handing the remaining
bytes (byte-for-byte identical to the original PR140/PR141 inner payload
layout) to the inherited, unmodified `read_residual_archive()` parsing
logic -- no changes to inherited container-parsing code are needed.
"""
from __future__ import annotations

import struct
import zipfile
from pathlib import Path

TRAILER_LEN_FORMAT = "<H"  # uint16_le
TRAILER_LEN_BYTES = struct.calcsize(TRAILER_LEN_FORMAT)


def build_inner_payload(
    compressed_models: bytes,
    compact_residual: bytes,
    warped_token_stream: bytes,
    warp_signal: bytes,
) -> bytes:
    if len(warp_signal) > 0xFFFF:
        raise ValueError(f"warp_signal too large for uint16_le length field: {len(warp_signal)} bytes")
    core = compressed_models + compact_residual + warped_token_stream
    trailer_len = struct.pack(TRAILER_LEN_FORMAT, len(warp_signal))
    return core + warp_signal + trailer_len


def split_inner_payload(p: bytes) -> tuple[bytes, bytes]:
    """Inverse of build_inner_payload's trailer: returns (core, warp_signal).

    `core` is byte-for-byte identical in layout to an ordinary PR140/PR141
    inner "p" payload (compressed_models || compact_residual || token_stream)
    and can be handed unchanged to the inherited `read_residual_archive()`
    parsing logic (e.g. by rewrapping it in a temporary single-member zip).
    """
    if len(p) < TRAILER_LEN_BYTES:
        raise ValueError("payload too short to contain a warp trailer")
    (warp_signal_nbytes,) = struct.unpack(TRAILER_LEN_FORMAT, p[-TRAILER_LEN_BYTES:])
    signal_start = len(p) - TRAILER_LEN_BYTES - warp_signal_nbytes
    if signal_start < 0:
        raise ValueError("trailer length field exceeds payload size")
    core = p[:signal_start]
    warp_signal = p[signal_start:len(p) - TRAILER_LEN_BYTES]
    return core, warp_signal


def write_archive(output_path: Path, inner_payload: bytes, external_attr: int) -> None:
    """Write the candidate archive.zip: single ZIP_STORED member "p", matching
    the real archives' ZipInfo metadata convention (see container/archive_writer.py).
    """
    output_path = Path(output_path)
    info = zipfile.ZipInfo(filename="p", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = external_attr
    info.extract_version = 20
    info.flag_bits = 0
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, inner_payload)


def read_archive(archive_path: Path) -> bytes:
    with zipfile.ZipFile(archive_path) as zf:
        if zf.namelist() != ["p"]:
            raise ValueError(f"unexpected zip members: {zf.namelist()}")
        return zf.read("p")


def core_to_core_only_archive(core: bytes, tmp_path: Path, external_attr: int) -> Path:
    """Wrap `core` (ordinary PR140/PR141-layout payload) in a plain single-member
    zip so the inherited, UNMODIFIED `read_residual_archive()` can parse it
    with zero changes to that function.
    """
    write_archive(tmp_path, core, external_attr)
    return tmp_path
