"""structural inverse of runtime.residual_archive.read_residual_archive().

read_residual_archive() (external/pr140/runtime/residual_archive.py:463-516) shows
the single zip member "p" decomposes as pure concatenation:

    outer   = zip.read("p")
    outer   = compressed_models + section
    section = section[:compact_size] + token_stream
    residual_payload = FIXED_MAGIC + section[:compact_size]

So reassembly needs no reimplementation of compress.py's model-repacking
pipeline (which requires a pinned base archive this repo doesn't have) --
it's exactly the inverse concatenation, confirmed byte-identical against the
real PR140 archive in the original round-trip test.

This is only valid for candidates that reuse `compressed_models` and
`residual_payload` UNCHANGED (true for every context-warp candidate,
since the warp only ever changes which symbols the entropy coder emits for
`token_stream`, never the semantic/carrier/hpac model weights or the fixed
boundary-residual table). A candidate with its own PR141-style packing search
must still go through that packer's own logic -- this module only reassembles
the archive's own internal framing.
"""
from __future__ import annotations

import zipfile
from pathlib import Path


def build_outer_payload(compressed_models: bytes, residual_payload: bytes, token_stream: bytes, fixed_magic: bytes) -> bytes:
    """Reassemble the inner "p" zip-member payload from archive parts.

    `residual_payload` is the FULL synthetic form (FIXED_MAGIC-prefixed) as
    returned by `ResidualArchiveParts.residual_payload` -- the prefix is
    stripped back off here since it was never actually stored on disk
    (`read_residual_archive` re-adds it synthetically on parse).
    """
    if not residual_payload.startswith(fixed_magic):
        raise ValueError("residual_payload does not start with the expected FIXED_MAGIC prefix")
    compact_residual = residual_payload[len(fixed_magic):]
    return compressed_models + compact_residual + token_stream


#: PR140's archive.zip ZipInfo.external_attr (stat mode 0o100644 << 16).
EXTERNAL_ATTR_PR140 = 2175008768
#: PR141's archive.zip ZipInfo.external_attr -- differs from PR140's (a
#: harmless Unix-permission-bits cosmetic field in the zip central
#: directory; doesn't affect parsed content or total size either way).
EXTERNAL_ATTR_PR141 = 27525120


def write_archive(output_path: Path, outer_payload: bytes, external_attr: int = EXTERNAL_ATTR_PR140) -> None:
    """Write a PR140/PR141-shaped archive.zip: single ZIP_STORED member "p".

    Matches the real archives' exact ZipInfo metadata (verified against
    external/pr140 and pr141's own archive infolist()) so a no-op rebuild is
    byte-for-byte identical to the original, not just content-equivalent --
    the real archives use the canonical MS-DOS epoch-zero timestamp (a
    reproducible-build convention), not an actual mtime. `external_attr`
    (Unix permission bits) is the one field observed to differ between the
    two real archives; pass `EXTERNAL_ATTR_PR141` to match that family
    exactly, or leave the PR140 default -- it never affects size or content.
    """
    output_path = Path(output_path)
    info = zipfile.ZipInfo(filename="p", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = external_attr
    info.extract_version = 20
    info.flag_bits = 0
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, outer_payload)


def read_outer_payload(archive_path: Path) -> bytes:
    with zipfile.ZipFile(archive_path) as zf:
        if zf.namelist() != ["p"]:
            raise ValueError(f"unexpected zip members: {zf.namelist()}")
        return zf.read("p")
