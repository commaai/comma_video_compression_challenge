#!/usr/bin/env python3
"""Recode attributed PR #135 integer weights without changing any model byte.

This repacks an existing learned artifact; it does not retrain models or encode
the source video. All inherited neural assets remain inside the counted ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path
import zipfile

from runtime import rowpack

REFERENCE_SHA256 = "12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004"
REFERENCE_BYTES = 186_724
FILTERS = [{"id": lzma.FILTER_LZMA2, "dict_size": 65536, "lc": 0, "lp": 1,
            "pb": 0, "mode": lzma.MODE_NORMAL, "nice_len": 273,
            "mf": lzma.MF_BT4, "depth": 0}]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_payload(payload: bytes) -> tuple[bytes, bytes]:
    """Separate raw-LZMA model representation from unchanged residual/tokens."""
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=FILTERS)
    try:
        models = decoder.decompress(payload, max_length=1 << 20)
    except lzma.LZMAError as error:
        raise ValueError("invalid raw-LZMA model section") from error
    if not decoder.eof or not decoder.unused_data:
        raise ValueError("truncated model section or missing residual/token suffix")
    return models, decoder.unused_data


def single_payload(archive_bytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        if archive.namelist() != ["p"]:
            raise ValueError("archive must contain exactly one member named p")
        return archive.read("p")


def canonical_zip(payload: bytes) -> bytes:
    """Stable metadata and stored payload: compression is already inside p."""
    output = io.BytesIO()
    info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(info, payload)
    return output.getvalue()


def repack(reference_bytes: bytes) -> tuple[bytes, dict]:
    if len(reference_bytes) != REFERENCE_BYTES or sha256(reference_bytes) != REFERENCE_SHA256:
        raise ValueError("input must match the hash-pinned PR #135 F26 reference ZIP")
    original_models, original_suffix = split_payload(single_payload(reference_bytes))
    if not original_models.startswith(b"F24S"):
        raise ValueError("reference contains an unexpected model representation")
    encoded = rowpack.pack_models(original_models)
    if rowpack.unpack_models(encoded) != original_models:
        raise ValueError("row packing did not restore the exact original model bytes")
    new_payload = lzma.compress(encoded, format=lzma.FORMAT_RAW, filters=FILTERS) + original_suffix
    result = canonical_zip(new_payload)
    restored_models, restored_suffix = split_payload(single_payload(result))
    if rowpack.unpack_models(restored_models) != original_models or restored_suffix != original_suffix:
        raise ValueError("finished archive failed exact model/suffix reconstruction")
    savings = len(reference_bytes) - len(result)
    report = {
        "reference_url": "https://github.com/commaai/comma_video_compression_challenge/pull/135",
        "reference_archive_bytes": len(reference_bytes), "reference_sha256": REFERENCE_SHA256,
        "archive_bytes": len(result), "archive_sha256": sha256(result), "saved_bytes": savings,
        "restored_model_sha256": sha256(original_models),
        "unchanged_suffix_sha256": sha256(original_suffix),
        "exact_model_bytes": True, "exact_residual_and_token_bytes": True,
        "conditional_score_rate_improvement": 25 * savings / 37_545_489,
        "limitation": "Byte parity is verified. This command does not render video or run official evaluation.",
    }
    return result, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="hash-pinned upstream archive.zip")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "archive.zip")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("source and output must be different paths")
    archive, report = repack(args.source.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(args.output.name + ".partial")
    pending.write_bytes(archive)
    pending.replace(args.output)
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
