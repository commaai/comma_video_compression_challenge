#!/usr/bin/env python3
"""Verify the archived F26 artifact before submitting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from runtime.residual_archive import read_residual_archive

EXPECTED_SHA256 = "12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004"
EXPECTED_ARCHIVE_BYTES = 186_724
EXPECTED_PAYLOAD_BYTES = 186_624


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(archive_path: Path) -> dict[str, object]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    digest = _sha256(archive_path)
    if digest != EXPECTED_SHA256:
        raise ValueError("archive digest does not match F26")
    if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("archive size does not match F26")
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if len(entries) != 1 or entries[0].filename != "p":
            raise ValueError("archive must contain exactly one file named p")
        if entries[0].file_size != EXPECTED_PAYLOAD_BYTES:
            raise ValueError("payload size does not match F26")
    parts = read_residual_archive(archive_path)
    if parts.schema != "fixed_boundary_int6" or parts.token_codec != "rc64":
        raise ValueError("archive does not use the required F26 wire format")
    if parts.table is None:
        raise ValueError("archive is missing the residual correction table")
    return {
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": digest,
        "carrier_bytes": len(parts.carrier_blob),
        "hpac_bytes": len(parts.hpac_blob),
        "payload_bytes": EXPECTED_PAYLOAD_BYTES,
        "residual_schema": parts.schema,
        "semantic_bytes": len(parts.semantic_blob),
        "token_codec": parts.token_codec,
        "token_bytes": len(parts.token_stream),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(__file__).resolve().parent / "archive.zip",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.archive), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
