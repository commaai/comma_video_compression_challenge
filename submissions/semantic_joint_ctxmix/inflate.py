#!/usr/bin/env python3
"""Inflate the fixed F26 archive for the public challenge video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

from runtime.f26_inflate import inflate_archive

ARCHIVE_SHA256 = "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
ARCHIVE_BYTES = 180_002


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_input(data_dir: Path, archive_path: Path) -> None:
    if _sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("archive.zip does not match the promoted F26 artifact")
    if archive_path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("archive.zip has an unexpected size")
    payload_path = data_dir / "p"
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"]:
            raise ValueError("archive.zip must contain exactly the payload file p")
        if payload_path.read_bytes() != archive.read("p"):
            raise ValueError("extracted payload does not match archive.zip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("base")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    if args.base != "0":
        raise ValueError(f"unsupported public video: {args.base}")
    here = Path(__file__).resolve().parent
    archive_path = here / "archive.zip"
    _verify_input(args.data_dir, archive_path)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "semantic_joint_ctxmix requires CUDA inflation on linux-nvidia-t4; "
            "the measured CPU path exceeded the 1,800-second contest budget"
        )
    device_name = "cuda"
    report = inflate_archive(
        archive_path,
        args.destination,
        renderer_dir=here / "cpr1",
        device_name=device_name,
        num_threads=4,
        checkpoint_dir=args.destination.parent / ".f26_decode_checkpoints",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
