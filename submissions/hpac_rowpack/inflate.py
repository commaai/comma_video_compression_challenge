#!/usr/bin/env python3
"""Inflate the accounted row-packed ZIP using the inherited HPAC renderer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from runtime.f26_inflate import inflate_archive
from runtime.residual_archive import read_residual_archive

EXPECTED_RAW_BYTES = 1200 * 1164 * 874 * 3


def verify_input(data_dir: Path, archive_path: Path) -> None:
    """The decoder must consume exactly the payload charged by evaluation."""
    payload_path = data_dir / "p"
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    if archive_path.stat().st_size > 2 * (1 << 20) or payload_path.stat().st_size > (1 << 20):
        raise ValueError("archive or extracted payload exceeds fixed-codec size limit")
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"]:
            raise ValueError("archive.zip must contain exactly one member named p")
        if archive.getinfo("p").file_size > (1 << 20):
            raise ValueError("archive member exceeds fixed-codec size limit")
        if archive.read("p") != payload_path.read_bytes():
            raise ValueError("extracted payload differs from the accounted archive.zip")
    # Validate reconstruction and all inherited container boundaries before GPU work.
    read_residual_archive(archive_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("video_names_file", type=Path)
    parser.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    args = parser.parse_args()
    names = [line.strip() for line in args.video_names_file.read_text().splitlines() if line.strip()]
    if names != ["0.mkv"]:
        raise ValueError("this inherited artifact supports exactly the public video 0.mkv")
    here = Path(__file__).resolve().parent
    archive_path = here / "archive.zip"
    verify_input(args.data_dir, archive_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "0.raw"
    pending = args.output_dir / "0.raw.partial"
    pending.unlink(missing_ok=True)
    report = inflate_archive(archive_path, pending, renderer_dir=here / "cpr1", device_name=args.device)
    if pending.stat().st_size != EXPECTED_RAW_BYTES:
        raise ValueError(f"expected {EXPECTED_RAW_BYTES} raw bytes, got {pending.stat().st_size}")
    # Keep an earlier complete result available until its replacement is verified.
    pending.replace(destination)
    report["validated_frame_count"] = 1200
    report["device"] = args.device
    report["output"] = str(destination.resolve())
    report["archive_sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (args.output_dir / "inflate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
