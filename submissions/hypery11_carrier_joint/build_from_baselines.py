"""Repack credited PR151/PR152 inputs with a newly encoded carrier component."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import carrier_joint
import packing


def read_payload(path):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["p"]:
            raise ValueError("Expected one payload named p")
        return archive.read("p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference151", required=True, type=Path)
    parser.add_argument("--reference152", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "archive.zip")
    args = parser.parse_args()
    source = read_payload(args.reference151)
    _, parts, _ = packing.split(source)
    previous = read_payload(args.reference152)
    magic, nh, nr, nc, table, reserved = packing.HEADER.unpack_from(previous)
    if magic != b"SMP3":
        raise ValueError("The second reference must use PR152's SMP3 format")
    start = packing.HEADER.size + nh + nr
    carrier = carrier_joint.encode(parts[2], fraction=5, prior_bits=10)
    payload = (packing.HEADER.pack(b"SMP4", nh, nr, len(carrier), table, reserved)
               + previous[packing.HEADER.size:start] + carrier + previous[start + nc:])
    restored = packing.decode(payload)
    expected = packing.component_hashes(source)
    if packing.component_hashes(restored) != expected:
        raise ValueError("Reconstructed model/token components differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    entry = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_STORED
    entry.external_attr = 0o600 << 16
    with zipfile.ZipFile(args.output, "w") as archive:
        archive.writestr(entry, payload)
    result = {
        "status": "component_round_trip_verified_cuda_pending",
        "archive_bytes": args.output.stat().st_size,
        "archive_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "saved_vs_pr151_bytes": args.reference151.stat().st_size - args.output.stat().st_size,
        "saved_vs_pr152_bytes": args.reference152.stat().st_size - args.output.stat().st_size,
        "carrier_bytes": len(carrier),
        "reference_component_hashes": expected,
    }
    args.output.with_suffix(".verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
