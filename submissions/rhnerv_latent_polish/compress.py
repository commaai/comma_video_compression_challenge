#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Deterministic encoder: rebuild archive.zip byte-for-byte.

Re-runs the PR #112 context range coder (`codec_ctx`) on three raw inputs in
`encoder/`:

  decoder_streams.bin    - the 7 concatenated raw HNeRV decoder weight streams,
                           byte-identical to PR #101/#95 (verbatim, frozen).
  selector_payload.bin   - the raw FEC6 K=16 selector wire payload, byte-
                           identical to PR #110 (verbatim, frozen).
  polished_latent_raw.bin- THIS submission's contribution: the per-pair latent
                           payload after exact-score-gated discrete polish
                           (sidecar folded in; 2,162 net code changes from
                           verified +-1/+-2 steps, selected on the CPU axis).

encode_decoder_section / encode_selector_section are deterministic and
reproduce PR #112's exact decoder and selector sections; encode_latent_section
compresses the polished latents. The three sections are packed into the ctx
container (no sidecar) and stored in a deterministic ZIP (fixed 1980 epoch,
ZIP_STORED member `x`). The build asserts the member SHA-256 and prints the
archive SHA-256, so any drift is caught.

Run: `bash compress.sh`  (or `python compress.py`).
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codec_ctx as cc  # noqa: E402

ENC = HERE / "encoder"
OUT = HERE / "archive.zip"

# integrity anchors (checked at build time)
EXPECTED_MEMBER_SHA = "8f7b808e34c0f679fc7fd4fa5b58395acb03d76f981cd183bbae2453f65f6f22"
EXPECTED_ARCHIVE_SHA = "cfd941de10e5c27a5c855f97b0c84e39f6171f23c53c150e4afd90915f41e395"
EXPECTED_ARCHIVE_BYTES = 176_531


def split_decoder_streams(blob: bytes) -> list[bytes]:
    """Split the concatenated raw decoder streams by schema-derived lengths."""
    out, pos, start = [], 0, 0
    for end in cc.STREAM_ENDS:
        length = sum(cc.TENSOR_SCHEMA[p][1] + 2 for p in range(start, end))
        out.append(blob[pos:pos + length])
        pos += length
        start = end
    if pos != len(blob):
        raise ValueError(f"decoder_streams.bin length {len(blob)} != expected {pos}")
    return out


def build_member() -> bytes:
    streams = split_decoder_streams((ENC / "decoder_streams.bin").read_bytes())
    latent_raw = (ENC / "polished_latent_raw.bin").read_bytes()
    selector_payload = (ENC / "selector_payload.bin").read_bytes()

    dec_sec = cc.encode_decoder_section(streams)
    lat_sec = cc.encode_latent_section(latent_raw)
    sel_sec = cc.encode_selector_section(selector_payload)
    return cc.pack_container(dec_sec, lat_sec, sel_sec, (1, 1, 1))


def write_deterministic_zip(member: bytes, path: Path) -> None:
    info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0
    info.create_system = 0
    if path.exists():
        path.unlink()
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(info, member)


def main() -> int:
    member = build_member()
    member_sha = hashlib.sha256(member).hexdigest()
    print(f"member: {len(member):,} B  sha256 {member_sha}")
    if member_sha != EXPECTED_MEMBER_SHA:
        print(f"ERROR: member SHA mismatch (expected {EXPECTED_MEMBER_SHA})", file=sys.stderr)
        return 1

    write_deterministic_zip(member, OUT)
    archive = OUT.read_bytes()
    archive_sha = hashlib.sha256(archive).hexdigest()
    print(f"archive.zip: {len(archive):,} B  sha256 {archive_sha}")
    if len(archive) != EXPECTED_ARCHIVE_BYTES:
        print(f"ERROR: archive size {len(archive)} != {EXPECTED_ARCHIVE_BYTES}", file=sys.stderr)
        return 1
    if EXPECTED_ARCHIVE_SHA != "ARCHIVE_SHA_PLACEHOLDER" and archive_sha != EXPECTED_ARCHIVE_SHA:
        print(f"ERROR: archive SHA mismatch (expected {EXPECTED_ARCHIVE_SHA})", file=sys.stderr)
        return 1
    print(f"OK: wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
