#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Rebuild this submission's archive.zip from the pinned upstream archives.

Inputs (verified by SHA-256 before use, never redistributed):
  PR #101 archive.zip  b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e
  PR #110 archive.zip  6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf

Pipeline (deterministic; the ctx coder uses only IEEE-exact float64 model
construction, so the output bytes are platform-independent):
  1. #101 member `x` -> decoder Brotli blob / latent LZMA blob / 607-B sidecar.
  2. decode the entropy layers back to RAW bytes (7 decoder streams, 16,912-B
     latent payload). These raw bytes are the information content.
  3. #110 member `x` -> trailing 249-B FEC6 selector wire payload (and a
     cross-check that #110 embeds #101's source payload byte-for-byte).
  4. re-code each section with the ctx coder (codec_ctx), per-section best-of
     {ctx, passthrough}; decode everything back and assert byte equality.
  5. member = ctx container ++ verbatim sidecar; zip as single member `x`,
     ZIP_STORED, fixed 1980-01-01 timestamp.

Encode-side deps: numpy, constriction, brotli (all present in the challenge
repo's `--group cpu` environment). torch is NOT needed to compress.
"""
from __future__ import annotations

import argparse
import hashlib
import lzma
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import codec_ctx as cc  # type: ignore[import-not-found]

PR101_SHA256 = "b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e"
PR110_SHA256 = "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"

PR101_MEMBER_LEN = 178_158
DECODER_BLOB_LEN = 162_164
LATENT_BLOB_LEN = 15_387
SIDECAR_LEN = 607
SELECTOR_LEN = 249
DECODER_RAW_LEN = 229_014
LATENT_RAW_LEN = 16_912
FP11_HEADER_LEN = 8  # b"FP11" + u32 source_len

EXPECTED_MEMBER_LEN = 177_036
EXPECTED_ARCHIVE_LEN = 177_136


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_input(path: Path, want_sha: str, label: str) -> None:
    got = sha256_file(path)
    if got != want_sha:
        raise SystemExit(f"ERROR: {label} SHA-256 mismatch\n  got  {got}\n  want {want_sha}")
    print(f"{label}: SHA-256 OK ({path})")


def write_stored_zip(zip_path: Path, member_name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(info, payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr101", type=Path, required=True, help="PR #101 archive.zip")
    ap.add_argument("--pr110", type=Path, required=True, help="PR #110 archive.zip")
    ap.add_argument("--out", type=Path, default=HERE / "archive.zip")
    args = ap.parse_args()

    check_input(args.pr101, PR101_SHA256, "PR #101 archive")
    check_input(args.pr110, PR110_SHA256, "PR #110 archive")

    # ---- 1/2: extract + entropy-decode the #101 payload sections ----
    payload101 = zipfile.ZipFile(args.pr101).read("x")
    assert len(payload101) == PR101_MEMBER_LEN, len(payload101)
    decoder_blob = payload101[:DECODER_BLOB_LEN]
    latent_blob = payload101[DECODER_BLOB_LEN:DECODER_BLOB_LEN + LATENT_BLOB_LEN]
    sidecar = payload101[DECODER_BLOB_LEN + LATENT_BLOB_LEN:]
    assert len(sidecar) == SIDECAR_LEN, len(sidecar)

    raws = cc._split_legacy_brotli(decoder_blob)
    assert sum(len(r) for r in raws) == DECODER_RAW_LEN
    latent_raw = lzma.decompress(
        latent_blob, format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 4096,
                  "lc": 3, "lp": 0, "pb": 0}])
    assert len(latent_raw) == LATENT_RAW_LEN

    # ---- 3: extract the FEC6 selector from #110 + lineage cross-check ----
    payload110 = zipfile.ZipFile(args.pr110).read("x")
    assert payload110[:4] == b"FP11"
    assert payload110[FP11_HEADER_LEN:FP11_HEADER_LEN + PR101_MEMBER_LEN] == payload101, \
        "#110 does not embed #101's source payload byte-for-byte"
    selector = payload110[-SELECTOR_LEN:]
    assert selector[:4] == b"FEC6"

    # ---- 4: re-code with the ctx coder, per-section best-of ----
    print("encoding (deterministic, ~1-2 min) ...")
    sections = [
        ("decoder", decoder_blob, cc.encode_decoder_section(raws)),
        ("latents", latent_blob, cc.encode_latent_section(latent_raw)),
        ("selector", selector, cc.encode_selector_section(selector)),
    ]
    chosen, ids = [], []
    for name, cur, new in sections:
        win = len(new) < len(cur)
        chosen.append(new if win else cur)
        ids.append(cc.CODER_CTX if win else cc.CODER_PASSTHROUGH)
        print(f"  {name:<8} {len(cur):>7,} -> {len(new):>7,} B "
              f"[{'ctx' if win else 'passthrough'}]")

    container = cc.pack_container(*chosen, coder_ids=tuple(ids))

    # decode-side round trip must be byte-exact before we ship anything
    s2, l2, sel2 = cc.unpack_container(container)
    assert all(a == b for a, b in zip(raws, s2)) and len(s2) == len(raws)
    assert l2 == latent_raw
    assert sel2 == selector
    print("round-trip: all sections byte-exact")

    # ---- 5: member = container ++ verbatim sidecar; deterministic zip ----
    member = container + sidecar
    if len(member) != EXPECTED_MEMBER_LEN:
        print(f"WARNING: member is {len(member):,} B, expected "
              f"{EXPECTED_MEMBER_LEN:,} B (container {len(container):,} "
              f"+ sidecar {len(sidecar)})")
    write_stored_zip(args.out, "x", member)

    size = args.out.stat().st_size
    print(f"{args.out}: {size:,} B (member {len(member):,} B = container "
          f"{len(container):,} + sidecar {len(sidecar)}; zip overhead {size - len(member)})")
    print(f"archive SHA-256: {sha256_file(args.out)}")
    if size != EXPECTED_ARCHIVE_LEN:
        print(f"WARNING: archive is {size:,} B, expected {EXPECTED_ARCHIVE_LEN:,} B")


if __name__ == "__main__":
    main()
