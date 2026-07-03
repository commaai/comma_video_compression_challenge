#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Pack polished latents into a rhnerv_comma-style ctx container archive.

Member layout (no trailing sidecar — the polish subsumes it):
  [7B header | decoder section | latent section | selector section]

Decoder and selector sections are byte-identical to PR #112's (the decoder
raw streams come verbatim from PR #101, the FEC6 selector wire payload from
PR #110). Only the latent section is rebuilt from the polished grid codes,
per-section best-of {ctx range coder, PR #101 raw-LZMA1}.

Usage:
  python pack.py --codes work/pilot_run/best_codes.pt --out work/archive.zip
"""
from __future__ import annotations

import argparse
import lzma
import sys
import zipfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
QLP_DIR = HERE.parent
REPO = QLP_DIR.parent.parent
RHNERV = REPO / "submissions" / "rhnerv_comma"
sys.path.insert(0, str(RHNERV))
sys.path.insert(0, str(HERE))

import codec_ctx as cc  # noqa: E402
from latent_codec import encode_raw_latent_payload  # noqa: E402

DECODER_BLOB_LEN = 162_164
LATENT_BLOB_LEN = 15_387
SELECTOR_LEN = 249
LZMA_FILTERS = [{"id": lzma.FILTER_LZMA1, "dict_size": 4096,
                 "lc": 3, "lp": 0, "pb": 0}]


def write_stored_zip(zip_path: Path, member_name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(info, payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", type=Path, required=True,
                    help="best_codes.pt from polish.py")
    ap.add_argument("--out", type=Path, default=QLP_DIR / "work" / "archive.zip")
    args = ap.parse_args()

    work = QLP_DIR / "work"

    # Decoder raw streams + selector wire payload, verbatim from upstream.
    payload101 = zipfile.ZipFile(work / "pr101_archive.zip").read("x")
    decoder_blob = payload101[:DECODER_BLOB_LEN]
    raws = cc._split_legacy_brotli(decoder_blob)
    payload110 = zipfile.ZipFile(work / "pr110_archive.zip").read("x")
    selector = payload110[-SELECTOR_LEN:]
    assert selector[:4] == b"FEC6"

    # Polished latent section.
    ck = torch.load(args.codes, weights_only=False)
    latent_raw = encode_raw_latent_payload(ck["q_codes"], ck["mins"], ck["scales"])
    print(f"polished latent raw payload: {len(latent_raw):,} B "
          f"(seg={ck['seg'].mean():.8f} pose={ck['pose'].mean():.8f} on GPU axis)")

    sections = [
        ("decoder", decoder_blob, cc.encode_decoder_section(raws)),
        ("latents", lzma.compress(latent_raw, format=lzma.FORMAT_RAW,
                                  filters=LZMA_FILTERS),
         cc.encode_latent_section(latent_raw)),
        ("selector", selector, cc.encode_selector_section(selector)),
    ]
    chosen, ids = [], []
    for name, passthrough, ctx in sections:
        win = len(ctx) < len(passthrough)
        chosen.append(ctx if win else passthrough)
        ids.append(cc.CODER_CTX if win else cc.CODER_PASSTHROUGH)
        print(f"  {name:<8} passthrough {len(passthrough):>7,} B | "
              f"ctx {len(ctx):>7,} B -> [{'ctx' if win else 'passthrough'}]")

    container = cc.pack_container(*chosen, coder_ids=tuple(ids))

    # Round trip before shipping anything.
    s2, l2, sel2 = cc.unpack_container(container)
    assert list(s2) == list(raws)
    assert l2 == latent_raw
    assert sel2 == selector
    print("round-trip: all sections byte-exact")

    write_stored_zip(args.out, "x", container)
    size = args.out.stat().st_size
    print(f"{args.out}: {size:,} B (member {len(container):,}; "
          f"zip overhead {size - len(container)})")
    print(f"vs #1 (177,136 B): {size - 177_136:+,} B "
          f"-> rate delta {25 * (size - 177_136) / 37_545_489:+.6f}")


if __name__ == "__main__":
    main()
