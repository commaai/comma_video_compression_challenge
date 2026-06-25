#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Reproduce archive.zip from the open, attributed HNeRV teacher weights.

Recovers the decoder + latents from PR #110's public release (weights byte-identical
to PR #95), re-encodes them with our int8 codec, and writes archive.zip containing a
single member `payload.bin`. No new training. This is a reproduction baseline for the
accompanying study (see README.md / WRITEUP.md).
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE / "src"))
from codec import build_payload  # noqa: E402

FEC6_URL = ("https://github.com/adpena/comma_video_compression_challenge/releases/"
            "download/fec6-frontier-submission-20260520/archive.zip")


def load_teacher():
    """Return (decoder_sd, latents, meta) from PR #110's release (attributed)."""
    cache = ROOT / "work" / "base_decoder.pt"
    if cache.exists():
        b = torch.load(cache)
        return b["decoder_sd"], b["latents"], b["meta"]
    # download + parse via the sibling fec6 submission (same repo)
    fec6 = ROOT / "submissions" / "hnerv_fec6_fixed_huffman_k16"
    sys.path.insert(0, str(fec6)); sys.path.insert(0, str(fec6 / "src"))
    azip = HERE / "_teacher.zip"
    if not azip.exists():
        print("downloading PR #110 release archive ...")
        urllib.request.urlretrieve(FEC6_URL, azip)
    with zipfile.ZipFile(azip) as z:
        data = z.read("x")
    from inflate import parse_pr101_frame_selector_archive  # type: ignore
    from codec import parse_archive  # type: ignore  (fec6 src/codec)
    source_payload, *_ = parse_pr101_frame_selector_archive(data)
    return parse_archive(source_payload)


def main():
    decoder_sd, latents, meta = load_teacher()
    meta = {"latent_dim": meta["latent_dim"], "base_channels": meta["base_channels"],
            "eval_size": list(meta["eval_size"]), "n_pairs": int(latents.shape[0])}
    payload = build_payload(decoder_sd, latents, meta)
    archive_dir = HERE / "archive"
    archive_dir.mkdir(exist_ok=True)
    (archive_dir / "payload.bin").write_bytes(payload)
    out = HERE / "archive.zip"
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(archive_dir / "payload.bin", "payload.bin")
    print(f"wrote {out} ({out.stat().st_size} bytes); payload {len(payload)} bytes")


if __name__ == "__main__":
    main()
