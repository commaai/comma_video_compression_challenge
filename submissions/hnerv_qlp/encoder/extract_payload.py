#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Extract the leaderboard-#1 payload into training-ready tensors.

Reads the PR #110 (fec6) archive — whose information content equals the
current #1 (PR #112 rhnerv_comma, a lossless re-code) — and writes to work/:

  payload.pt:
    decoder_sd     : dict[str, float32 tensor]  (PR #95 weights, frozen forever)
    latents_grid   : (600, 28) float32 — uint8-grid latents, NO sidecar
    latents_eff    : (600, 28) float32 — sidecar-applied (the shipping values)
    q_codes        : (600, 28) uint8   — raw grid codes
    mins, scales   : (28,) float32     — per-dim dequant grid (fp16-exact)
    selector_codes : (600,) int64      — FEC6 K=16 per-pair mode indices
    selector_specs : list of (family, params, frame_index)
    meta           : dict

The quantization grid (mins/scales) is kept FIXED during polish so the
re-encoded latent payload stays format-identical to PR #101's grammar.
"""
from __future__ import annotations

import lzma
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
QLP_DIR = HERE.parent
REPO = QLP_DIR.parent.parent
FEC6 = REPO / "submissions" / "hnerv_fec6_fixed_huffman_k16"
sys.path.insert(0, str(FEC6 / "src"))
sys.path.insert(0, str(FEC6))

from inflate import parse_pr101_frame_selector_archive  # noqa: E402
from codec import (  # noqa: E402
    parse_archive,
    decode_latents_compact,
    DECODER_BLOB_LEN,
    LATENT_BLOB_LEN,
    LATENT_LZMA_FILTERS,
    LATENT_DIM,
    N_PAIRS,
)


def main() -> None:
    work = QLP_DIR / "work"
    archive_zip = work / "pr110_archive.zip"
    with zipfile.ZipFile(archive_zip) as zf:
        bin_bytes = zf.read("x")

    source_payload, sel_kind, sel_codes, sel_specs = (
        parse_pr101_frame_selector_archive(bin_bytes))
    assert sel_kind == "compact" and len(sel_codes) == N_PAIRS

    decoder_sd, latents_eff, meta = parse_archive(source_payload)

    latent_blob = source_payload[DECODER_BLOB_LEN:DECODER_BLOB_LEN + LATENT_BLOB_LEN]
    latents_grid = decode_latents_compact(latent_blob)

    raw = lzma.decompress(latent_blob, format=lzma.FORMAT_RAW,
                          filters=LATENT_LZMA_FILTERS)
    mins = np.frombuffer(raw[: LATENT_DIM * 2], dtype=np.float16).astype(np.float32)
    scales = np.frombuffer(raw[LATENT_DIM * 2: LATENT_DIM * 4],
                           dtype=np.float16).astype(np.float32)

    # Recover the raw grid codes by inverting the dequant (exact: grid values
    # are min + q*scale in float32).
    q = torch.round((latents_grid - torch.from_numpy(mins)) /
                    torch.from_numpy(scales)).to(torch.uint8)
    recon = torch.from_numpy(mins) + q.float() * torch.from_numpy(scales)
    assert torch.equal(recon, latents_grid), "grid code inversion mismatch"

    sidecar_delta = latents_eff - latents_grid
    n_side = int((sidecar_delta.abs() > 1e-9).any(dim=1).sum())

    torch.save({
        "decoder_sd": decoder_sd,
        "latents_grid": latents_grid,
        "latents_eff": latents_eff,
        "q_codes": q,
        "mins": torch.from_numpy(mins.copy()),
        "scales": torch.from_numpy(scales.copy()),
        "selector_codes": torch.tensor(sel_codes, dtype=torch.int64),
        "selector_specs": list(sel_specs),
        "meta": meta,
    }, work / "payload.pt")

    n_params = sum(v.numel() for v in decoder_sd.values())
    print(f"decoder tensors: {len(decoder_sd)} ({n_params:,} params)")
    print(f"latents: {tuple(latents_eff.shape)}  grid step mean: {scales.mean():.5f}")
    print(f"sidecar-corrected pairs: {n_side}/{N_PAIRS} "
          f"(|delta| max {sidecar_delta.abs().max():.4f})")
    print(f"selector: {len(sel_codes)} codes over K={len(sel_specs)} modes, "
          f"non-identity: {int(sum(1 for c in sel_codes if sel_specs[c][0] != 'identity'))}")
    print(f"wrote {work / 'payload.pt'}")


if __name__ == "__main__":
    main()
