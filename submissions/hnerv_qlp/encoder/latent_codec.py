#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Encode latent grid codes into PR #101's raw latent payload grammar.

Inverse of hnerv_fec6_fixed_huffman_k16/src/codec.py::decode_latents_compact
(pre-LZMA layer). The 16,912-byte raw payload is what rhnerv_comma's ctx
coder (codec_ctx.encode_latent_section) consumes:

  mins   fp16[28] | scales fp16[28] | delta stream uint8[28*600]

where the delta stream is dim-major in LATENT_DIM_ORDER, first column raw,
subsequent columns (q[j] - q[j-1] + 128) mod 256.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
FEC6_SRC = REPO / "submissions" / "hnerv_fec6_fixed_huffman_k16" / "src"

# fec6's codec.py needs its sibling modules resolvable; several submissions
# ship modules named `codec`/`model`, so load codec.py itself by explicit path.
sys.path.insert(0, str(FEC6_SRC))
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("_fec6_codec", FEC6_SRC / "codec.py")
_fec6_codec = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fec6_codec)
LATENT_DIM = _fec6_codec.LATENT_DIM
LATENT_DIM_ORDER = _fec6_codec.LATENT_DIM_ORDER
N_PAIRS = _fec6_codec.N_PAIRS


def encode_raw_latent_payload(q_codes: torch.Tensor,
                              mins: torch.Tensor,
                              scales: torch.Tensor) -> bytes:
    """(600, 28) uint8 codes + fp32 grid -> 16,912-byte raw payload."""
    q = q_codes.cpu().numpy().astype(np.uint8)
    assert q.shape == (N_PAIRS, LATENT_DIM)
    q_ordered = q[:, list(LATENT_DIM_ORDER)].T.astype(np.int16)  # (28, 600)

    delta = np.empty_like(q_ordered, dtype=np.uint8)
    delta[:, 0] = q_ordered[:, 0].astype(np.uint8)
    diffs = (q_ordered[:, 1:] - q_ordered[:, :-1]) & 255
    delta[:, 1:] = ((diffs + 128) & 255).astype(np.uint8)

    out = bytearray()
    out += mins.cpu().numpy().astype(np.float16).tobytes()
    out += scales.cpu().numpy().astype(np.float16).tobytes()
    out += delta.tobytes()
    return bytes(out)


def quantize_to_grid(latents: torch.Tensor, mins: torch.Tensor,
                     scales: torch.Tensor) -> torch.Tensor:
    """Continuous latents -> uint8 grid codes (clamped)."""
    codes = torch.round((latents - mins) / scales).clamp(0, 255)
    return codes.to(torch.uint8)


def dequantize(q_codes: torch.Tensor, mins: torch.Tensor,
               scales: torch.Tensor) -> torch.Tensor:
    return mins + q_codes.float() * scales


def _selftest() -> None:
    """Byte-exact round trip against the original PR #101 latent blob."""
    import lzma
    import zipfile

    sys.path.insert(0, str(FEC6_SRC.parent))
    from inflate import parse_pr101_frame_selector_archive
    DECODER_BLOB_LEN = _fec6_codec.DECODER_BLOB_LEN
    LATENT_BLOB_LEN = _fec6_codec.LATENT_BLOB_LEN
    LATENT_LZMA_FILTERS = _fec6_codec.LATENT_LZMA_FILTERS

    work = HERE.parent / "work"
    payload = torch.load(work / "payload.pt", weights_only=False)

    with zipfile.ZipFile(work / "pr110_archive.zip") as zf:
        bin_bytes = zf.read("x")
    source_payload, _, _, _ = parse_pr101_frame_selector_archive(bin_bytes)
    latent_blob = source_payload[DECODER_BLOB_LEN:DECODER_BLOB_LEN + LATENT_BLOB_LEN]
    raw_orig = lzma.decompress(latent_blob, format=lzma.FORMAT_RAW,
                               filters=LATENT_LZMA_FILTERS)

    raw_ours = encode_raw_latent_payload(
        payload["q_codes"], payload["mins"], payload["scales"])
    assert raw_ours == raw_orig, (
        f"raw payload mismatch: {len(raw_ours)} vs {len(raw_orig)} bytes")
    print(f"round-trip OK: {len(raw_ours):,} bytes byte-exact vs PR #101")

    deq = dequantize(payload["q_codes"], payload["mins"], payload["scales"])
    assert torch.equal(deq, payload["latents_grid"])
    print("dequantize OK: matches latents_grid exactly")


if __name__ == "__main__":
    _selftest()
