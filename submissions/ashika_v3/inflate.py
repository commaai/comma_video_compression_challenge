#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Inflate the ashika_v3 archive.

Archive layout (single ZIP member `x`, ZIP_STORED):
  [ctx container: 7-byte header | decoder section | latent section |
   selector section] ++ [verbatim 607-byte latent sidecar]

Decode pipeline:
  1. Unpack ctx container -> raw decoder weight streams + raw latent bytes
     + range-coded FEC6 selector payload.
  2. Reconstruct HNeRV decoder state dict from quantised weight streams.
  3. Reconstruct per-pair latents (fp16 header + temporal-delta codes),
     then apply the 607-byte latent sidecar corrections.
  4. Run the HNeRV decoder on CPU in batches of 32 pairs.
  5. Bicubic upsample each frame from eval size (384x512) to camera
     resolution (874x1164).
  6. Apply per-channel integer biases tuned to minimise metric distortion:
       frame0: R -= 1,  B -= 2
       frame1: G -= 1
  7. Clamp to [0,255], round, apply per-pair FEC6 selector transforms,
     then write uint8 NHWC frames streamed to the output .raw file.

Inflate deps: numpy, torch, constriction (all in the harness base env).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import codec_ctx  # type: ignore[import-not-found]
from codec import (  # type: ignore[import-not-found]
    BASE_CHANNELS,
    EVAL_SIZE,
    LATENT_DIM,
    N_PAIRS,
    decode_decoder_compact,
    decode_latents_compact,
)
from codec_sidecar import SIDECAR_HUFF_ENUM_LEN, apply_latent_sidecar  # type: ignore[import-not-found]
from frame_selector import _blue_tile as selector_blue_tile  # type: ignore[import-not-found]
from model import HNeRVDecoder  # type: ignore[import-not-found]

# ── output resolution ─────────────────────────────────────────────────────────
CAMERA_H, CAMERA_W = 874, 1164

# ── decode batch size (pairs) ─────────────────────────────────────────────────
BATCH_PAIRS = 32

# ── per-channel integer biases applied before final clamp+round ───────────────
# Tuned on the public test video to minimise 100*segnet + sqrt(10*posenet).
#   frame0: R -= 1,  B -= 1
#   frame1: G -= 1
BIAS_F0 = (-1.0, 0.0, -1.0)   # (R, G, B)
BIAS_F1 = ( 0.0, -1.0, 0.0)   # (R, G, B)

# ── FEC6 K=16 compact-selector grammar ───────────────────────────────────────
_FEC6_MODE_IDS = (
    "none",
    "frame0_blue_chroma_amp_1",
    "frame0_blue_chroma_amp_3",
    "frame0_luma_bias_+1",
    "frame0_luma_bias_-1",
    "frame0_luma_bias_-2",
    "frame0_luma_bias_-4",
    "frame0_rgb_bias_m2_p1_p1",
    "frame0_rgb_bias_m4_p2_p2",
    "frame0_rgb_bias_p0_m1_p1",
    "frame0_rgb_bias_p0_m2_p2",
    "frame0_rgb_bias_p0_p1_m1",
    "frame0_rgb_bias_p0_p2_m2",
    "frame0_rgb_bias_p2_m1_m1",
    "frame0_rgb_bias_p4_m2_m2",
    "frame0_roll_dx+0_dy+1",
)
_FEC6_CODE_BITS = (
    "00", "1100", "01", "111010", "11010", "111011", "111100", "100",
    "111101", "11011", "1111110", "111110", "11111110", "101", "11100",
    "11111111",
)
_FEC6_DECODE = {bits: code for code, bits in enumerate(_FEC6_CODE_BITS)}


# ── selector helpers ──────────────────────────────────────────────────────────

def _parse_signed_token(tok: str) -> int:
    if tok.startswith("p"):
        return int(tok[1:])
    if tok.startswith("m"):
        return -int(tok[1:])
    return int(tok)


def _build_mode_spec(mode_id: str) -> tuple[str, tuple[int, ...], int]:
    """Return (family, params, frame_index) for a FEC6 mode-id string."""
    if mode_id == "none":
        return ("identity", (), 0)
    fi = 1 if mode_id.startswith("frame1_") else 0
    base = mode_id.replace("frame1_", "frame0_", 1)
    if base.startswith("frame0_luma_bias_"):
        v = int(base.removeprefix("frame0_luma_bias_"))
        return ("rgb_bias", (v, v, v), fi)
    if base.startswith("frame0_rgb_bias_"):
        p = tuple(_parse_signed_token(t) for t in base.removeprefix("frame0_rgb_bias_").split("_"))
        return ("rgb_bias", p, fi)
    if base.startswith("frame0_blue_chroma_amp_"):
        return ("blue_chroma", (int(base.removeprefix("frame0_blue_chroma_amp_")),), fi)
    if base.startswith("frame0_roll_dx"):
        suf = base.removeprefix("frame0_roll_dx")
        dx_tok, dy_tok = suf.split("_dy", 1)
        return ("roll", (int(dx_tok), int(dy_tok)), fi)
    raise ValueError(f"unsupported FEC6 mode {mode_id!r}")


# Pre-build spec table at import time (fixed for this archive)
_SPECS: tuple[tuple[str, tuple[int, ...], int], ...] = tuple(
    _build_mode_spec(m) for m in _FEC6_MODE_IDS
)


def _decode_fec6_huffman(payload: bytes, n_pairs: int) -> list[int]:
    codes: list[int] = []
    prefix = ""
    for byte_val in payload:
        for shift in range(7, -1, -1):
            prefix += "1" if (byte_val >> shift) & 1 else "0"
            code = _FEC6_DECODE.get(prefix)
            if code is not None:
                codes.append(code)
                prefix = ""
                if len(codes) == n_pairs:
                    return codes
    raise ValueError("FEC6 bitstream truncated")


def _unpack_fec6(payload: bytes) -> list[int]:
    if payload[:4] != b"FEC6":
        raise ValueError(f"bad FEC6 magic: {payload[:4]!r}")
    n_pairs = struct.unpack_from("<H", payload, 4)[0]
    return _decode_fec6_huffman(payload[6:], n_pairs)


def _apply_mode(frame: torch.Tensor, spec: tuple, device: torch.device) -> torch.Tensor:
    """Apply one FEC6 transform to a (3, H, W) float frame."""
    family, params, _ = spec
    if family == "identity":
        return frame
    out = frame.clone()
    if family == "rgb_bias":
        out.add_(torch.tensor(params, dtype=out.dtype, device=device).view(3, 1, 1))
        return out
    if family == "blue_chroma":
        amp = float(params[0])
        tile = selector_blue_tile(out.shape[1], out.shape[2], device=device, dtype=out.dtype)
        out[0].add_(tile * amp)
        out[2].sub_(tile * amp)
        return out
    if family == "roll":
        return torch.roll(out, shifts=(int(params[1]), int(params[0])), dims=(1, 2))
    raise ValueError(f"unsupported family {family!r}")


def _apply_selector(
    frames: torch.Tensor,
    codes: list[int],
    pair_start: int,
) -> torch.Tensor:
    """Apply per-pair FEC6 selector transforms to a (2*batch, 3, H, W) tensor,
    then clamp and round in place."""
    out = frames.clone()
    n_pairs = frames.shape[0] // 2
    device = frames.device
    for k in range(n_pairs):
        spec = _SPECS[codes[pair_start + k]]
        family, _, frame_idx = spec
        if family != "identity":
            out[k * 2 + frame_idx] = _apply_mode(out[k * 2 + frame_idx], spec, device)
    return out.clamp_(0.0, 255.0).round_()


# ── archive parsing ───────────────────────────────────────────────────────────

def _load_archive(member_bytes: bytes):
    """Parse archive member -> (state_dict, latents, selector_codes)."""
    if len(member_bytes) <= SIDECAR_HUFF_ENUM_LEN + 7:
        raise ValueError("archive member too short")
    sidecar = member_bytes[-SIDECAR_HUFF_ENUM_LEN:]
    container = member_bytes[:-SIDECAR_HUFF_ENUM_LEN]
    streams, latent_raw, selector_payload = codec_ctx.unpack_container(container)
    state_dict = decode_decoder_compact(b"".join(streams))
    latents = apply_latent_sidecar(decode_latents_compact(latent_raw), sidecar)
    selector_codes = _unpack_fec6(selector_payload)
    return state_dict, latents, selector_codes


# ── main inflate ──────────────────────────────────────────────────────────────

def inflate(src_bin: str, dst_raw: str) -> int:
    state_dict, latents, selector_codes = _load_archive(Path(src_bin).read_bytes())

    if len(selector_codes) != N_PAIRS:
        raise SystemExit(f"selector has {len(selector_codes)} pairs; expected {N_PAIRS}")

    device = torch.device("cpu")
    eval_h, eval_w = EVAL_SIZE

    # Build bias tensors once
    bias_f0 = torch.tensor(BIAS_F0, dtype=torch.float32, device=device).view(3, 1, 1)
    bias_f1 = torch.tensor(BIAS_F1, dtype=torch.float32, device=device).view(3, 1, 1)

    decoder = HNeRVDecoder(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        eval_size=EVAL_SIZE,
    ).to(device)
    decoder.load_state_dict(state_dict)
    decoder.eval()
    latents = latents.to(device)

    n_written = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, N_PAIRS, BATCH_PAIRS):
            j = min(i + BATCH_PAIRS, N_PAIRS)
            batch = j - i

            # Decode latents -> upsample to camera resolution
            decoded = decoder(latents[i:j])          # (batch, 2, 3, eval_h, eval_w)
            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W),
                               mode="bicubic", align_corners=False)
            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)

            # Apply tuned per-channel integer bias corrections
            up[:, 0].add_(bias_f0)
            up[:, 1].add_(bias_f1)

            # Clamp, round, then per-pair FEC6 selector transforms
            frames = up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W).clamp(0.0, 255.0).round()
            frames = _apply_selector(frames, selector_codes, i)

            # Write NHWC uint8 frames
            fout.write(
                frames.to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy().tobytes()
            )
            n_written += batch * 2

    print(f"wrote {n_written} frames to {dst_raw}")
    return n_written


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
