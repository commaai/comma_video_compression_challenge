#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Inflate the rhnerv_comma archive: PR #110's exact decoded output, with the
payload losslessly re-coded by a context-modeled range coder.

Member layout (single ZIP member `x`, ZIP_STORED):
  [ctx container: 7-byte header | decoder section | latent section |
   selector section] ++ [verbatim 607-byte PR #101 latent sidecar]

The ctx container (`codec_ctx.unpack_container`) reproduces, bit-exactly:
  - the 7 raw decoder weight streams of PR #101 (after its Brotli layer),
  - the raw latent payload of PR #101 (after its LZMA layer),
  - the 249-byte FEC6 selector wire payload of PR #110.
The sidecar is carried verbatim and is length-implicit trailing bytes (607,
constant in code), mirroring #101's own trailing-sidecar convention.

Everything from the reconstructed bytes to pixels is the EXACT fec6 (#110)
inflate chain -- HNeRVDecoder, 16-pair batches, bicubic 874x1164 upsample
(align_corners=False), the #98 channel biases (frame0 R-=1, frame0 B-=1,
frame1 G-=1), clamp(0,255).round(), then the per-pair FEC6 selector
transforms (applied AFTER bias+clamp+round, with a final batch
clamp_/round_), uint8, NHWC, streamed write -- so the decoded 0.raw is
byte-identical to PR #110's on the same hardware. Device pinned to CPU.

Inflate-time deps: numpy, torch, constriction (all in the harness base env).
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


CAMERA_H, CAMERA_W = 874, 1164

# --- FEC6 K=16 selector grammar, verbatim from fec6 inflate.py (PR #110) ---
FEC6_FIXED_K16_MODE_IDS = (
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
FEC6_FIXED_K16_CODE_BITS = (
    "00",
    "1100",
    "01",
    "111010",
    "11010",
    "111011",
    "111100",
    "100",
    "111101",
    "11011",
    "1111110",
    "111110",
    "11111110",
    "101",
    "11100",
    "11111111",
)
FEC6_FIXED_K16_DECODE = {bits: code for code, bits in enumerate(FEC6_FIXED_K16_CODE_BITS)}


def parse_signed_token(token: str) -> int:
    if token.startswith("p"):
        return int(token[1:])
    if token.startswith("m"):
        return -int(token[1:])
    return int(token)


def mode_spec_from_static_mode_id(mode_id: str) -> tuple[str, tuple[int, ...], int]:
    if mode_id == "none":
        return ("identity", (), 0)
    frame_index = 1 if mode_id.startswith("frame1_") else 0
    base = mode_id.replace("frame1_", "frame0_", 1)
    if base.startswith("frame0_luma_bias_"):
        value = int(base.removeprefix("frame0_luma_bias_"))
        return ("rgb_bias", (value, value, value), frame_index)
    if base.startswith("frame0_rgb_bias_"):
        params = tuple(parse_signed_token(part) for part in base.removeprefix("frame0_rgb_bias_").split("_"))
        if len(params) != 3:
            raise ValueError(f"bad RGB compact selector mode {mode_id!r}")
        return ("rgb_bias", params, frame_index)
    if base.startswith("frame0_blue_chroma_amp_"):
        return ("blue_chroma", (int(base.removeprefix("frame0_blue_chroma_amp_")),), frame_index)
    if base.startswith("frame0_roll_dx"):
        suffix = base.removeprefix("frame0_roll_dx")
        dx_token, dy_token = suffix.split("_dy", 1)
        return ("roll", (int(dx_token), int(dy_token)), frame_index)
    raise ValueError(f"unsupported static compact selector mode {mode_id!r}")


def unpack_fec6_fixed_huffman_codes(payload: bytes, *, n_pairs: int) -> list[int]:
    codes: list[int] = []
    prefix = ""
    bit_pos = 0
    max_bits = len(payload) * 8
    while len(codes) < n_pairs:
        if bit_pos >= max_bits:
            raise ValueError("FEC6 compact selector bitstream truncated")
        bit = (payload[bit_pos // 8] >> (7 - (bit_pos % 8))) & 1
        bit_pos += 1
        prefix += "1" if bit else "0"
        code = FEC6_FIXED_K16_DECODE.get(prefix)
        if code is not None:
            codes.append(int(code))
            prefix = ""
            continue
        if len(prefix) > 8:
            raise ValueError("FEC6 compact selector contains invalid prefix code")
    if prefix:
        raise ValueError("FEC6 compact selector ended mid-symbol")
    for trailing in range(bit_pos, max_bits):
        if (payload[trailing // 8] >> (7 - (trailing % 8))) & 1:
            raise ValueError("FEC6 compact selector has non-zero padding bits")
    return codes


def unpack_fec6_selector(selector_payload: bytes) -> tuple[list[int], tuple[tuple[str, tuple[int, ...], int], ...]]:
    if selector_payload[:4] != b"FEC6":
        raise ValueError(f"selector magic mismatch: {selector_payload[:4]!r}")
    n_pairs = struct.unpack_from("<H", selector_payload, 4)[0]
    codes = unpack_fec6_fixed_huffman_codes(selector_payload[6:], n_pairs=n_pairs)
    specs = tuple(mode_spec_from_static_mode_id(mode_id) for mode_id in FEC6_FIXED_K16_MODE_IDS)
    return codes, specs


def apply_dynamic_mode(frame_chw: torch.Tensor, spec: tuple[str, tuple[int, ...], int]) -> torch.Tensor:
    family, params, _frame_index = spec
    if family == "identity":
        return frame_chw
    out = frame_chw.clone()
    if family == "rgb_bias":
        delta = torch.tensor(params, dtype=out.dtype, device=out.device).view(3, 1, 1)
        return out + delta
    if family == "blue_chroma":
        amp = float(params[0])
        _channels, height, width = out.shape
        tile = selector_blue_tile(height, width, device=out.device, dtype=out.dtype)
        out[0].add_(tile * amp)
        out[2].sub_(tile * amp)
        return out
    if family == "roll":
        dx, dy = int(params[0]), int(params[1])
        return torch.roll(out, shifts=(dy, dx), dims=(1, 2))
    raise ValueError(f"unsupported compact selector family {family!r}")


def apply_compact_selector_to_frames(
    frames_bchw: torch.Tensor,
    selector_codes: list[int],
    selector_specs: tuple[tuple[str, tuple[int, ...], int], ...],
    *,
    pair_start: int,
) -> torch.Tensor:
    """Verbatim fec6 compact-selector apply: transforms AFTER bias+clamp+round,
    then one final batch clamp_/round_."""
    if frames_bchw.shape[0] % 2 != 0:
        raise ValueError("compact selector expects complete frame pairs")
    out = frames_bchw.clone()
    n_pairs = frames_bchw.shape[0] // 2
    for offset in range(n_pairs):
        pair_index = pair_start + offset
        if pair_index >= len(selector_codes):
            raise ValueError(f"selector has {len(selector_codes)} entries; missing pair {pair_index}")
        spec = selector_specs[int(selector_codes[pair_index])]
        family, _params, frame_index = spec
        if family == "identity":
            continue
        frame_offset = offset * 2 + int(frame_index)
        out[frame_offset] = apply_dynamic_mode(out[frame_offset], spec)
    return out.clamp_(0.0, 255.0).round_()


def parse_member(member_bytes: bytes):
    """ctx container + trailing verbatim sidecar -> (state_dict, latents,
    selector codes, selector specs)."""
    if len(member_bytes) <= SIDECAR_HUFF_ENUM_LEN + 7:
        raise ValueError("member too short")
    sidecar = member_bytes[-SIDECAR_HUFF_ENUM_LEN:]
    container = member_bytes[:-SIDECAR_HUFF_ENUM_LEN]
    streams, latent_raw, selector_payload = codec_ctx.unpack_container(container)
    decoder_sd = decode_decoder_compact(b"".join(streams))
    latents = apply_latent_sidecar(decode_latents_compact(latent_raw), sidecar)
    selector_codes, selector_specs = unpack_fec6_selector(selector_payload)
    return decoder_sd, latents, selector_codes, selector_specs


def inflate(src_bin: str, dst_raw: str) -> int:
    decoder_sd, latents, selector_codes, selector_specs = parse_member(Path(src_bin).read_bytes())

    device = torch.device("cpu")  # pinned: CPU is the leaderboard axis and the byte-stability reference
    decoder = HNeRVDecoder(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        eval_size=EVAL_SIZE,
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = N_PAIRS
    if len(selector_codes) != n_pairs:
        raise SystemExit(f"selector has {len(selector_codes)} pairs; archive requires exactly {n_pairs}")
    eval_h, eval_w = EVAL_SIZE

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            rounded = up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W).clamp(0, 255).round()
            rounded = apply_compact_selector_to_frames(
                rounded,
                selector_codes,
                selector_specs,
                pair_start=i,
            )
            frames = rounded.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            fout.write(frames.tobytes())
            n += batch * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
