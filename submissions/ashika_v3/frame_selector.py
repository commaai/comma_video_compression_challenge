# SPDX-License-Identifier: MIT
"""Frame-exploit selector grammar and deterministic frame-0 transforms."""

from __future__ import annotations

import math
import struct

import torch


SELECTOR_MAGIC = b"FES1"

# INNOVATION (sister of inflate.py INNOVATION 1): canonical 31-mode palette superset for the frame-exploit
# selector. FEC6's active K=16 subset (see inflate.py FEC6_FIXED_K16_MODE_IDS) is drawn from this palette;
# earlier FES1 / FEC2 / FEC3 formats use the full 31 modes. Modes are deterministic transforms on frame-0
# (luma bias / RGB bias / blue chroma amp / roll) or frame-1 (sister modes for late-pair adjustment).
PALETTE_MODE_IDS = (
    "none",
    "frame0_luma_bias_-4",
    "frame0_luma_bias_-2",
    "frame0_luma_bias_-1",
    "frame0_luma_bias_+1",
    "frame0_luma_bias_+2",
    "frame0_luma_bias_+4",
    "frame0_rgb_bias_p0_m1_p1",
    "frame0_rgb_bias_p0_p1_m1",
    "frame0_rgb_bias_p2_m1_m1",
    "frame0_rgb_bias_m2_p1_p1",
    "frame0_rgb_bias_p0_m2_p2",
    "frame0_rgb_bias_p0_p2_m2",
    "frame0_rgb_bias_p4_m2_m2",
    "frame0_rgb_bias_m4_p2_p2",
    "frame0_blue_chroma_amp_1",
    "frame0_blue_chroma_amp_2",
    "frame0_blue_chroma_amp_3",
    "frame0_roll_dx+1_dy+0",
    "frame0_roll_dx-1_dy+0",
    "frame0_roll_dx+0_dy+1",
    "frame0_roll_dx+0_dy-1",
    "frame1_rgb_bias_p2_m1_m1",
    "frame1_rgb_bias_m2_p1_p1",
    "frame1_luma_bias_-1",
    "frame1_blue_chroma_amp_3",
    "frame1_rgb_bias_p0_m1_p1",
    "frame1_rgb_bias_p0_p1_m1",
    "frame1_luma_bias_+1",
    "frame1_blue_chroma_amp_1",
    "frame1_luma_bias_-2",
)

MODE_PARAMS: dict[str, tuple[str, tuple[int, ...]]] = {
    "none": ("identity", ()),
    "frame0_luma_bias_-4": ("rgb_bias", (-4, -4, -4)),
    "frame0_luma_bias_-2": ("rgb_bias", (-2, -2, -2)),
    "frame0_luma_bias_-1": ("rgb_bias", (-1, -1, -1)),
    "frame0_luma_bias_+1": ("rgb_bias", (1, 1, 1)),
    "frame0_luma_bias_+2": ("rgb_bias", (2, 2, 2)),
    "frame0_luma_bias_+4": ("rgb_bias", (4, 4, 4)),
    "frame0_rgb_bias_p0_m1_p1": ("rgb_bias", (0, -1, 1)),
    "frame0_rgb_bias_p0_p1_m1": ("rgb_bias", (0, 1, -1)),
    "frame0_rgb_bias_p2_m1_m1": ("rgb_bias", (2, -1, -1)),
    "frame0_rgb_bias_m2_p1_p1": ("rgb_bias", (-2, 1, 1)),
    "frame0_rgb_bias_p0_m2_p2": ("rgb_bias", (0, -2, 2)),
    "frame0_rgb_bias_p0_p2_m2": ("rgb_bias", (0, 2, -2)),
    "frame0_rgb_bias_p4_m2_m2": ("rgb_bias", (4, -2, -2)),
    "frame0_rgb_bias_m4_p2_p2": ("rgb_bias", (-4, 2, 2)),
    "frame0_blue_chroma_amp_1": ("blue_chroma", (1,)),
    "frame0_blue_chroma_amp_2": ("blue_chroma", (2,)),
    "frame0_blue_chroma_amp_3": ("blue_chroma", (3,)),
    "frame0_roll_dx+1_dy+0": ("roll", (1, 0)),
    "frame0_roll_dx-1_dy+0": ("roll", (-1, 0)),
    "frame0_roll_dx+0_dy+1": ("roll", (0, 1)),
    "frame0_roll_dx+0_dy-1": ("roll", (0, -1)),
    "frame1_rgb_bias_p2_m1_m1": ("rgb_bias", (2, -1, -1)),
    "frame1_rgb_bias_m2_p1_p1": ("rgb_bias", (-2, 1, 1)),
    "frame1_luma_bias_-1": ("rgb_bias", (-1, -1, -1)),
    "frame1_blue_chroma_amp_3": ("blue_chroma", (3,)),
    "frame1_rgb_bias_p0_m1_p1": ("rgb_bias", (0, -1, 1)),
    "frame1_rgb_bias_p0_p1_m1": ("rgb_bias", (0, 1, -1)),
    "frame1_luma_bias_+1": ("rgb_bias", (1, 1, 1)),
    "frame1_blue_chroma_amp_1": ("blue_chroma", (1,)),
    "frame1_luma_bias_-2": ("rgb_bias", (-2, -2, -2)),
}


def _bits_per_selector(palette_size: int) -> int:
    return max(1, math.ceil(math.log2(max(int(palette_size), 2))))


def unpack_selector_indices(payload: bytes) -> list[int]:
    """Decode the compact ``FES1`` little-endian bit-packed selector payload."""

    if len(payload) < 10:
        raise ValueError("selector payload truncated before header")
    magic, n_pairs, palette_size, bits_per_index, packed_len = struct.unpack_from(
        "<4sHBBH", payload, 0
    )
    if magic != SELECTOR_MAGIC:
        raise ValueError(f"selector magic mismatch: {magic!r}")
    expected_bits = _bits_per_selector(palette_size)
    if bits_per_index != expected_bits:
        raise ValueError(
            f"selector bits_per_index={bits_per_index}, expected {expected_bits}"
        )
    if palette_size != len(PALETTE_MODE_IDS):
        raise ValueError(
            f"selector palette_size={palette_size}, runtime palette={len(PALETTE_MODE_IDS)}"
        )
    packed = payload[10 : 10 + packed_len]
    if len(packed) != packed_len or len(payload) != 10 + packed_len:
        raise ValueError("selector payload length mismatch")
    expected_packed_len = math.ceil(n_pairs * bits_per_index / 8)
    if packed_len != expected_packed_len:
        raise ValueError(
            f"selector packed_len={packed_len}, expected {expected_packed_len}"
        )

    indices: list[int] = []
    accumulator = 0
    nbits = 0
    cursor = 0
    mask = (1 << bits_per_index) - 1
    for _ in range(n_pairs):
        while nbits < bits_per_index:
            if cursor >= len(packed):
                raise ValueError("selector bitstream truncated")
            accumulator |= int(packed[cursor]) << nbits
            cursor += 1
            nbits += 8
        idx = accumulator & mask
        accumulator >>= bits_per_index
        nbits -= bits_per_index
        if idx >= palette_size:
            raise ValueError(f"selector index {idx} out of range {palette_size}")
        indices.append(idx)
    if accumulator != 0:
        raise ValueError("selector bitstream has non-zero trailing bits")
    return indices


def _blue_tile(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tile = torch.tensor(
        [
            [-1, 1, -1, 1, 1, -1, 1, -1],
            [1, -1, 1, -1, -1, 1, -1, 1],
            [-1, 1, 1, -1, 1, -1, -1, 1],
            [1, -1, -1, 1, -1, 1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [-1, -1, 1, 1, -1, -1, 1, 1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [-1, 1, 1, -1, -1, 1, 1, -1],
        ],
        dtype=dtype,
        device=device,
    )
    reps_h = (height + 7) // 8
    reps_w = (width + 7) // 8
    return tile.repeat(reps_h, reps_w)[:height, :width]


def apply_frame0_mode(frame_chw: torch.Tensor, mode_id: str) -> torch.Tensor:
    family, params = MODE_PARAMS[mode_id]
    if family == "identity":
        return frame_chw
    out = frame_chw.clone()
    if family == "rgb_bias":
        delta = torch.tensor(params, dtype=out.dtype, device=out.device).view(3, 1, 1)
        return out + delta
    if family == "blue_chroma":
        amp = float(params[0])
        _channels, height, width = out.shape
        tile = _blue_tile(height, width, device=out.device, dtype=out.dtype)
        out[0].add_(tile * amp)
        out[2].sub_(tile * amp)
        return out
    if family == "roll":
        dx, dy = int(params[0]), int(params[1])
        return torch.roll(out, shifts=(dy, dx), dims=(1, 2))
    raise ValueError(f"unsupported frame-exploit mode family {family!r}")


def apply_selector_to_frames(
    frames_bchw: torch.Tensor,
    selector_indices: list[int],
    *,
    pair_start: int,
) -> torch.Tensor:
    """Apply archive-packed per-pair frame-0 selector modes to a flat frame batch."""

    if frames_bchw.shape[0] % 2 != 0:
        raise ValueError("frame-exploit selector expects complete frame pairs")
    out = frames_bchw.clone()
    n_pairs = frames_bchw.shape[0] // 2
    for offset in range(n_pairs):
        pair_index = pair_start + offset
        if pair_index >= len(selector_indices):
            raise ValueError(
                f"selector has {len(selector_indices)} entries; missing pair {pair_index}"
            )
        mode_id = PALETTE_MODE_IDS[int(selector_indices[pair_index])]
        if mode_id == "none":
            continue
        frame_offset = offset * 2 + (1 if mode_id.startswith("frame1_") else 0)
        out[frame_offset] = apply_frame0_mode(out[frame_offset], mode_id)
    return out.clamp_(0.0, 255.0).round_()


__all__ = [
    "PALETTE_MODE_IDS",
    "apply_frame0_mode",
    "apply_selector_to_frames",
    "unpack_selector_indices",
]
