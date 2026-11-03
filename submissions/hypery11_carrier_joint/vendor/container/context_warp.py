"""Minimal prior-art warp family (W0-W4).

Integer nearest-neighbor index remapping only, edge-replicated. No floating
point, no grid_sample, no GPU-dependent interpolation -- per plan requirement.

Convention: applying warp (dx, dy) to a frame produces
    warped[y, x] = frame[clip(y - dy, 0, H-1), clip(x - dx, 0, W-1)]
i.e. the warped frame at (x, y) samples the source at (x-dx, y-dy), clamped
to the valid range at the border (edge replication) instead of wrapping.

This is applied ONLY to the `previous` tensor fed into
`model.prepare_frame_context` -- never to the literal previous frame used by
`_boundary_buckets` or the corrector's own run/agreement state (see
the original analysis for why).
"""
from __future__ import annotations

import torch

WARP_IDS: tuple[str, ...] = ("identity", "dx-1", "dx+1", "dy-1", "dy+1")

_SHIFTS: dict[str, tuple[int, int]] = {
    "identity": (0, 0),
    "dx-1": (-1, 0),
    "dx+1": (1, 0),
    "dy-1": (0, -1),
    "dy+1": (0, 1),
}


def apply_warp(frame: torch.Tensor, warp_id: str) -> torch.Tensor:
    """Apply warp `warp_id` to `frame`, shape (1, H, W) (or (H, W)).

    Returns a new tensor (frame is not modified in place).
    """
    if warp_id == "identity":
        return frame
    dx, dy = _SHIFTS[warp_id]
    if frame.ndim == 3:
        h, w = frame.shape[1], frame.shape[2]
    elif frame.ndim == 2:
        h, w = frame.shape[0], frame.shape[1]
    else:
        raise ValueError(f"expected 2D or 3D frame tensor, got shape {tuple(frame.shape)}")

    ys = torch.arange(h, device=frame.device)
    xs = torch.arange(w, device=frame.device)
    src_y = (ys - dy).clamp_(0, h - 1)
    src_x = (xs - dx).clamp_(0, w - 1)

    if frame.ndim == 3:
        return frame[:, src_y][:, :, src_x]
    return frame[src_y][:, src_x]
