"""Encoder-selected context warping for the HPAC token model.

HPAC predicts frame t from a context map that #135 sets to the decoded frame t-1. Here the encoder
may instead hand HPAC a slightly warped copy of frame t-1 (a 1-2 px shift, or a tiny zoom about a
point near the horizon, i.e. the ego-motion between frames), picking whichever warp makes frame t
cheapest to code. The per-frame choices are stored with a canonical Huffman code (~3.5 bits/frame). All warps are integer
nearest-neighbour index maps, so encoder and decoder build identical contexts on any machine.
"""

from __future__ import annotations

import numpy as np

H, W = 384, 512
WARP_FRAMES = 600  # one catalogue index per frame (frame 0 is always identity), Huffman coded


def _zoom_index(scale: float, cy: float, cx: float) -> tuple[np.ndarray, np.ndarray]:
    ys = np.clip(np.floor((np.arange(H) - cy) / scale + cy + 0.5).astype(np.int64), 0, H - 1)
    xs = np.clip(np.floor((np.arange(W) - cx) / scale + cx + 0.5).astype(np.int64), 0, W - 1)
    return ys, xs


def _shift_index(dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    # out[y, x] = prev[y - dy, x - dx], edge rows/columns repeat the original border
    ys = np.arange(H) - dy
    xs = np.arange(W) - dx
    ys = np.where((ys < 0) | (ys >= H), np.arange(H), ys)
    xs = np.where((xs < 0) | (xs >= W), np.arange(W), xs)
    return ys.astype(np.int64), xs.astype(np.int64)


def _catalogue() -> list[tuple[np.ndarray, np.ndarray]]:
    identity = (np.arange(H, dtype=np.int64), np.arange(W, dtype=np.int64))
    cats = [identity]
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (0, -2), (0, 2)):
        cats.append(_shift_index(dy, dx))
    for scale in (1.002, 1.003, 1.004, 1.006):
        for cy in (155, 170, 185):
            cats.append(_zoom_index(scale, cy, 256))
    zy, zx = _zoom_index(1.004, 170, 256)
    for dx in (-1, 1):
        sy, sx = _shift_index(0, dx)
        cats.append((zy[sy], zx[sx]))
    return cats


CATALOGUE = _catalogue()


def warp_numpy(prev: np.ndarray, k: int) -> np.ndarray:
    ys, xs = CATALOGUE[k]
    return prev[ys[:, None], xs[None, :]]


def warp_torch(prev, k: int):
    """prev: [1, H, W] long tensor -> warped copy (same device)."""
    import torch

    ys, xs = CATALOGUE[k]
    ys_t = torch.from_numpy(ys).to(prev.device)
    xs_t = torch.from_numpy(xs).to(prev.device)
    return prev[:, ys_t[:, None], xs_t[None, :]]


def _canonical_codes(lengths: list[int]) -> dict[int, tuple[int, int]]:
    """symbol -> (code, length) for canonical Huffman code lengths (0 = unused)."""
    codes, code, prev = {}, 0, 0
    for length, sym in sorted((l, i) for i, l in enumerate(lengths) if l):
        code <<= length - prev
        codes[sym] = (code, length)
        code += 1
        prev = length
    return codes


def encode_choices(choices) -> bytes:
    """Frame 0 is implicit. Header: one 4-bit code length per catalogue entry; then MSB-first codes."""
    import heapq

    choices = [int(c) for c in choices]
    if len(choices) != WARP_FRAMES or choices[0] != 0:
        raise ValueError("expected one choice per frame with frame 0 = identity")
    counts = np.bincount(choices[1:], minlength=len(CATALOGUE))
    used = [i for i in range(len(CATALOGUE)) if counts[i]]
    lengths = [0] * len(CATALOGUE)
    if len(used) == 1:
        lengths[used[0]] = 1
    else:
        heap = [(int(counts[i]), n, [i]) for n, i in enumerate(used)]
        heapq.heapify(heap)
        tie = len(heap)
        while len(heap) > 1:
            a, _, sa = heapq.heappop(heap)
            b, _, sb = heapq.heappop(heap)
            for i in sa + sb:
                lengths[i] += 1
            heapq.heappush(heap, (a + b, tie, sa + sb))
            tie += 1
    if max(lengths) > 15:
        raise ValueError("code length exceeds 4-bit header field")
    codes = _canonical_codes(lengths)
    bits = []
    for sym in choices[1:]:
        code, length = codes[sym]
        bits.extend((code >> (length - 1 - i)) & 1 for i in range(length))
    header = bytes((lengths[i] << 4) | (lengths[i + 1] if i + 1 < len(lengths) else 0) for i in range(0, len(lengths), 2))
    return header + np.packbits(np.array(bits, dtype=np.uint8)).tobytes()


def parse_choices(block: bytes) -> tuple[np.ndarray, int]:
    """Decode the warp block; returns (choices [600], bytes consumed)."""
    header_bytes = (len(CATALOGUE) + 1) // 2
    if len(block) < header_bytes:
        raise ValueError("truncated context warp header")
    lengths = []
    for byte in block[:header_bytes]:
        lengths += [byte >> 4, byte & 15]
    lengths = lengths[: len(CATALOGUE)]
    decode = {(code, length): sym for sym, (code, length) in _canonical_codes(lengths).items()}
    if not decode:
        raise ValueError("empty context warp code")
    bits = np.unpackbits(np.frombuffer(block[header_bytes:], dtype=np.uint8))
    choices, pos = [0], 0
    while len(choices) < WARP_FRAMES:
        code = length = 0
        while (code, length) not in decode:
            if pos >= len(bits) or length >= 15:
                raise ValueError("invalid context warp bitstream")
            code, length, pos = (code << 1) | int(bits[pos]), length + 1, pos + 1
        choices.append(decode[(code, length)])
    return np.asarray(choices, dtype=np.int64), header_bytes + (pos + 7) // 8
