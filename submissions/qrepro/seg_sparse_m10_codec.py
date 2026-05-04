"""Sparse-M10 seg_targets codec: M4 / M5 / M7 with sparse-M10 override on frames 2+.

Extends the shipped ma1 a1 codec (frame 0 M4, frame 1 M5, frames 2+ M7) with a
sparse-M10 context override on frames 2+. For each frame-2+ pixel we compute a
10-dim context (M7 base + diag_tltl, left_left, top_top_top). If that M10 ctx
landed on the fired-list at encode time we use its empirical row (shipped as a
delta-indexlist); otherwise we fall back to the dense M7 row.

Prune criterion (per `feedback_ctw_prune_0p01pct_retains_98pct.md`):
    ship row iff Σ_s c[ctx,s] × log2( P_emp_q[ctx,s] / P_M7_parent[s] ) > 64 bits

Fire rate on ma1: 166 / 9.77M (0.0017%). Teacher-forced Δ_net = −8,277 B vs the
deployed ma1 a1 archive.

Three new features, all raster-causal:
  - diag_tltl    : (y-2, x-2)
  - left_left    : (y,   x-2)
  - top_top_top  : (y-3, x)

Blob layout (little-endian, extends the shipped a1 layout):

  <HHHBBB>   n_pairs, H, W, precision, peel_class, mask_format
  <I>        mask_payload_len
  <mask_payload>
  <HHI>     spatial_size, m5_size, m6_size
  spatial_freqs     uint16[5^4, N_SYM]
  m5_freqs          uint16[5^5, N_SYM]
  m6_freqs          uint16[5^7, N_SYM]
  # sparse-M10 extension section:
  <B>        m10_version  (=1)
  <B>        n_feats
  <B>*n_feats feat_ids   (0=diag_tltl, 1=left_left, 2=top_top_top, 3=prev_prev_prev)
  <H>        threshold_bits_q8 (informative; decoder ignores)
  <I>        m10_compressed_len
  <m10_compressed_bytes>   # zlib(-15) of: <II>(n_fired, n_ctx) || uint32[n_fired] deltas
                           #              || uint16[n_fired, N_SYM] freqs
  <I>        bs_len
  <bitstream>

Decoder entry point (matches shipped signature for drop-in inflate.py swap):

    decode_seg_split_m10(path: Path) -> np.ndarray           # returns (n_pairs, H, W) uint8
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent

from range_coder import RangeDecoder, RangeEncoder  # noqa: E402
from encode_seg_c2split_purepy import (  # noqa: E402
    BORDER,
    MASK_FORMAT_BZ2_PACKBITS,
    MASK_FORMAT_BZ2_RAW,
    MASK_FORMAT_LZMA_PACKBITS,
    MASK_FORMAT_LZMA_RAW,
    N_CLASSES,
    PRECISION,
    compute_spatial_contexts,
    decode_mask_payload,
    encode_mask_best,
    load_seg_targets_lzma,
    make_remap_tables,
    quantize_freqs,
)
from encode_seg_c2split_a1_purepy import (  # noqa: E402
    build_spatial_counts_m4,
    build_temporal_counts_m5,
    build_temporal_counts_m6,
    compute_tt,
)
from encode_seg_c2split_tr_purepy import compute_tr  # noqa: E402

N_SYM = N_CLASSES - 1

# Feature id table: integer ids serialized in the blob.
FEAT_DIAG_TLTL = 0  # (y-2, x-2)
FEAT_LEFT_LEFT = 1  # (y,   x-2)
FEAT_TOP_TOP_TOP = 2  # (y-3, x)
FEAT_PREV_PREV_PREV = 3  # (t-3, y, x)
FEAT_DIAG_TRTR = 4  # (y-2, x+2) anti-diagonal distance-2; row y-2 raster-safe at decode time
FEAT_PREV_LEFT = 5  # (t-1, y, x-1)
FEAT_PREV_RIGHT = 6  # (t-1, y, x+1)
FEAT_PREV_TOP = 7  # (t-1, y-1, x)
FEAT_PREV_BOTTOM = 8  # (t-1, y+1, x)
FEAT_PREV2_LEFT = 9  # (t-2, y, x-1)
FEAT_PREV2_RIGHT = 10  # (t-2, y, x+1)
FEAT_PREV_BOTTOM_RIGHT = 11  # (t-1, y+1, x+1)
FEAT_PREV_BOTTOM_LEFT = 12  # (t-1, y+1, x-1)
FEAT_PREV_TOP_RIGHT = 13  # (t-1, y-1, x+1)
FEAT_PREV_BOTTOM2 = 14  # (t-1, y+2, x)
FEAT_PREV_RIGHT2 = 15  # (t-1, y, x+2)
FEAT_X_BIN5 = 16  # x coordinate bucket in [0, 4]
FEAT_Y_BIN5 = 17  # y coordinate bucket in [0, 4]
FEAT_X_BIN5_SHIFT = 20  # half-bin-shifted x coordinate bucket in [0, 4]
FEAT_PEEL_DIST42 = 30  # distance above the decoded class-4 support boundary, 42px buckets
FEAT_PEEL_BOUND5 = 31  # decoded class-4 support boundary y bucket in [0, 4]
FEAT_PEEL_SLOPE5 = 32  # local decoded class-4 support boundary slope bucket in [0, 4]

FEAT_ID_TO_NAME = {
    FEAT_DIAG_TLTL: "diag_tltl",
    FEAT_LEFT_LEFT: "left_left",
    FEAT_TOP_TOP_TOP: "top_top_top",
    FEAT_PREV_PREV_PREV: "prev_prev_prev",
    FEAT_DIAG_TRTR: "diag_trtr",
    FEAT_PREV_LEFT: "prev_left",
    FEAT_PREV_RIGHT: "prev_right",
    FEAT_PREV_TOP: "prev_top",
    FEAT_PREV_BOTTOM: "prev_bottom",
    FEAT_PREV2_LEFT: "prev2_left",
    FEAT_PREV2_RIGHT: "prev2_right",
    FEAT_PREV_BOTTOM_RIGHT: "prev_bottom_right",
    FEAT_PREV_BOTTOM_LEFT: "prev_bottom_left",
    FEAT_PREV_TOP_RIGHT: "prev_top_right",
    FEAT_PREV_BOTTOM2: "prev_bottom2",
    FEAT_PREV_RIGHT2: "prev_right2",
    FEAT_X_BIN5: "x_bin5",
    FEAT_Y_BIN5: "y_bin5",
    FEAT_X_BIN5_SHIFT: "x_bin5_shift",
    FEAT_PEEL_DIST42: "peel_dist42",
    FEAT_PEEL_BOUND5: "peel_bound5",
    FEAT_PEEL_SLOPE5: "peel_slope5",
}
FEAT_NAME_TO_ID = {v: k for k, v in FEAT_ID_TO_NAME.items()}

# Deployed triple per step-2 prototype; fired on 166 ctx at thr=64 bits.
DEFAULT_FEATS = ("diag_tltl", "left_left", "top_top_top")
DEFAULT_THRESHOLD_BITS = 64.0

M10_VERSION = 2  # v2 = 3-of-N_SYM freqs (last col reconstructed from row_sum = 1<<precision)
M10_VERSION_V1 = 1
M10_VERSION_V2_3OF4 = 2
M10_VERSION_V3_VARINT = 3  # v2 + LEB128-encoded delta ctx indices (stacks)


# -----------------------------------------------------------------------------
# Feature evaluation helpers (causal, same formulas used in encoder + decoder).
# -----------------------------------------------------------------------------


def feature_frame(
    seg_fi: np.ndarray,
    seg_prev: np.ndarray | None,
    seg_prev_prev: np.ndarray | None,
    seg_prev_prev_prev: np.ndarray | None,
    name: str,
) -> np.ndarray:
    """Compute a causal feature as a (H, W) int64 array.

    seg_fi              : the current frame (fully defined at encode time, fully
                          decoded at decode time since all refs are raster-prior).
    seg_prev*           : previous decoded frames used by temporal-offset features.
    """
    h, w = seg_fi.shape
    out = np.zeros((h, w), dtype=np.int64)
    if name == "diag_tltl":
        out[2:, 2:] = seg_fi[:-2, :-2]
    elif name == "left_left":
        out[:, 2:] = seg_fi[:, :-2]
    elif name == "top_top_top":
        out[3:, :] = seg_fi[:-3, :]
    elif name == "prev_prev_prev":
        if seg_prev_prev_prev is not None:
            out[:] = seg_prev_prev_prev
    elif name == "diag_trtr":
        out[2:, :-2] = seg_fi[:-2, 2:]
    elif name == "prev_left":
        if seg_prev is not None:
            out[:, 1:] = seg_prev[:, :-1]
    elif name == "prev_right":
        if seg_prev is not None:
            out[:, :-1] = seg_prev[:, 1:]
    elif name == "prev_top":
        if seg_prev is not None:
            out[1:, :] = seg_prev[:-1, :]
    elif name == "prev_bottom":
        if seg_prev is not None:
            out[:-1, :] = seg_prev[1:, :]
    elif name == "prev2_left":
        if seg_prev_prev is not None:
            out[:, 1:] = seg_prev_prev[:, :-1]
    elif name == "prev2_right":
        if seg_prev_prev is not None:
            out[:, :-1] = seg_prev_prev[:, 1:]
    elif name == "prev_bottom_right":
        if seg_prev is not None:
            out[:-1, :-1] = seg_prev[1:, 1:]
    elif name == "prev_bottom_left":
        if seg_prev is not None:
            out[:-1, 1:] = seg_prev[1:, :-1]
    elif name == "prev_top_right":
        if seg_prev is not None:
            out[1:, :-1] = seg_prev[:-1, 1:]
    elif name == "prev_bottom2":
        if seg_prev is not None:
            out[:-2, :] = seg_prev[2:, :]
    elif name == "prev_right2":
        if seg_prev is not None:
            out[:, :-2] = seg_prev[:, 2:]
    elif name == "x_bin5":
        out[:] = (np.arange(w, dtype=np.int64)[None, :] * 5) // w
    elif name == "y_bin5":
        out[:] = (np.arange(h, dtype=np.int64)[:, None] * 5) // h
    elif name == "x_bin5_shift":
        out[:] = np.minimum(((np.arange(w, dtype=np.int64)[None, :] + w // 10) * 5) // w, 4)
    elif name == "peel_dist42":
        mask = seg_fi == 4
        bounds = np.full(w, h, dtype=np.int64)
        for x in range(w):
            if mask[h - 1, x]:
                y = h - 1
                while y >= 0 and mask[y, x]:
                    y -= 1
                bounds[x] = y + 1
        d = bounds[None, :] - np.arange(h, dtype=np.int64)[:, None]
        out[:] = np.where(d <= 0, 0, np.minimum(((d - 1) // 42) + 1, 4))
    elif name == "peel_bound5":
        mask = seg_fi == 4
        bounds = np.full(w, h, dtype=np.int64)
        for x in range(w):
            if mask[h - 1, x]:
                y = h - 1
                while y >= 0 and mask[y, x]:
                    y -= 1
                bounds[x] = y + 1
        out[:] = np.minimum((bounds[None, :] * 5) // h, 4)
    elif name == "peel_slope5":
        mask = seg_fi == 4
        bounds = np.full(w, h, dtype=np.int64)
        for x in range(w):
            if mask[h - 1, x]:
                y = h - 1
                while y >= 0 and mask[y, x]:
                    y -= 1
                bounds[x] = y + 1
        prev_bounds = np.concatenate([bounds[:1], bounds[:-1]])
        slope = np.clip(bounds - prev_bounds, -2, 2) + 2
        out[:] = slope[None, :]
    else:
        raise ValueError(name)
    return out


def m7_ctx_for_frame(seg: np.ndarray, fi: int) -> np.ndarray:
    """M7 flat-ctx array for frame fi: (top, left, tl, tr, prev, prev_prev, top_top)."""
    frame = seg[fi]
    top, left, tl = compute_spatial_contexts(frame)
    tr = compute_tr(frame)
    tt = compute_tt(frame)
    prev = seg[fi - 1].astype(np.int64)
    prev_prev = seg[fi - 2].astype(np.int64)
    return (
        ((((((top.astype(np.int64) * 5 + left) * 5 + tl) * 5 + tr) * 5 + prev) * 5 + prev_prev) * 5 + tt)
    )


# -----------------------------------------------------------------------------
# M10 empirical counts + prune.
# -----------------------------------------------------------------------------


def build_m10_counts(
    seg: np.ndarray,
    peel_class: int,
    feats: tuple[str, ...],
) -> tuple[np.ndarray, int]:
    """Frames 2+ M10 counts. Shape (5^(7 + n_feats), N_SYM)."""
    forward, _ = make_remap_tables(peel_class)
    n_extra = len(feats)
    n_ctx = 5 ** (7 + n_extra)
    counts = np.zeros((n_ctx, N_SYM), dtype=np.int64)
    n_pairs = seg.shape[0]
    for fi in range(2, n_pairs):
        frame = seg[fi]
        non_peel = frame != peel_class
        if not non_peel.any():
            continue
        m7_ctx = m7_ctx_for_frame(seg, fi)
        ext_ctx = m7_ctx
        prev = seg[fi - 1].astype(np.int64) if fi >= 1 else None
        prev_prev = seg[fi - 2].astype(np.int64) if fi >= 2 else None
        ppp = seg[fi - 3].astype(np.int64) if fi >= 3 else None
        for name in feats:
            feat = feature_frame(frame, prev, prev_prev, ppp, name)
            ext_ctx = ext_ctx * 5 + feat
        target = forward[frame[non_peel]]
        np.add.at(
            counts,
            (ext_ctx[non_peel].ravel(), target.ravel()),
            1,
        )
    return counts, n_ctx


def quantize_row(counts_row: np.ndarray, precision: int) -> np.ndarray:
    """Single-row floor-1 uint16 quantize (matches encode_seg_c2split_purepy.quantize_freqs)."""
    total = max(counts_row.sum(), 1)
    scaled = (counts_row.astype(np.float64) / total * (1 << precision)).astype(np.int64)
    scaled = np.maximum(scaled, 1)
    diff = (1 << precision) - scaled.sum()
    scaled[scaled.argmax()] += diff
    return scaled.astype(np.uint16)


def quantize_row_batch(counts: np.ndarray, precision: int) -> np.ndarray:
    row_sum = counts.sum(axis=1, keepdims=True).astype(np.float64)
    row_sum = np.maximum(row_sum, 1.0)
    scaled = (counts.astype(np.float64) / row_sum * (1 << precision)).astype(np.int64)
    scaled = np.maximum(scaled, 1)
    diff = (1 << precision) - scaled.sum(axis=1)
    argmax = scaled.argmax(axis=1)
    scaled[np.arange(scaled.shape[0]), argmax] += diff
    return scaled.astype(np.uint16)


def m7_parent_of_m10(n_ctx_m10: int, n_extra_feats: int) -> np.ndarray:
    """Strip the extra-feature suffix from an m10 flat ctx to recover the M7 parent index."""
    return np.arange(n_ctx_m10, dtype=np.int64) // (5 ** n_extra_feats)


def pick_fired_ctx(
    counts10: np.ndarray,
    m7_freqs_flat: np.ndarray,
    m7_parent_of: np.ndarray,
    threshold_bits: float,
    precision: int,
) -> np.ndarray:
    """Return sorted fired_idx (int64) — contexts whose ship-gain exceeds threshold_bits."""
    row_totals = counts10.sum(axis=1)
    observed = row_totals > 0
    if not observed.any():
        return np.zeros((0,), dtype=np.int64)

    m7_probs = m7_freqs_flat.astype(np.float64) / float(1 << precision)
    obs_idx = np.where(observed)[0]
    obs_counts = counts10[obs_idx]
    obs_freqs_q = quantize_row_batch(obs_counts, precision).astype(np.float64) / float(1 << precision)

    bits_ship = -(obs_counts * np.log2(np.clip(obs_freqs_q, 1e-30, 1.0))).sum(axis=1)
    m7_rows = m7_probs[m7_parent_of[obs_idx]]
    bits_m7 = -(obs_counts * np.log2(np.clip(m7_rows, 1e-30, 1.0))).sum(axis=1)
    savings = bits_m7 - bits_ship

    fire_mask = savings > threshold_bits
    fired = np.sort(obs_idx[fire_mask])
    return fired


# -----------------------------------------------------------------------------
# Sparse-M10 table: indexlist format (delta-coded ctx indices + freqs, zlib -15).
# -----------------------------------------------------------------------------


def _leb128_encode_deltas(deltas: np.ndarray) -> bytes:
    """LEB128-encode an array of non-negative integer deltas (numpy -> bytes).

    Each value stored 7-bit base-128 little-endian with continuation-bit top-set
    on all but the last byte. Delta range here is bounded by n_ctx < 10M → max
    4 LEB128 bytes per delta (worse than raw uint32 at 4 bytes only when delta
    ≥ 2^28; ≤ 3 bytes for deltas ≤ 2^21, which is the typical fired-ctx spacing).
    """
    out = bytearray()
    for val in deltas.tolist():
        v = int(val)
        while True:
            byte = v & 0x7F
            v >>= 7
            if v:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                break
    return bytes(out)


def _leb128_decode_deltas(buf: bytes, pos: int, count: int) -> tuple[np.ndarray, int]:
    """Decode `count` LEB128-encoded unsigned deltas from buf starting at pos.

    Returns (deltas uint32 array, new_pos).
    """
    deltas = np.empty(count, dtype=np.uint32)
    for i in range(count):
        result = 0
        shift = 0
        while True:
            byte = buf[pos]
            pos += 1
            result |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        deltas[i] = result
    return deltas, pos


def pack_sparse_m10(
    fired_idx: np.ndarray,
    fired_freqs: np.ndarray,
    n_ctx: int,
    version: int = M10_VERSION,
) -> bytes:
    """Pack sparse table: (n_fired, n_ctx) header || delta-ctx || freqs, zlib-compressed.

    v1: ships full fired_freqs (n_fired × N_SYM uint16) with uint32 deltas.
    v2: ships first N_SYM-1 cols (last col reconstructed at decode from row_sum=1<<precision).
        Still uses uint32 deltas.
    v3: v2 payload + LEB128-encoded deltas (variable-length, 1-4 bytes each).
    """
    if fired_idx.size == 0:
        raw = struct.pack("<II", 0, n_ctx)
    else:
        deltas = np.diff(np.concatenate(([-1], fired_idx.astype(np.int64)))).astype(np.uint32)
        if version in (M10_VERSION_V2_3OF4, M10_VERSION_V3_VARINT):
            freqs_ship = fired_freqs[:, :-1]  # drop last col, reconstruct at decode
        elif version == M10_VERSION_V1:
            freqs_ship = fired_freqs
        else:
            raise ValueError(f"unsupported pack version {version}")
        if version == M10_VERSION_V3_VARINT:
            deltas_bytes = _leb128_encode_deltas(deltas)
        else:
            deltas_bytes = deltas.tobytes()
        raw = (
            struct.pack("<II", int(fired_idx.size), n_ctx)
            + deltas_bytes
            + freqs_ship.astype("<u2").tobytes()
        )
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    return co.compress(raw) + co.flush()


def unpack_sparse_m10(
    compressed: bytes,
    version: int = M10_VERSION,
    precision: int = PRECISION,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Inverse of pack_sparse_m10. Returns (fired_idx int64, fired_freqs uint16, n_ctx).

    For v2, reconstructs the last column of fired_freqs using the row_sum = 1<<precision invariant.
    """
    raw = zlib.decompress(compressed, -15)
    n_fired, n_ctx = struct.unpack_from("<II", raw, 0)
    pos = 8
    if n_fired == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, N_SYM), dtype=np.uint16), n_ctx
    if version == M10_VERSION_V3_VARINT:
        deltas, pos = _leb128_decode_deltas(raw, pos, n_fired)
    elif version in (M10_VERSION_V1, M10_VERSION_V2_3OF4):
        deltas = np.frombuffer(raw, dtype="<u4", count=n_fired, offset=pos).copy()
        pos += 4 * n_fired
    else:
        raise ValueError(f"unsupported unpack version {version}")
    if version in (M10_VERSION_V2_3OF4, M10_VERSION_V3_VARINT):
        cols_shipped = N_SYM - 1
        fired_freqs_partial = (
            np.frombuffer(raw, dtype="<u2", count=n_fired * cols_shipped, offset=pos)
            .reshape(n_fired, cols_shipped)
            .copy()
        )
        # reconstruct last col: row_sum = 1<<precision invariant from quantize_row_batch
        row_sum_const = 1 << precision
        last_col = row_sum_const - fired_freqs_partial.astype(np.int64).sum(axis=1)
        if (last_col < 0).any() or (last_col > 0xFFFF).any():
            raise ValueError(
                f"last-col reconstruction out of uint16 range "
                f"(min={last_col.min()}, max={last_col.max()}); row_sum invariant violated"
            )
        fired_freqs = np.empty((n_fired, N_SYM), dtype=np.uint16)
        fired_freqs[:, :cols_shipped] = fired_freqs_partial
        fired_freqs[:, cols_shipped] = last_col.astype(np.uint16)
    elif version == M10_VERSION_V1:
        fired_freqs = (
            np.frombuffer(raw, dtype="<u2", count=n_fired * N_SYM, offset=pos)
            .reshape(n_fired, N_SYM)
            .copy()
        )
    fired_idx = (np.cumsum(deltas.astype(np.int64)) - 1).astype(np.int64)
    return fired_idx, fired_freqs, n_ctx


# -----------------------------------------------------------------------------
# Blob pack / unpack.
# -----------------------------------------------------------------------------


def pack_archive_blob_m10(
    n_pairs: int,
    h: int,
    w: int,
    precision: int,
    peel_class: int,
    mask_payload: bytes,
    mask_format: int,
    spatial_freqs: np.ndarray,
    m5_freqs: np.ndarray,
    m6_freqs: np.ndarray,
    feat_ids: tuple[int, ...],
    threshold_bits: float,
    m10_compressed: bytes,
    bitstream: bytes,
) -> bytes:
    header = struct.pack("<HHHBBB", n_pairs, h, w, precision, peel_class, mask_format)
    mask_len = struct.pack("<I", len(mask_payload))
    spatial_bytes = spatial_freqs.astype("<u2").tobytes()
    m5_bytes = m5_freqs.astype("<u2").tobytes()
    m6_bytes = m6_freqs.astype("<u2").tobytes()
    sizes = struct.pack("<HHI", len(spatial_bytes), len(m5_bytes), len(m6_bytes))

    thr_q8 = max(0, min(0xFFFF, int(round(threshold_bits * 256))))
    m10_header = struct.pack("<BB", M10_VERSION, len(feat_ids))
    m10_header += bytes(feat_ids)
    m10_header += struct.pack("<HI", thr_q8, len(m10_compressed))

    bslen = struct.pack("<I", len(bitstream))
    return (
        header + mask_len + mask_payload
        + sizes + spatial_bytes + m5_bytes + m6_bytes
        + m10_header + m10_compressed
        + bslen + bitstream
    )


def unpack_archive_blob_m10(blob: bytes) -> dict:
    pos = 0
    n_pairs, h, w, precision, peel_class, mask_format = struct.unpack_from("<HHHBBB", blob, pos)
    pos += struct.calcsize("<HHHBBB")
    (mask_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    mask_payload = blob[pos : pos + mask_len]
    pos += mask_len
    spatial_size, m5_size, m6_size = struct.unpack_from("<HHI", blob, pos)
    pos += struct.calcsize("<HHI")
    spatial_freqs = np.frombuffer(
        blob, dtype="<u2", count=spatial_size // 2, offset=pos
    ).reshape((N_CLASSES,) * 4 + (N_SYM,)).copy()
    pos += spatial_size
    m5_freqs = np.frombuffer(
        blob, dtype="<u2", count=m5_size // 2, offset=pos
    ).reshape((N_CLASSES,) * 5 + (N_SYM,)).copy()
    pos += m5_size
    m6_freqs = np.frombuffer(
        blob, dtype="<u2", count=m6_size // 2, offset=pos
    ).reshape((N_CLASSES,) * 7 + (N_SYM,)).copy()
    pos += m6_size

    m10_version, n_feats = struct.unpack_from("<BB", blob, pos)
    pos += 2
    if m10_version not in (M10_VERSION_V1, M10_VERSION_V2_3OF4, M10_VERSION_V3_VARINT):
        raise ValueError(f"unsupported m10_version {m10_version}")
    feat_ids = tuple(blob[pos : pos + n_feats])
    pos += n_feats
    thr_q8, m10_compressed_len = struct.unpack_from("<HI", blob, pos)
    pos += 6
    m10_compressed = blob[pos : pos + m10_compressed_len]
    pos += m10_compressed_len

    (bs_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    bitstream = blob[pos : pos + bs_len]

    return {
        "n_pairs": n_pairs, "h": h, "w": w,
        "precision": precision, "peel_class": peel_class, "mask_format": mask_format,
        "mask_payload": mask_payload,
        "spatial_freqs": spatial_freqs,
        "m5_freqs": m5_freqs,
        "m6_freqs": m6_freqs,
        "feat_ids": feat_ids,
        "threshold_bits_q8": thr_q8,
        "m10_compressed": m10_compressed,
        "m10_version": m10_version,
        "bitstream": bitstream,
    }


# -----------------------------------------------------------------------------
# Encoder.
# -----------------------------------------------------------------------------


def encode_seg_split_m10(
    seg: np.ndarray,
    peel_class: int,
    spatial_freqs: np.ndarray,
    m5_freqs: np.ndarray,
    m6_freqs: np.ndarray,
    fired_idx: np.ndarray,
    fired_freqs: np.ndarray,
    feats: tuple[str, ...],
    precision: int,
) -> bytes:
    """Range-encode the 4-class residual. Frame 0 M4, frame 1 M5, frames 2+ sparse-M10 w/ M7 fallback."""
    total = 1 << precision
    n_other = N_SYM
    forward, _ = make_remap_tables(peel_class)

    spatial_cdf = np.zeros((N_CLASSES,) * 4 + (n_other + 1,), dtype=np.int64)
    spatial_cdf[..., 1:] = np.cumsum(spatial_freqs.astype(np.int64), axis=-1)
    m5_cdf = np.zeros((N_CLASSES,) * 5 + (n_other + 1,), dtype=np.int64)
    m5_cdf[..., 1:] = np.cumsum(m5_freqs.astype(np.int64), axis=-1)
    m7_cdf_flat = np.zeros((5 ** 7, n_other + 1), dtype=np.int64)
    m7_cdf_flat[:, 1:] = np.cumsum(m6_freqs.reshape(5 ** 7, n_other).astype(np.int64), axis=-1)

    # Sparse-M10 CDFs: one row per fired ctx, indexed by position in fired_idx.
    n_fired = int(fired_idx.size)
    n_extra = len(feats)
    if n_fired > 0:
        fired_cdf = np.zeros((n_fired, n_other + 1), dtype=np.int64)
        fired_cdf[:, 1:] = np.cumsum(fired_freqs.astype(np.int64), axis=-1)
    else:
        fired_cdf = np.zeros((0, n_other + 1), dtype=np.int64)

    # Fast-lookup: m10_ctx → fired slot (−1 if not fired).
    n_ctx_m10 = 5 ** (7 + n_extra)
    m10_to_slot = np.full(n_ctx_m10, -1, dtype=np.int64)
    if n_fired > 0:
        m10_to_slot[fired_idx] = np.arange(n_fired, dtype=np.int64)

    enc = RangeEncoder()
    n_pairs = seg.shape[0]
    es = enc.encode_symbol

    def encode_run(cl: list[int], ch: list[int]) -> None:
        for i in range(len(cl)):
            es(cl[i], ch[i], total)

    # frame 0: M4
    frame = seg[0]
    top, left, tl = compute_spatial_contexts(frame)
    tr = compute_tr(frame)
    non_peel = frame != peel_class
    cdf_pair = spatial_cdf[top, left, tl, tr]
    targets_safe = np.where(non_peel, forward[frame].astype(np.int64), 0)
    cum_low = np.take_along_axis(cdf_pair, targets_safe[..., None], axis=-1)[..., 0]
    cum_high = np.take_along_axis(cdf_pair, targets_safe[..., None] + 1, axis=-1)[..., 0]
    keep = non_peel.ravel()
    encode_run(cum_low.ravel()[keep].tolist(), cum_high.ravel()[keep].tolist())

    # frame 1: M5
    if n_pairs >= 2:
        frame = seg[1]
        prev = seg[0]
        top, left, tl = compute_spatial_contexts(frame)
        tr = compute_tr(frame)
        non_peel = frame != peel_class
        cdf_pair = m5_cdf[top, left, tl, tr, prev]
        targets_safe = np.where(non_peel, forward[frame].astype(np.int64), 0)
        cum_low = np.take_along_axis(cdf_pair, targets_safe[..., None], axis=-1)[..., 0]
        cum_high = np.take_along_axis(cdf_pair, targets_safe[..., None] + 1, axis=-1)[..., 0]
        keep = non_peel.ravel()
        encode_run(cum_low.ravel()[keep].tolist(), cum_high.ravel()[keep].tolist())

    # frames 2+: sparse-M10 with M7 fallback
    for fi in range(2, n_pairs):
        frame = seg[fi]
        non_peel = frame != peel_class
        if not non_peel.any():
            continue

        m7_ctx = m7_ctx_for_frame(seg, fi)
        m10_ctx = m7_ctx
        prev = seg[fi - 1].astype(np.int64) if fi >= 1 else None
        prev_prev = seg[fi - 2].astype(np.int64) if fi >= 2 else None
        ppp = seg[fi - 3].astype(np.int64) if fi >= 3 else None
        for name in feats:
            feat = feature_frame(frame, prev, prev_prev, ppp, name)
            m10_ctx = m10_ctx * 5 + feat

        slot = m10_to_slot[m10_ctx]  # (h, w) int64, -1 if not fired
        use_sparse = slot >= 0

        # Vectorised CDF row lookup: where fired, use fired_cdf[slot]; else use m7_cdf[m7_ctx].
        targets_safe = np.where(non_peel, forward[frame].astype(np.int64), 0)
        # M7 row indices for all pixels
        m7_rows_flat = m7_cdf_flat[m7_ctx.ravel()]  # (h*w, n_other+1)
        if n_fired > 0:
            fired_slots_flat = slot.ravel()
            safe_fired_slots = np.where(fired_slots_flat >= 0, fired_slots_flat, 0)
            fired_rows_flat = fired_cdf[safe_fired_slots]
            use_sparse_flat = use_sparse.ravel()
            cdf_rows_flat = np.where(
                use_sparse_flat[:, None],
                fired_rows_flat,
                m7_rows_flat,
            )
        else:
            cdf_rows_flat = m7_rows_flat

        t_flat = targets_safe.ravel()
        cum_low = cdf_rows_flat[np.arange(cdf_rows_flat.shape[0]), t_flat]
        cum_high = cdf_rows_flat[np.arange(cdf_rows_flat.shape[0]), t_flat + 1]

        keep = non_peel.ravel()
        encode_run(cum_low[keep].tolist(), cum_high[keep].tolist())
        if fi % 50 == 0:
            print(f"  encoded frame {fi}/{n_pairs}", flush=True)

    return enc.finish()


# -----------------------------------------------------------------------------
# Decoder.
# -----------------------------------------------------------------------------


def _decode_frame_m4(
    dec: RangeDecoder,
    frame: np.ndarray,
    mask_frame: np.ndarray,
    cdf_py: list,
    total: int,
    h: int,
    w: int,
    peel_class: int,
    inv_remap_py: list,
) -> None:
    frame_list = [[0] * w for _ in range(h)]
    mask_list = mask_frame.tolist()
    inv = inv_remap_py
    get_target = dec.decode_target
    advance = dec.advance
    for y in range(h):
        for x in range(w):
            if mask_list[y][x]:
                frame_list[y][x] = peel_class
                continue
            top_v = frame_list[y - 1][x] if y > 0 else BORDER
            left_v = frame_list[y][x - 1] if x > 0 else BORDER
            tl_v = frame_list[y - 1][x - 1] if (y > 0 and x > 0) else BORDER
            tr_v = frame_list[y - 1][x + 1] if (y > 0 and x + 1 < w) else BORDER
            cdf_row = cdf_py[top_v][left_v][tl_v][tr_v]
            t = get_target(total)
            s = 0
            while cdf_row[s + 1] <= t:
                s += 1
            frame_list[y][x] = inv[s]
            advance(cdf_row[s], cdf_row[s + 1], total)
    for y in range(h):
        frame[y] = frame_list[y]


def _decode_frame_m5(
    dec: RangeDecoder,
    frame: np.ndarray,
    prev_frame: np.ndarray,
    mask_frame: np.ndarray,
    cdf_py: list,
    total: int,
    h: int,
    w: int,
    peel_class: int,
    inv_remap_py: list,
) -> None:
    frame_list = [[0] * w for _ in range(h)]
    mask_list = mask_frame.tolist()
    prev_list = prev_frame.tolist()
    inv = inv_remap_py
    get_target = dec.decode_target
    advance = dec.advance
    for y in range(h):
        prev_row = prev_list[y]
        for x in range(w):
            if mask_list[y][x]:
                frame_list[y][x] = peel_class
                continue
            top_v = frame_list[y - 1][x] if y > 0 else BORDER
            left_v = frame_list[y][x - 1] if x > 0 else BORDER
            tl_v = frame_list[y - 1][x - 1] if (y > 0 and x > 0) else BORDER
            tr_v = frame_list[y - 1][x + 1] if (y > 0 and x + 1 < w) else BORDER
            prev_v = prev_row[x]
            cdf_row = cdf_py[top_v][left_v][tl_v][tr_v][prev_v]
            t = get_target(total)
            s = 0
            while cdf_row[s + 1] <= t:
                s += 1
            frame_list[y][x] = inv[s]
            advance(cdf_row[s], cdf_row[s + 1], total)
    for y in range(h):
        frame[y] = frame_list[y]


def _decode_frame_m10(
    dec: RangeDecoder,
    frame: np.ndarray,
    prev_frame: np.ndarray,
    prev_prev_frame: np.ndarray,
    prev_prev_prev_frame: np.ndarray | None,
    mask_frame: np.ndarray,
    m7_cdf_flat_py: list,
    fired_cdf_flat_py: list,
    m10_to_slot_dict: dict,
    feat_ids: tuple[int, ...],
    n_extra: int,
    total: int,
    h: int,
    w: int,
    peel_class: int,
    inv_remap_py: list,
) -> None:
    """Frames 2+ sparse-M10 w/ M7 fallback decode."""
    frame_list = [[0] * w for _ in range(h)]
    mask_list = mask_frame.tolist()
    prev_list = prev_frame.tolist()
    pp_list = prev_prev_frame.tolist()
    ppp_list = prev_prev_prev_frame.tolist() if prev_prev_prev_frame is not None else None
    inv = inv_remap_py
    get_target = dec.decode_target
    advance = dec.advance

    fid_diag_tltl = FEAT_DIAG_TLTL
    fid_left_left = FEAT_LEFT_LEFT
    fid_top_top_top = FEAT_TOP_TOP_TOP
    fid_prev_prev_prev = FEAT_PREV_PREV_PREV
    fid_diag_trtr = FEAT_DIAG_TRTR
    fid_prev_left = FEAT_PREV_LEFT
    fid_prev_right = FEAT_PREV_RIGHT
    fid_prev_top = FEAT_PREV_TOP
    fid_prev_bottom = FEAT_PREV_BOTTOM
    fid_prev2_left = FEAT_PREV2_LEFT
    fid_prev2_right = FEAT_PREV2_RIGHT

    for y in range(h):
        prev_row = prev_list[y]
        pp_row = pp_list[y]
        prev_row_above = prev_list[y - 1] if y > 0 else None
        prev_row_below = prev_list[y + 1] if y + 1 < h else None
        ppp_row = ppp_list[y] if ppp_list is not None else None
        for x in range(w):
            if mask_list[y][x]:
                frame_list[y][x] = peel_class
                continue
            top_v = frame_list[y - 1][x] if y > 0 else BORDER
            left_v = frame_list[y][x - 1] if x > 0 else BORDER
            tl_v = frame_list[y - 1][x - 1] if (y > 0 and x > 0) else BORDER
            tr_v = frame_list[y - 1][x + 1] if (y > 0 and x + 1 < w) else BORDER
            prev_v = prev_row[x]
            pp_v = pp_row[x]
            tt_v = frame_list[y - 2][x] if y > 1 else BORDER

            m7_ctx = ((((((top_v * 5 + left_v) * 5 + tl_v) * 5 + tr_v) * 5 + prev_v) * 5 + pp_v) * 5 + tt_v)

            # Build M10 ctx by multiplying in each configured feature.
            m10_ctx = m7_ctx
            for fid in feat_ids:
                if fid == fid_diag_tltl:
                    fv = frame_list[y - 2][x - 2] if (y >= 2 and x >= 2) else BORDER
                elif fid == fid_left_left:
                    fv = frame_list[y][x - 2] if x >= 2 else BORDER
                elif fid == fid_top_top_top:
                    fv = frame_list[y - 3][x] if y >= 3 else BORDER
                elif fid == fid_prev_prev_prev:
                    fv = ppp_row[x] if ppp_row is not None else BORDER
                elif fid == fid_diag_trtr:
                    fv = frame_list[y - 2][x + 2] if (y >= 2 and x + 2 < w) else BORDER
                elif fid == fid_prev_left:
                    fv = prev_row[x - 1] if x >= 1 else BORDER
                elif fid == fid_prev_right:
                    fv = prev_row[x + 1] if x + 1 < w else BORDER
                elif fid == fid_prev_top:
                    fv = prev_row_above[x] if prev_row_above is not None else BORDER
                elif fid == fid_prev_bottom:
                    fv = prev_row_below[x] if prev_row_below is not None else BORDER
                elif fid == fid_prev2_left:
                    fv = pp_row[x - 1] if x >= 1 else BORDER
                elif fid == fid_prev2_right:
                    fv = pp_row[x + 1] if x + 1 < w else BORDER
                else:
                    raise ValueError(fid)
                m10_ctx = m10_ctx * 5 + fv

            slot = m10_to_slot_dict.get(m10_ctx)
            if slot is None:
                cdf_row = m7_cdf_flat_py[m7_ctx]
            else:
                cdf_row = fired_cdf_flat_py[slot]

            t = get_target(total)
            s = 0
            while cdf_row[s + 1] <= t:
                s += 1
            frame_list[y][x] = inv[s]
            advance(cdf_row[s], cdf_row[s + 1], total)
    for y in range(h):
        frame[y] = frame_list[y]


def decode_seg_split_m10(path: "Path | str") -> np.ndarray:
    """Decode sparse-M10 seg_targets.bin → (n_pairs, H, W) uint8 array."""
    blob = Path(path).read_bytes()
    f = unpack_archive_blob_m10(blob)
    n_pairs = f["n_pairs"]
    h = f["h"]
    w = f["w"]
    precision = f["precision"]
    peel_class = f["peel_class"]
    spatial_freqs = f["spatial_freqs"]
    m5_freqs = f["m5_freqs"]
    m6_freqs = f["m6_freqs"]
    feat_ids = f["feat_ids"]
    bitstream = f["bitstream"]

    mask = decode_mask_payload(f["mask_payload"], f["mask_format"], n_pairs, h, w)
    _, inverse = make_remap_tables(peel_class)
    inv_remap_py = inverse.tolist()

    total = 1 << precision
    n_other = N_SYM
    spatial_cdf = np.zeros((N_CLASSES,) * 4 + (n_other + 1,), dtype=np.int64)
    spatial_cdf[..., 1:] = np.cumsum(spatial_freqs.astype(np.int64), axis=-1)
    m5_cdf = np.zeros((N_CLASSES,) * 5 + (n_other + 1,), dtype=np.int64)
    m5_cdf[..., 1:] = np.cumsum(m5_freqs.astype(np.int64), axis=-1)
    m7_cdf_flat = np.zeros((5 ** 7, n_other + 1), dtype=np.int64)
    m7_cdf_flat[:, 1:] = np.cumsum(m6_freqs.reshape(5 ** 7, n_other).astype(np.int64), axis=-1)
    spatial_py = spatial_cdf.tolist()
    m5_py = m5_cdf.tolist()
    m7_cdf_flat_py = m7_cdf_flat.tolist()

    # Load sparse-M10 table (version-aware unpack)
    fired_idx, fired_freqs, n_ctx_m10 = unpack_sparse_m10(
        f["m10_compressed"], version=f["m10_version"], precision=precision,
    )
    n_extra = len(feat_ids)
    if n_ctx_m10 != 5 ** (7 + n_extra):
        raise ValueError(f"n_ctx mismatch: blob has {n_ctx_m10}, expected 5^{7 + n_extra}")
    fired_cdf_flat_py: list = []
    if fired_idx.size > 0:
        fired_cdf_arr = np.zeros((fired_idx.size, n_other + 1), dtype=np.int64)
        fired_cdf_arr[:, 1:] = np.cumsum(fired_freqs.astype(np.int64), axis=-1)
        fired_cdf_flat_py = fired_cdf_arr.tolist()
    m10_to_slot_dict: dict[int, int] = {
        int(ctx): i for i, ctx in enumerate(fired_idx.tolist())
    }

    dec = RangeDecoder(bitstream)
    out = np.zeros((n_pairs, h, w), dtype=np.uint8)

    _decode_frame_m4(
        dec, out[0], mask[0], spatial_py, total, h, w, peel_class, inv_remap_py
    )
    if n_pairs >= 2:
        _decode_frame_m5(
            dec, out[1], out[0], mask[1],
            m5_py, total, h, w, peel_class, inv_remap_py,
        )
    for fi in range(2, n_pairs):
        ppp = out[fi - 3] if fi >= 3 else None
        _decode_frame_m10(
            dec, out[fi], out[fi - 1], out[fi - 2], ppp, mask[fi],
            m7_cdf_flat_py, fired_cdf_flat_py, m10_to_slot_dict,
            feat_ids, n_extra,
            total, h, w, peel_class, inv_remap_py,
        )
        if fi % 50 == 0:
            print(f"  decoded frame {fi}/{n_pairs}", flush=True)
    return out


# -----------------------------------------------------------------------------
# Build blob end-to-end.
# -----------------------------------------------------------------------------


def build_blob_from_seg(
    seg: np.ndarray,
    peel_class: int,
    precision: int = PRECISION,
    feats: tuple[str, ...] = DEFAULT_FEATS,
    threshold_bits: float = DEFAULT_THRESHOLD_BITS,
) -> tuple[bytes, dict]:
    """End-to-end encode. Returns (blob, diagnostics dict)."""
    for name in feats:
        if name not in FEAT_NAME_TO_ID:
            raise ValueError(f"unknown feat: {name}")
    feat_ids = tuple(FEAT_NAME_TO_ID[n] for n in feats)

    print("  building M4 counts (frame 0)...", flush=True)
    m4_counts = build_spatial_counts_m4(seg, peel_class)
    print("  building M5 counts (frame 1)...", flush=True)
    m5_counts = build_temporal_counts_m5(seg, peel_class)
    print("  building M7 counts (frames 2+)...", flush=True)
    m6_counts = build_temporal_counts_m6(seg, peel_class)

    print(f"  building M10 counts (frames 2+, feats={feats})...", flush=True)
    m10_counts, n_ctx_m10 = build_m10_counts(seg, peel_class, feats)

    print("  quantizing M4/M5/M7 freqs...", flush=True)
    spatial_freqs = quantize_freqs(m4_counts, precision)
    m5_freqs = quantize_freqs(m5_counts, precision)
    m6_freqs = quantize_freqs(m6_counts, precision)

    print(f"  picking fired M10 ctx (thr={threshold_bits} bits)...", flush=True)
    m7_parent = m7_parent_of_m10(n_ctx_m10, len(feats))
    fired_idx = pick_fired_ctx(m10_counts, m6_freqs.reshape(-1, N_SYM), m7_parent, threshold_bits, precision)
    n_fired = int(fired_idx.size)
    fire_rate = n_fired / n_ctx_m10
    print(f"    {n_fired:,} / {n_ctx_m10:,} ctx fired ({fire_rate * 100:.4f}%)", flush=True)

    fired_freqs = (
        quantize_row_batch(m10_counts[fired_idx], precision)
        if n_fired > 0
        else np.zeros((0, N_SYM), dtype=np.uint16)
    )
    m10_compressed = pack_sparse_m10(fired_idx, fired_freqs, n_ctx_m10)
    print(f"    sparse-M10 table compressed: {len(m10_compressed):,} B", flush=True)

    print(f"  encoding peel-class binary mask (peel=c{peel_class})...", flush=True)
    is_peel = (seg == peel_class).astype(np.uint8).reshape(-1)
    mask_payload, mask_format = encode_mask_best(is_peel)
    mask_format_names = {
        MASK_FORMAT_BZ2_RAW: "bz2-raw",
        MASK_FORMAT_BZ2_PACKBITS: "bz2-packbits",
        MASK_FORMAT_LZMA_RAW: "lzma-raw",
        MASK_FORMAT_LZMA_PACKBITS: "lzma-packbits",
    }
    print(f"    chose {mask_format_names[mask_format]}: {len(mask_payload):,} B", flush=True)

    print("  range-encoding 4-class residual (M4/M5/sparse-M10)...", flush=True)
    t0 = time.time()
    bitstream = encode_seg_split_m10(
        seg, peel_class, spatial_freqs, m5_freqs, m6_freqs,
        fired_idx, fired_freqs, feats, precision,
    )
    elapsed = time.time() - t0
    n_non_peel = int((seg != peel_class).sum())
    print(
        f"    encoded {n_non_peel:,} non-peel symbols in {elapsed:.1f}s "
        f"({n_non_peel / max(elapsed, 1e-6):.0f} sym/s, {len(bitstream):,} B bitstream)",
        flush=True,
    )

    blob = pack_archive_blob_m10(
        seg.shape[0], seg.shape[1], seg.shape[2], precision,
        peel_class, mask_payload, mask_format,
        spatial_freqs, m5_freqs, m6_freqs,
        feat_ids, threshold_bits, m10_compressed, bitstream,
    )

    diag = {
        "blob_size": len(blob),
        "bitstream_size": len(bitstream),
        "mask_size": len(mask_payload),
        "m10_compressed_size": len(m10_compressed),
        "n_fired": n_fired,
        "n_ctx_m10": n_ctx_m10,
        "fire_rate": fire_rate,
        "feats": feats,
        "threshold_bits": threshold_bits,
    }
    return blob, diag


# -----------------------------------------------------------------------------
# CLI / round-trip check.
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peel", type=int, default=4, choices=list(range(N_CLASSES)))
    parser.add_argument(
        "--src", type=str,
        default="seg_targets.bin",
        help="source seg_targets.bin",
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--ref",
        type=str,
        default=None,
        help="optional reference seg_targets.bin for byte-delta reporting",
    )
    parser.add_argument("--roundtrip", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--feats", type=str, default=",".join(DEFAULT_FEATS),
        help="comma-separated feature names",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_BITS)
    args = parser.parse_args()

    repo = REPO
    src_path = Path(args.src)
    if not src_path.is_absolute():
        src_path = Path.cwd() / src_path
    out_path = Path(args.out) if args.out else Path(__file__).resolve().with_name("seg_targets_m10.bin")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Source load: try shipped a1 codec first, else lzma fallback.
    print(f"loading seg from {src_path}", flush=True)
    try:
        from seg_c2split_a1_codec import decode_seg_split_a1
        seg_arr = decode_seg_split_a1(src_path)
        if isinstance(seg_arr, np.ndarray):
            seg = seg_arr.astype(np.uint8)
        else:
            raise TypeError("unexpected type")
    except Exception:
        seg = load_seg_targets_lzma(src_path)

    if args.limit is not None:
        seg = seg[: args.limit]
    print(f"  shape: {seg.shape}  peel_class: c{args.peel}", flush=True)

    feats = tuple(s.strip() for s in args.feats.split(",") if s.strip())
    blob, diag = build_blob_from_seg(seg, args.peel, PRECISION, feats, args.threshold)
    out_path.write_bytes(blob)

    try:
        rel = out_path.relative_to(repo)
    except ValueError:
        rel = out_path
    print(f"\nwrote {rel}", flush=True)
    print(f"  total bytes: {len(blob):,}", flush=True)

    ref_path = (repo / args.ref).resolve()
    if ref_path.exists():
        ref_bytes = ref_path.stat().st_size
        delta = len(blob) - ref_bytes
        print(f"  vs reference {ref_path.name} ({ref_bytes:,} B): {delta:+,} B", flush=True)

    if args.roundtrip:
        print("\nroundtrip: decoding blob...", flush=True)
        t0 = time.time()
        recovered = decode_seg_split_m10(out_path)
        elapsed = time.time() - t0
        print(f"  decoded {recovered.shape} in {elapsed:.1f}s", flush=True)
        if not np.array_equal(recovered, seg):
            n_diff = int((recovered != seg).sum())
            first_diff = np.argwhere(recovered != seg)[:5]
            raise SystemExit(
                f"ROUNDTRIP FAILED: {n_diff:,} pixels differ. first diffs:\n{first_diff}"
            )
        print("  PASS: byte-identical round trip", flush=True)


if __name__ == "__main__":
    main()
