from __future__ import annotations

import struct
import bz2
from pathlib import Path

import brotli
import numpy as np

from encode_seg_c2split_purepy import N_CLASSES, compute_spatial_contexts, decode_mask_payload, make_remap_tables
from encode_seg_c2split_tr_purepy import compute_tr
from range_coder import RangeDecoder
from seg_sparse_m10_codec import (
    FEAT_DIAG_TLTL,
    FEAT_DIAG_TRTR,
    FEAT_LEFT_LEFT,
    FEAT_PREV2_LEFT,
    FEAT_PREV2_RIGHT,
    FEAT_PREV_BOTTOM,
    FEAT_PREV_BOTTOM2,
    FEAT_PREV_BOTTOM_LEFT,
    FEAT_PREV_BOTTOM_RIGHT,
    FEAT_PREV_LEFT,
    FEAT_PREV_PREV_PREV,
    FEAT_PREV_RIGHT,
    FEAT_PREV_RIGHT2,
    FEAT_PREV_TOP_RIGHT,
    FEAT_PREV_TOP,
    FEAT_PEEL_DIST42,
    FEAT_PEEL_BOUND5,
    FEAT_PEEL_SLOPE5,
    FEAT_TOP_TOP_TOP,
    FEAT_X_BIN5,
    FEAT_X_BIN5_SHIFT,
    FEAT_Y_BIN5,
    M10_VERSION,
    N_SYM,
    unpack_sparse_m10,
)

MAGIC = b"QSM5\0"
MAGIC_SHIFT = b"QSM5S\0"
MAGIC_SHIFT_BIG = b"QSM5S7\0"
MAGIC_SHIFT_BIG3 = b"QSM5S8\0"
MAGIC_SHIFT_BIG4 = b"QSM5S9\0"
MAGIC_SHIFT_BIG5 = b"QSM5SA\0"
MAGIC_TOPBAND = b"QTBM1\0"
MAGIC_TOPBAND2 = b"QTBM2\0"
MAGIC_TOPBAND3 = b"QTBM3\0"
MAGIC_TOPBAND4 = b"QTBM4\0"
MAGIC_TOPBAND5 = b"QTBM5\0"
BINARY_MASK_FORMAT = 255
BOUNDARY_MASK_FORMAT = 254
BINARY_FEATURES = {
    0: "top",
    1: "left",
    2: "tl",
    3: "tr",
    4: "tt",
    5: "ll",
    6: "prev",
    7: "prev2",
    8: "prev_bottom",
    9: "prev_right",
    10: "prev_left",
    11: "prev_top",
    12: "prev_bottom_right",
    13: "prev_right2",
    14: "prev_bottom2",
}


def _m5_ctx(top_v: int, left_v: int, tl_v: int, tr_v: int, prev_v: int) -> int:
    return ((((top_v * 5 + left_v) * 5 + tl_v) * 5 + tr_v) * 5 + prev_v)


def _binary_ctx_frame(mask: np.ndarray, fi: int, feat_ids: tuple[int, ...]) -> np.ndarray:
    frame = mask[fi]
    h, w = frame.shape
    out = np.zeros((h, w), dtype=np.int64)
    for fid in feat_ids:
        name = BINARY_FEATURES[fid]
        feat = np.zeros((h, w), dtype=np.int64)
        if name == "top":
            feat[1:] = frame[:-1]
        elif name == "left":
            feat[:, 1:] = frame[:, :-1]
        elif name == "tl":
            feat[1:, 1:] = frame[:-1, :-1]
        elif name == "tr":
            feat[1:, :-1] = frame[:-1, 1:]
        elif name == "tt":
            feat[2:] = frame[:-2]
        elif name == "ll":
            feat[:, 2:] = frame[:, :-2]
        elif name == "prev":
            if fi >= 1:
                feat = mask[fi - 1].astype(np.int64)
        elif name == "prev2":
            if fi >= 2:
                feat = mask[fi - 2].astype(np.int64)
        elif name == "prev_bottom":
            if fi >= 1:
                feat[:-1] = mask[fi - 1, 1:]
        elif name == "prev_right":
            if fi >= 1:
                feat[:, :-1] = mask[fi - 1, :, 1:]
        elif name == "prev_left":
            if fi >= 1:
                feat[:, 1:] = mask[fi - 1, :, :-1]
        elif name == "prev_top":
            if fi >= 1:
                feat[1:] = mask[fi - 1, :-1]
        else:
            raise ValueError(name)
        out = out * 2 + feat
    return out


def decode_binary_mask_payload(payload: bytes, n_pairs: int, h: int, w: int) -> np.ndarray:
    if payload[:5] != b"QBM1\0":
        raise ValueError("bad QBM1 mask payload")
    pos = 5
    precision, n0, n = struct.unpack_from("<BBB", payload, pos)
    pos += 3
    feat0 = tuple(payload[pos : pos + n0])
    pos += n0
    feat = tuple(payload[pos : pos + n])
    pos += n
    freq0_size, freq_size, bs_len = struct.unpack_from("<HHI", payload, pos)
    pos += struct.calcsize("<HHI")
    freq0 = np.frombuffer(payload, dtype="<u2", count=freq0_size // 2, offset=pos).reshape(2 ** n0, 2).copy()
    pos += freq0_size
    freq = np.frombuffer(payload, dtype="<u2", count=freq_size // 2, offset=pos).reshape(2 ** n, 2).copy()
    pos += freq_size
    bitstream = payload[pos : pos + bs_len]

    cdf0 = np.zeros((2 ** n0, 3), dtype=np.int64)
    cdf0[:, 1:] = np.cumsum(freq0.astype(np.int64), axis=1)
    cdf = np.zeros((2 ** n, 3), dtype=np.int64)
    cdf[:, 1:] = np.cumsum(freq.astype(np.int64), axis=1)
    dec = RangeDecoder(bitstream)
    total = 1 << precision
    out = np.zeros((n_pairs, h, w), dtype=np.uint8)
    for fi in range(n_pairs):
        feats = feat0 if fi == 0 else feat
        rows = cdf0 if fi == 0 else cdf
        for y in range(h):
            for x in range(w):
                ctx = 0
                for fid in feats:
                    name = BINARY_FEATURES[fid]
                    if name == "top":
                        fv = int(out[fi, y - 1, x]) if y > 0 else 0
                    elif name == "left":
                        fv = int(out[fi, y, x - 1]) if x > 0 else 0
                    elif name == "tl":
                        fv = int(out[fi, y - 1, x - 1]) if (y > 0 and x > 0) else 0
                    elif name == "tr":
                        fv = int(out[fi, y - 1, x + 1]) if (y > 0 and x + 1 < w) else 0
                    elif name == "tt":
                        fv = int(out[fi, y - 2, x]) if y > 1 else 0
                    elif name == "ll":
                        fv = int(out[fi, y, x - 2]) if x > 1 else 0
                    elif name == "prev":
                        fv = int(out[fi - 1, y, x]) if fi >= 1 else 0
                    elif name == "prev2":
                        fv = int(out[fi - 2, y, x]) if fi >= 2 else 0
                    elif name == "prev_bottom":
                        fv = int(out[fi - 1, y + 1, x]) if (fi >= 1 and y + 1 < h) else 0
                    elif name == "prev_right":
                        fv = int(out[fi - 1, y, x + 1]) if (fi >= 1 and x + 1 < w) else 0
                    elif name == "prev_left":
                        fv = int(out[fi - 1, y, x - 1]) if (fi >= 1 and x >= 1) else 0
                    elif name == "prev_top":
                        fv = int(out[fi - 1, y - 1, x]) if (fi >= 1 and y >= 1) else 0
                    elif name == "prev_bottom_right":
                        fv = int(out[fi - 1, y + 1, x + 1]) if (fi >= 1 and y + 1 < h and x + 1 < w) else 0
                    elif name == "prev_right2":
                        fv = int(out[fi - 1, y, x + 2]) if (fi >= 1 and x + 2 < w) else 0
                    elif name == "prev_bottom2":
                        fv = int(out[fi - 1, y + 2, x]) if (fi >= 1 and y + 2 < h) else 0
                    else:
                        raise ValueError(name)
                    ctx = ctx * 2 + fv
                row = rows[ctx]
                target = dec.decode_target(total)
                sym = 0 if row[1] > target else 1
                out[fi, y, x] = sym
                dec.advance(int(row[sym]), int(row[sym + 1]), total)
        if fi % 50 == 0 and fi:
            print(f"  decoded binary mask frame {fi}/{n_pairs}", flush=True)
    return out


def _leb128_decode_big_deltas(buf: bytes, pos: int, count: int) -> tuple[np.ndarray, int]:
    deltas = np.empty(count, dtype=np.int64)
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


def decode_boundary_mask_payload(payload: bytes, n_pairs: int, h: int, w: int) -> np.ndarray:
    if payload[:5] == b"QBD1\0":
        pos = 5
        first_len, dx_len, err_len, err_count = struct.unpack_from("<IIII", payload, pos)
        pos += 16
        first = np.frombuffer(bz2.decompress(payload[pos : pos + first_len]), dtype="<u2", count=n_pairs).astype(np.int16)
        pos += first_len
        dx = np.frombuffer(bz2.decompress(payload[pos : pos + dx_len]), dtype=np.int8, count=n_pairs * (w - 1)).reshape(n_pairs, w - 1)
        pos += dx_len
        err_raw = brotli.decompress(payload[pos : pos + err_len])
    elif payload[:5] == b"QBD2\0":
        pos = 5
        bins, dx_nsym, dx_offset = struct.unpack_from("<BBB", payload, pos)
        pos += 3
        first_len, dx_len, err_len, err_count = struct.unpack_from("<IIII", payload, pos)
        pos += 16
        first = np.frombuffer(bz2.decompress(payload[pos : pos + first_len]), dtype="<u2", count=n_pairs).astype(np.int16)
        pos += first_len
        freqs = np.frombuffer(payload, dtype="<u2", count=bins * dx_nsym, offset=pos).astype(np.int64).reshape(bins, dx_nsym)
        pos += bins * dx_nsym * 2
        bitstream = payload[pos : pos + dx_len]
        pos += dx_len
        err_raw = brotli.decompress(payload[pos : pos + err_len])
        cdf = np.zeros((bins, dx_nsym + 1), dtype=np.int64)
        cdf[:, 1:] = np.cumsum(freqs, axis=1)
        total = int(cdf[0, -1])
        dec = RangeDecoder(bitstream)
        dx = np.empty((n_pairs, w - 1), dtype=np.int16)
        for fi in range(n_pairs):
            for x in range(w - 1):
                row = cdf[(x * bins) // (w - 1)]
                target = dec.decode_target(total)
                sym = int(np.searchsorted(row, target, side="right") - 1)
                dx[fi, x] = sym - int(dx_offset)
                dec.advance(int(row[sym]), int(row[sym + 1]), total)
    else:
        raise ValueError("bad QBD mask payload")

    bounds = np.empty((n_pairs, w), dtype=np.int16)
    bounds[:, 0] = first
    bounds[:, 1:] = first[:, None] + np.cumsum(dx.astype(np.int16), axis=1)
    yy = np.arange(h, dtype=np.int16)[None, :, None]
    out = (yy >= bounds[:, None, :]).astype(np.uint8)

    if err_count:
        deltas, used = _leb128_decode_big_deltas(err_raw, 0, err_count)
        if used != len(err_raw):
            raise ValueError("trailing QBD1 error bytes")
        idx = np.cumsum(deltas, dtype=np.int64) - 1
        flat = out.reshape(-1)
        flat[idx] ^= 1
    return out


def decode_topband_payload(payload: bytes, n_pairs: int, h: int, w: int) -> np.ndarray:
    if payload[:5] not in (b"QTB1\0", b"QTB2\0", b"QTB3\0", b"QTBZ\0"):
        raise ValueError("bad QTB1 top-band payload")
    version = payload[:5]
    pos = 5
    if version == b"QTBZ\0":
        n2, w2, bins, bounds_len = struct.unpack_from("<HHHI", payload, pos)
        pos += struct.calcsize("<HHHI")
        bounds_raw = bz2.decompress(payload[pos : pos + bounds_len])
        pos += bounds_len
        bounds = np.frombuffer(bounds_raw, dtype="<u2", count=n_pairs * bins).reshape(n_pairs, bins)
        err_len = 0
    elif version == b"QTB2\0":
        n2, w2, bins, err_len = struct.unpack_from("<HHHI", payload, pos)
        pos += struct.calcsize("<HHHI")
        bounds_dtype = "<u2"
    elif version == b"QTB3\0":
        n2, w2, bins = struct.unpack_from("<HHH", payload, pos)
        err_len = 0
        bounds_dtype = "u1"
        pos += struct.calcsize("<HHH")
    else:
        n2, w2, bins = struct.unpack_from("<HHH", payload, pos)
        err_len = 0
        bounds_dtype = "<u2"
        pos += struct.calcsize("<HHH")
    if n2 != n_pairs or w2 != w:
        raise ValueError("top-band dimensions do not match stream")
    if version != b"QTBZ\0":
        bounds = np.frombuffer(payload, dtype=bounds_dtype, count=n_pairs * bins, offset=pos).reshape(n_pairs, bins)
        pos += bounds.size * bounds.dtype.itemsize
    out = np.zeros((n_pairs, h, w), dtype=np.uint8)
    for b in range(bins):
        x0 = (b * w) // bins
        x1 = ((b + 1) * w) // bins
        for fi in range(n_pairs):
            out[fi, : int(bounds[fi, b]), x0:x1] = 1
    if err_len:
        err_raw = brotli.decompress(payload[pos : pos + err_len])
        (err_count,) = struct.unpack_from("<I", err_raw, 0)
        deltas, used = _leb128_decode_big_deltas(err_raw, 4, err_count)
        if used != len(err_raw):
            raise ValueError("trailing QTB2 exception bytes")
        idx = np.cumsum(deltas, dtype=np.int64) - 1
        out.reshape(-1)[idx] = 0
    return out


def unpack_sparse_big(compressed: bytes, precision: int) -> tuple[np.ndarray, np.ndarray]:
    import zlib

    raw = zlib.decompress(compressed, -15)
    (n_fired,) = struct.unpack_from("<I", raw, 0)
    if n_fired == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, N_SYM), dtype=np.uint16)
    pos = 4
    deltas, pos = _leb128_decode_big_deltas(raw, pos, n_fired)
    partial = np.frombuffer(raw, dtype="<u2", count=n_fired * (N_SYM - 1), offset=pos).reshape(n_fired, N_SYM - 1).copy()
    last = (1 << precision) - partial.astype(np.int64).sum(axis=1)
    if (last < 0).any() or (last > 0xFFFF).any():
        raise ValueError("invalid sparse-big frequency row")
    freqs = np.empty((n_fired, N_SYM), dtype=np.uint16)
    freqs[:, : N_SYM - 1] = partial
    freqs[:, N_SYM - 1] = last.astype(np.uint16)
    fired_idx = (np.cumsum(deltas, dtype=np.int64) - 1).astype(np.int64)
    return fired_idx, freqs


def unpack_sparse_big_plain(raw: bytes, precision: int) -> tuple[np.ndarray, np.ndarray]:
    (n_fired,) = struct.unpack_from("<I", raw, 0)
    if n_fired == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, N_SYM), dtype=np.uint16)
    pos = 4
    deltas, pos = _leb128_decode_big_deltas(raw, pos, n_fired)
    partial = np.frombuffer(raw, dtype="<u2", count=n_fired * (N_SYM - 1), offset=pos).reshape(n_fired, N_SYM - 1).copy()
    last = (1 << precision) - partial.astype(np.int64).sum(axis=1)
    if (last < 0).any() or (last > 0xFFFF).any():
        raise ValueError("invalid sparse-big-plain frequency row")
    freqs = np.empty((n_fired, N_SYM), dtype=np.uint16)
    freqs[:, : N_SYM - 1] = partial
    freqs[:, N_SYM - 1] = last.astype(np.uint16)
    fired_idx = (np.cumsum(deltas, dtype=np.int64) - 1).astype(np.int64)
    return fired_idx, freqs


def unpack_sparse_big_plain_colsfirst(raw: bytes, precision: int) -> tuple[np.ndarray, np.ndarray]:
    (n_fired,) = struct.unpack_from("<I", raw, 0)
    if n_fired == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, N_SYM), dtype=np.uint16)
    pos = 4
    partial = np.frombuffer(raw, dtype="<u2", count=n_fired * (N_SYM - 1), offset=pos).reshape(N_SYM - 1, n_fired).T.copy()
    pos += n_fired * (N_SYM - 1) * 2
    deltas, pos = _leb128_decode_big_deltas(raw, pos, n_fired)
    if pos != len(raw):
        raise ValueError("trailing sparse-big-plain-colsfirst bytes")
    last = (1 << precision) - partial.astype(np.int64).sum(axis=1)
    if (last < 0).any() or (last > 0xFFFF).any():
        raise ValueError("invalid sparse-big-plain-colsfirst frequency row")
    freqs = np.empty((n_fired, N_SYM), dtype=np.uint16)
    freqs[:, : N_SYM - 1] = partial
    freqs[:, N_SYM - 1] = last.astype(np.uint16)
    fired_idx = (np.cumsum(deltas, dtype=np.int64) - 1).astype(np.int64)
    return fired_idx, freqs


def _decode_frame_m5fallback(
    dec: RangeDecoder,
    frame: np.ndarray,
    prev_frame: np.ndarray,
    prev_prev_frame: np.ndarray,
    prev_prev_prev_frame: np.ndarray | None,
    mask_frame: np.ndarray,
    m5_cdf_flat_py: list,
    fired_cdf_flat_py: list,
    m12_to_slot_dict: dict[int, int],
    feat_ids: tuple[int, ...],
    total: int,
    h: int,
    w: int,
    peel_class: int,
    inv_remap_py: list,
    shift_dy: int = 0,
    shift_dx: int = 0,
) -> None:
    frame_list = [[0] * w for _ in range(h)]
    mask_list = mask_frame.tolist()
    peel_bounds = [h] * w
    for bx in range(w):
        if mask_list[h - 1][bx]:
            by = h - 1
            while by >= 0 and mask_list[by][bx]:
                by -= 1
            peel_bounds[bx] = by + 1
    prev_list = prev_frame.tolist()
    pp_list = prev_prev_frame.tolist()
    ppp_list = prev_prev_prev_frame.tolist() if prev_prev_prev_frame is not None else None
    inv = inv_remap_py
    get_target = dec.decode_target
    advance = dec.advance

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

            top_v = frame_list[y - 1][x] if y > 0 else 0
            left_v = frame_list[y][x - 1] if x > 0 else 0
            tl_v = frame_list[y - 1][x - 1] if (y > 0 and x > 0) else 0
            tr_v = frame_list[y - 1][x + 1] if (y > 0 and x + 1 < w) else 0
            sy = y + shift_dy
            sx = x + shift_dx
            if 0 <= sy < h and 0 <= sx < w:
                prev_v = prev_list[sy][sx]
                pp_v = pp_list[sy][sx]
            else:
                prev_v = 0
                pp_v = 0
            tt_v = frame_list[y - 2][x] if y > 1 else 0

            m5_ctx = _m5_ctx(top_v, left_v, tl_v, tr_v, prev_v)
            m12_ctx = ((m5_ctx * 5 + pp_v) * 5 + tt_v)
            for fid in feat_ids:
                if fid == FEAT_DIAG_TLTL:
                    fv = frame_list[y - 2][x - 2] if (y >= 2 and x >= 2) else 0
                elif fid == FEAT_LEFT_LEFT:
                    fv = frame_list[y][x - 2] if x >= 2 else 0
                elif fid == FEAT_TOP_TOP_TOP:
                    fv = frame_list[y - 3][x] if y >= 3 else 0
                elif fid == FEAT_PREV_PREV_PREV:
                    fv = ppp_row[x] if ppp_row is not None else 0
                elif fid == FEAT_DIAG_TRTR:
                    fv = frame_list[y - 2][x + 2] if (y >= 2 and x + 2 < w) else 0
                elif fid == FEAT_PREV_LEFT:
                    fv = prev_row[x - 1] if x >= 1 else 0
                elif fid == FEAT_PREV_RIGHT:
                    fv = prev_row[x + 1] if x + 1 < w else 0
                elif fid == FEAT_PREV_TOP:
                    fv = prev_row_above[x] if prev_row_above is not None else 0
                elif fid == FEAT_PREV_BOTTOM:
                    fv = prev_row_below[x] if prev_row_below is not None else 0
                elif fid == FEAT_PREV2_LEFT:
                    fv = pp_row[x - 1] if x >= 1 else 0
                elif fid == FEAT_PREV2_RIGHT:
                    fv = pp_row[x + 1] if x + 1 < w else 0
                elif fid == FEAT_PREV_BOTTOM_RIGHT:
                    fv = prev_row_below[x + 1] if (prev_row_below is not None and x + 1 < w) else 0
                elif fid == FEAT_PREV_BOTTOM_LEFT:
                    fv = prev_row_below[x - 1] if (prev_row_below is not None and x >= 1) else 0
                elif fid == FEAT_PREV_TOP_RIGHT:
                    fv = prev_row_above[x + 1] if (prev_row_above is not None and x + 1 < w) else 0
                elif fid == FEAT_PREV_BOTTOM2:
                    fv = prev_list[y + 2][x] if y + 2 < h else 0
                elif fid == FEAT_PREV_RIGHT2:
                    fv = prev_row[x + 2] if x + 2 < w else 0
                elif fid == FEAT_X_BIN5:
                    fv = (x * 5) // w
                elif fid == FEAT_Y_BIN5:
                    fv = (y * 5) // h
                elif fid == FEAT_X_BIN5_SHIFT:
                    fv = min(((x + w // 10) * 5) // w, 4)
                elif fid == FEAT_PEEL_DIST42:
                    d = peel_bounds[x] - y
                    fv = 0 if d <= 0 else min(((d - 1) // 42) + 1, 4)
                elif fid == FEAT_PEEL_BOUND5:
                    fv = min((peel_bounds[x] * 5) // h, 4)
                elif fid == FEAT_PEEL_SLOPE5:
                    prev_bound = peel_bounds[x - 1] if x >= 1 else peel_bounds[x]
                    fv = max(0, min(peel_bounds[x] - prev_bound, 2) + 2)
                else:
                    raise ValueError(fid)
                m12_ctx = m12_ctx * 5 + fv

            slot = m12_to_slot_dict.get(m12_ctx)
            cdf_row = fired_cdf_flat_py[slot] if slot is not None else m5_cdf_flat_py[m5_ctx]
            t = get_target(total)
            s = 0
            while cdf_row[s + 1] <= t:
                s += 1
            frame_list[y][x] = inv[s]
            advance(cdf_row[s], cdf_row[s + 1], total)
    for y in range(h):
        frame[y] = frame_list[y]


def _decode_frame_topband(
    dec: RangeDecoder,
    frame: np.ndarray,
    prev_frame: np.ndarray | None,
    prev_prev_frame: np.ndarray | None,
    prev_prev_prev_frame: np.ndarray | None,
    top_support_frame: np.ndarray,
    road_frame: np.ndarray,
    spatial_cdf_flat_py: list,
    m5_cdf_flat_py: list,
    fired_cdf_flat_py: list,
    m12_to_slot_dict: dict[int, int],
    feat_ids: tuple[int, ...],
    total: int,
    h: int,
    w: int,
    shift_dy: int,
    shift_dx: int,
    inv: list[int],
) -> None:
    frame_list = [[0] * w for _ in range(h)]
    top_support = top_support_frame.tolist()
    road = road_frame.tolist()
    prev_list = prev_frame.tolist() if prev_frame is not None else None
    pp_list = prev_prev_frame.tolist() if prev_prev_frame is not None else None
    ppp_list = prev_prev_prev_frame.tolist() if prev_prev_prev_frame is not None else None
    get_target = dec.decode_target
    advance = dec.advance

    road_bounds = [h] * w
    for bx in range(w):
        if road[h - 1][bx]:
            by = h - 1
            while by >= 0 and road[by][bx]:
                by -= 1
            road_bounds[bx] = by + 1

    for y in range(h):
        prev_row = prev_list[y] if prev_list is not None else None
        pp_row = pp_list[y] if pp_list is not None else None
        prev_row_above = prev_list[y - 1] if (prev_list is not None and y > 0) else None
        prev_row_below = prev_list[y + 1] if (prev_list is not None and y + 1 < h) else None
        ppp_row = ppp_list[y] if ppp_list is not None else None
        for x in range(w):
            if top_support[y][x]:
                frame_list[y][x] = 2
                continue
            if road[y][x]:
                frame_list[y][x] = 4
                continue

            top_v = frame_list[y - 1][x] if y > 0 else 0
            left_v = frame_list[y][x - 1] if x > 0 else 0
            tl_v = frame_list[y - 1][x - 1] if (y > 0 and x > 0) else 0
            tr_v = frame_list[y - 1][x + 1] if (y > 0 and x + 1 < w) else 0
            if prev_list is None:
                cdf_row = spatial_cdf_flat_py[((top_v * 5 + left_v) * 5 + tl_v) * 5 + tr_v]
            else:
                sy = y + shift_dy
                sx = x + shift_dx
                if 0 <= sy < h and 0 <= sx < w:
                    prev_v = prev_list[sy][sx]
                    pp_v = pp_list[sy][sx] if pp_list is not None else 0
                else:
                    prev_v = 0
                    pp_v = 0
                tt_v = frame_list[y - 2][x] if y > 1 else 0
                m5_ctx = _m5_ctx(top_v, left_v, tl_v, tr_v, prev_v)
                cdf_row = m5_cdf_flat_py[m5_ctx]
                if pp_list is not None and fired_cdf_flat_py:
                    m12_ctx = ((m5_ctx * 5 + pp_v) * 5 + tt_v)
                    for fid in feat_ids:
                        if fid == FEAT_DIAG_TLTL:
                            fv = frame_list[y - 2][x - 2] if (y >= 2 and x >= 2) else 0
                        elif fid == FEAT_LEFT_LEFT:
                            fv = frame_list[y][x - 2] if x >= 2 else 0
                        elif fid == FEAT_TOP_TOP_TOP:
                            fv = frame_list[y - 3][x] if y >= 3 else 0
                        elif fid == FEAT_PREV_PREV_PREV:
                            fv = ppp_row[x] if ppp_row is not None else 0
                        elif fid == FEAT_DIAG_TRTR:
                            fv = frame_list[y - 2][x + 2] if (y >= 2 and x + 2 < w) else 0
                        elif fid == FEAT_PREV_LEFT:
                            fv = prev_row[x - 1] if (prev_row is not None and x >= 1) else 0
                        elif fid == FEAT_PREV_RIGHT:
                            fv = prev_row[x + 1] if (prev_row is not None and x + 1 < w) else 0
                        elif fid == FEAT_PREV_TOP:
                            fv = prev_row_above[x] if prev_row_above is not None else 0
                        elif fid == FEAT_PREV_BOTTOM:
                            fv = prev_row_below[x] if prev_row_below is not None else 0
                        elif fid == FEAT_PREV2_LEFT:
                            fv = pp_row[x - 1] if (pp_row is not None and x >= 1) else 0
                        elif fid == FEAT_PREV2_RIGHT:
                            fv = pp_row[x + 1] if (pp_row is not None and x + 1 < w) else 0
                        elif fid == FEAT_PREV_BOTTOM_RIGHT:
                            fv = prev_row_below[x + 1] if (prev_row_below is not None and x + 1 < w) else 0
                        elif fid == FEAT_PREV_BOTTOM_LEFT:
                            fv = prev_row_below[x - 1] if (prev_row_below is not None and x >= 1) else 0
                        elif fid == FEAT_PREV_TOP_RIGHT:
                            fv = prev_row_above[x + 1] if (prev_row_above is not None and x + 1 < w) else 0
                        elif fid == FEAT_PREV_BOTTOM2:
                            fv = prev_list[y + 2][x] if (prev_list is not None and y + 2 < h) else 0
                        elif fid == FEAT_PREV_RIGHT2:
                            fv = prev_row[x + 2] if (prev_row is not None and x + 2 < w) else 0
                        elif fid == FEAT_X_BIN5:
                            fv = (x * 5) // w
                        elif fid == FEAT_Y_BIN5:
                            fv = (y * 5) // h
                        elif fid == FEAT_X_BIN5_SHIFT:
                            fv = min(((x + w // 10) * 5) // w, 4)
                        elif fid == FEAT_PEEL_DIST42:
                            d = road_bounds[x] - y
                            fv = 0 if d <= 0 else min(((d - 1) // 42) + 1, 4)
                        elif fid == FEAT_PEEL_BOUND5:
                            fv = min((road_bounds[x] * 5) // h, 4)
                        elif fid == FEAT_PEEL_SLOPE5:
                            prev_bound = road_bounds[x - 1] if x >= 1 else road_bounds[x]
                            fv = max(0, min(road_bounds[x] - prev_bound, 2) + 2)
                        else:
                            raise ValueError(fid)
                        m12_ctx = m12_ctx * 5 + fv
                    slot = m12_to_slot_dict.get(m12_ctx)
                    if slot is not None:
                        cdf_row = fired_cdf_flat_py[slot]

            t = get_target(total)
            s = 0
            while cdf_row[s + 1] <= t:
                s += 1
            frame_list[y][x] = inv[s]
            advance(cdf_row[s], cdf_row[s + 1], total)
    for y in range(h):
        frame[y] = frame_list[y]


def _decode_frame_m5_shift(
    dec: RangeDecoder,
    frame: np.ndarray,
    prev_frame: np.ndarray,
    mask_frame: np.ndarray,
    m5_cdf_flat_py: list,
    total: int,
    h: int,
    w: int,
    peel_class: int,
    inv_remap_py: list,
    shift_dy: int,
    shift_dx: int,
) -> None:
    frame_list = [[0] * w for _ in range(h)]
    mask_list = mask_frame.tolist()
    prev_list = prev_frame.tolist()
    inv = inv_remap_py
    get_target = dec.decode_target
    advance = dec.advance
    for y in range(h):
        for x in range(w):
            if mask_list[y][x]:
                frame_list[y][x] = peel_class
                continue
            top_v = frame_list[y - 1][x] if y > 0 else 0
            left_v = frame_list[y][x - 1] if x > 0 else 0
            tl_v = frame_list[y - 1][x - 1] if (y > 0 and x > 0) else 0
            tr_v = frame_list[y - 1][x + 1] if (y > 0 and x + 1 < w) else 0
            sy = y + shift_dy
            sx = x + shift_dx
            prev_v = prev_list[sy][sx] if (0 <= sy < h and 0 <= sx < w) else 0
            cdf_row = m5_cdf_flat_py[_m5_ctx(top_v, left_v, tl_v, tr_v, prev_v)]
            t = get_target(total)
            s = 0
            while cdf_row[s + 1] <= t:
                s += 1
            frame_list[y][x] = inv[s]
            advance(cdf_row[s], cdf_row[s + 1], total)
    for y in range(h):
        frame[y] = frame_list[y]


def decode_seg_split_m5(path: "Path | str") -> np.ndarray:
    blob = Path(path).read_bytes()
    pos = 0
    shifted = False
    big_sparse = False
    compact_tables = False
    sparse_dense_tables = False
    sparse_dense_plain = False
    shift_dy = 0
    shift_dx = 0
    if blob[: len(MAGIC_SHIFT_BIG5)] == MAGIC_SHIFT_BIG5:
        shifted = True
        big_sparse = True
        sparse_dense_tables = True
        sparse_dense_plain = True
        pos += len(MAGIC_SHIFT_BIG5)
    elif blob[: len(MAGIC_SHIFT_BIG4)] == MAGIC_SHIFT_BIG4:
        shifted = True
        big_sparse = True
        sparse_dense_tables = True
        sparse_dense_plain = False
        pos += len(MAGIC_SHIFT_BIG4)
    elif blob[: len(MAGIC_SHIFT_BIG3)] == MAGIC_SHIFT_BIG3:
        shifted = True
        big_sparse = True
        compact_tables = True
        sparse_dense_plain = False
        pos += len(MAGIC_SHIFT_BIG3)
    elif blob[: len(MAGIC_SHIFT_BIG)] == MAGIC_SHIFT_BIG:
        shifted = True
        big_sparse = True
        pos += len(MAGIC_SHIFT_BIG)
    elif blob[: len(MAGIC_SHIFT)] == MAGIC_SHIFT:
        shifted = True
        pos += len(MAGIC_SHIFT)
    elif blob[: len(MAGIC)] == MAGIC:
        pos += len(MAGIC)
    else:
        raise ValueError("bad QSM5 magic")
    if shifted:
        n_pairs, h, w, precision, peel_class, mask_format, shift_dy, shift_dx = struct.unpack_from("<HHHBBBbb", blob, pos)
        pos += struct.calcsize("<HHHBBBbb")
    else:
        n_pairs, h, w, precision, peel_class, mask_format = struct.unpack_from("<HHHBBB", blob, pos)
        pos += struct.calcsize("<HHHBBB")
    (mask_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    mask_payload = blob[pos : pos + mask_len]
    pos += mask_len
    spatial_size, m5_size = struct.unpack_from("<HH", blob, pos)
    pos += struct.calcsize("<HH")
    if sparse_dense_tables:
        unpack_dense = unpack_sparse_big_plain if sparse_dense_plain else unpack_sparse_big
        spatial_idx, spatial_rows = unpack_dense(blob[pos : pos + spatial_size], precision=precision)
        pos += spatial_size
        m5_idx, m5_rows = unpack_dense(blob[pos : pos + m5_size], precision=precision)
        pos += m5_size
        total = 1 << precision
        default = np.array([1, 1, 1, total - 3], dtype=np.uint16)
        spatial_freqs = np.broadcast_to(default, (5**4, N_SYM)).copy()
        m5_freqs = np.broadcast_to(default, (5**5, N_SYM)).copy()
        spatial_freqs[spatial_idx] = spatial_rows
        m5_freqs[m5_idx] = m5_rows
        spatial_freqs = spatial_freqs.reshape((N_CLASSES,) * 4 + (N_SYM,))
        m5_freqs = m5_freqs.reshape((N_CLASSES,) * 5 + (N_SYM,))
    elif compact_tables:
        spatial_part = np.frombuffer(blob, dtype="<u2", count=spatial_size // 2, offset=pos).reshape(5**4, N_SYM - 1).copy()
        pos += spatial_size
        m5_part = np.frombuffer(blob, dtype="<u2", count=m5_size // 2, offset=pos).reshape(5**5, N_SYM - 1).copy()
        pos += m5_size
        total = 1 << precision
        spatial_last = total - spatial_part.astype(np.int64).sum(axis=1)
        m5_last = total - m5_part.astype(np.int64).sum(axis=1)
        if (spatial_last < 0).any() or (m5_last < 0).any():
            raise ValueError("invalid compact frequency table")
        spatial_freqs = np.empty((5**4, N_SYM), dtype=np.uint16)
        spatial_freqs[:, : N_SYM - 1] = spatial_part
        spatial_freqs[:, N_SYM - 1] = spatial_last.astype(np.uint16)
        spatial_freqs = spatial_freqs.reshape((N_CLASSES,) * 4 + (N_SYM,))
        m5_freqs = np.empty((5**5, N_SYM), dtype=np.uint16)
        m5_freqs[:, : N_SYM - 1] = m5_part
        m5_freqs[:, N_SYM - 1] = m5_last.astype(np.uint16)
        m5_freqs = m5_freqs.reshape((N_CLASSES,) * 5 + (N_SYM,))
    else:
        spatial_freqs = np.frombuffer(blob, dtype="<u2", count=spatial_size // 2, offset=pos).reshape((N_CLASSES,) * 4 + (N_SYM,)).copy()
        pos += spatial_size
        m5_freqs = np.frombuffer(blob, dtype="<u2", count=m5_size // 2, offset=pos).reshape((N_CLASSES,) * 5 + (N_SYM,)).copy()
        pos += m5_size
    n_feats = blob[pos]
    pos += 1
    feat_ids = tuple(blob[pos : pos + n_feats])
    pos += n_feats
    _thr_q8, sparse_len = struct.unpack_from("<HI", blob, pos)
    pos += 6
    sparse = blob[pos : pos + sparse_len]
    pos += sparse_len
    (bs_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    bitstream = blob[pos : pos + bs_len]

    if mask_format == BINARY_MASK_FORMAT:
        mask = decode_binary_mask_payload(mask_payload, n_pairs, h, w)
    elif mask_format == BOUNDARY_MASK_FORMAT:
        mask = decode_boundary_mask_payload(mask_payload, n_pairs, h, w)
    else:
        mask = decode_mask_payload(mask_payload, mask_format, n_pairs, h, w)
    _, inverse = make_remap_tables(peel_class)
    inv_remap_py = inverse.tolist()
    total = 1 << precision

    spatial_cdf = np.zeros((N_CLASSES,) * 4 + (N_SYM + 1,), dtype=np.int64)
    spatial_cdf[..., 1:] = np.cumsum(spatial_freqs.astype(np.int64), axis=-1)
    spatial_py = spatial_cdf.tolist()
    m5_cdf_flat = np.zeros((5 ** 5, N_SYM + 1), dtype=np.int64)
    m5_cdf_flat[:, 1:] = np.cumsum(m5_freqs.reshape(5 ** 5, N_SYM).astype(np.int64), axis=-1)
    m5_py = m5_cdf_flat.reshape((N_CLASSES,) * 5 + (N_SYM + 1,)).tolist()
    m5_flat_py = m5_cdf_flat.tolist()

    if big_sparse:
        fired_idx, fired_freqs = unpack_sparse_big(sparse, precision=precision)
    else:
        fired_idx, fired_freqs, _n_ctx = unpack_sparse_m10(sparse, version=M10_VERSION, precision=precision)
    fired_cdf_flat_py: list = []
    if fired_idx.size > 0:
        fired_cdf = np.zeros((fired_idx.size, N_SYM + 1), dtype=np.int64)
        fired_cdf[:, 1:] = np.cumsum(fired_freqs.astype(np.int64), axis=-1)
        fired_cdf_flat_py = fired_cdf.tolist()
    m12_to_slot = {int(ctx): i for i, ctx in enumerate(fired_idx.tolist())}

    from seg_sparse_m10_codec import _decode_frame_m4, _decode_frame_m5

    dec = RangeDecoder(bitstream)
    out = np.zeros((n_pairs, h, w), dtype=np.uint8)
    _decode_frame_m4(dec, out[0], mask[0], spatial_py, total, h, w, peel_class, inv_remap_py)
    if n_pairs >= 2:
        if shifted:
            _decode_frame_m5_shift(
                dec,
                out[1],
                out[0],
                mask[1],
                m5_flat_py,
                total,
                h,
                w,
                peel_class,
                inv_remap_py,
                shift_dy,
                shift_dx,
            )
        else:
            _decode_frame_m5(dec, out[1], out[0], mask[1], m5_py, total, h, w, peel_class, inv_remap_py)
    for fi in range(2, n_pairs):
        ppp = out[fi - 3] if fi >= 3 else None
        _decode_frame_m5fallback(
            dec,
            out[fi],
            out[fi - 1],
            out[fi - 2],
            ppp,
            mask[fi],
            m5_flat_py,
            fired_cdf_flat_py,
            m12_to_slot,
            feat_ids,
            total,
            h,
            w,
            peel_class,
            inv_remap_py,
            shift_dy if shifted else 0,
            shift_dx if shifted else 0,
        )
        if fi % 50 == 0:
            print(f"  decoded frame {fi}/{n_pairs}", flush=True)
    return out


def decode_seg_topband(path: "Path | str") -> np.ndarray:
    blob = Path(path).read_bytes()
    pos = 0
    sparse_colsfirst = False
    spatial_colsfirst = False
    if blob[: len(MAGIC_TOPBAND5)] == MAGIC_TOPBAND5:
        has_residual_order = True
        sparse_plain = True
        sparse_colsfirst = True
        spatial_colsfirst = True
        pos += len(MAGIC_TOPBAND5)
    elif blob[: len(MAGIC_TOPBAND4)] == MAGIC_TOPBAND4:
        has_residual_order = True
        sparse_plain = True
        sparse_colsfirst = True
        pos += len(MAGIC_TOPBAND4)
    elif blob[: len(MAGIC_TOPBAND3)] == MAGIC_TOPBAND3:
        has_residual_order = True
        sparse_plain = True
        pos += len(MAGIC_TOPBAND3)
    elif blob[: len(MAGIC_TOPBAND2)] == MAGIC_TOPBAND2:
        has_residual_order = True
        sparse_plain = False
        pos += len(MAGIC_TOPBAND2)
    elif blob[: len(MAGIC_TOPBAND)] == MAGIC_TOPBAND:
        has_residual_order = False
        sparse_plain = False
        pos += len(MAGIC_TOPBAND)
    else:
        raise ValueError("bad QTBM1 magic")
    n_pairs, h, w, precision, _top_bins, _boundary_xbins, shift_dy, shift_dx = struct.unpack_from("<HHHBBBbb", blob, pos)
    pos += struct.calcsize("<HHHBBBbb")
    if has_residual_order:
        inv = list(blob[pos : pos + N_SYM])
        pos += N_SYM
        if sorted(inv) != [0, 1, 2, 3]:
            raise ValueError("invalid QTBM2 residual order")
    else:
        inv = [0, 1, 2, 3]
    top_len, road_len = struct.unpack_from("<II", blob, pos)
    pos += 8
    top_payload = blob[pos : pos + top_len]
    pos += top_len
    road_payload = blob[pos : pos + road_len]
    pos += road_len
    spatial_size, m5_size = struct.unpack_from("<HH", blob, pos)
    pos += struct.calcsize("<HH")
    if spatial_colsfirst:
        spatial_idx, spatial_rows = unpack_sparse_big_plain_colsfirst(blob[pos : pos + spatial_size], precision=precision)
    else:
        spatial_idx, spatial_rows = unpack_sparse_big_plain(blob[pos : pos + spatial_size], precision=precision)
    pos += spatial_size
    m5_idx, m5_rows = unpack_sparse_big_plain(blob[pos : pos + m5_size], precision=precision)
    pos += m5_size
    total = 1 << precision
    default = np.array([1, 1, 1, total - 3], dtype=np.uint16)
    spatial_freqs = np.broadcast_to(default, (5**4, N_SYM)).copy()
    m5_freqs = np.broadcast_to(default, (5**5, N_SYM)).copy()
    spatial_freqs[spatial_idx] = spatial_rows
    m5_freqs[m5_idx] = m5_rows
    n_feats = blob[pos]
    pos += 1
    feat_ids = tuple(blob[pos : pos + n_feats])
    pos += n_feats
    _thr_q8, sparse_len = struct.unpack_from("<HI", blob, pos)
    pos += 6
    sparse = blob[pos : pos + sparse_len]
    pos += sparse_len
    (bs_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    bitstream = blob[pos : pos + bs_len]

    top_support = decode_topband_payload(top_payload, n_pairs, h, w)
    road_mask = decode_boundary_mask_payload(road_payload, n_pairs, h, w)
    spatial_cdf_flat = np.zeros((5**4, N_SYM + 1), dtype=np.int64)
    spatial_cdf_flat[:, 1:] = np.cumsum(spatial_freqs.astype(np.int64), axis=-1)
    m5_cdf_flat = np.zeros((5**5, N_SYM + 1), dtype=np.int64)
    m5_cdf_flat[:, 1:] = np.cumsum(m5_freqs.astype(np.int64), axis=-1)

    if sparse_colsfirst:
        fired_idx, fired_freqs = unpack_sparse_big_plain_colsfirst(sparse, precision=precision)
    elif sparse_plain:
        fired_idx, fired_freqs = unpack_sparse_big_plain(sparse, precision=precision)
    else:
        fired_idx, fired_freqs = unpack_sparse_big(sparse, precision=precision)
    fired_cdf_flat_py: list = []
    if fired_idx.size > 0:
        fired_cdf = np.zeros((fired_idx.size, N_SYM + 1), dtype=np.int64)
        fired_cdf[:, 1:] = np.cumsum(fired_freqs.astype(np.int64), axis=-1)
        fired_cdf_flat_py = fired_cdf.tolist()
    m12_to_slot = {int(ctx): i for i, ctx in enumerate(fired_idx.tolist())}

    dec = RangeDecoder(bitstream)
    out = np.zeros((n_pairs, h, w), dtype=np.uint8)
    spatial_py = spatial_cdf_flat.tolist()
    m5_py = m5_cdf_flat.tolist()
    _decode_frame_topband(
        dec,
        out[0],
        None,
        None,
        None,
        top_support[0],
        road_mask[0],
        spatial_py,
        m5_py,
        fired_cdf_flat_py,
        m12_to_slot,
        feat_ids,
        total,
        h,
        w,
        shift_dy,
        shift_dx,
        inv,
    )
    if n_pairs >= 2:
        _decode_frame_topband(
            dec,
            out[1],
            out[0],
            None,
            None,
            top_support[1],
            road_mask[1],
            spatial_py,
            m5_py,
            fired_cdf_flat_py,
            m12_to_slot,
            feat_ids,
            total,
            h,
            w,
            shift_dy,
            shift_dx,
            inv,
        )
    for fi in range(2, n_pairs):
        ppp = out[fi - 3] if fi >= 3 else None
        _decode_frame_topband(
            dec,
            out[fi],
            out[fi - 1],
            out[fi - 2],
            ppp,
            top_support[fi],
            road_mask[fi],
            spatial_py,
            m5_py,
            fired_cdf_flat_py,
            m12_to_slot,
            feat_ids,
            total,
            h,
            w,
            shift_dy,
            shift_dx,
            inv,
        )
        if fi % 50 == 0:
            print(f"  decoded topband frame {fi}/{n_pairs}", flush=True)
    return out
