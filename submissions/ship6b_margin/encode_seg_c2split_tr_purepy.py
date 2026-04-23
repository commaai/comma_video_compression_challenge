"""Build seg_targets.bin via class-split peel + higher-order AC residual (+tr neighbour).

Variant of encode_seg_c2split_purepy.py that extends the spatial context with the
top-right neighbour (y-1, x+1) — causal under raster scan. This adds one dimension
to the CDF tables:

  - frame 0: P(sym | top, left, tl, tr)           — Markov-4 spatial,  5000 B raw CDF
  - frames 1+: P(sym | top, left, tl, tr, prev)    — Markov-5 temporal, 25000 B raw CDF

v2 DEFLATE-aware scoping (2026-04-17) measured that the +24 KB raw CDF growth only
costs ~+1.7 KB in archive.zip because the expanded tables have huge near-uniform
regions after Laplace smoothing. Entropy savings on the 59.6M-symbol residual were
projected at -17.5 KB (model-level) and the net ship-layer delta is -19.6 KB.

On-disk format: IDENTICAL to the original c2split format except the two CDF
tables have new shapes:

    uint16 n_pairs, uint16 H, uint16 W
    uint8  precision
    uint8  peel_class
    uint8  mask_format
    uint32 mask_payload_len
    <mask_payload bytes>
    uint16 spatial_size_bytes   # 2 * 5**4 * 4 = 5000
    uint16 temporal_size_bytes  # 2 * 5**5 * 4 = 25000
    spatial freqs  : uint16[5,5,5,5,4]     P(remapped_target | top,left,tl,tr)  frame 0
    temporal freqs : uint16[5,5,5,5,5,4]   P(remapped_target | top,left,tl,tr,prev) frames 1+
    uint32 bitstream_length
    <bitstream bytes>

Decoder (seg_c2split_tr_codec.py, sibling) MUST match exactly.

Usage:
    uv run python experiments/arithmetic_coder/encode_seg_c2split_tr_purepy.py --peel 4 --roundtrip
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from range_coder import RangeDecoder, RangeEncoder  # noqa: E402
from encode_seg_c2split_purepy import (  # noqa: E402
    N_CLASSES,
    BORDER,
    PRECISION,
    MASK_FORMAT_BZ2_RAW,
    MASK_FORMAT_BZ2_PACKBITS,
    MASK_FORMAT_LZMA_RAW,
    MASK_FORMAT_LZMA_PACKBITS,
    load_seg_targets_lzma,
    make_remap_tables,
    compute_spatial_contexts,
    quantize_freqs,
    encode_mask_best,
    decode_mask_payload,
)


def compute_tr(frame: np.ndarray) -> np.ndarray:
    """Top-right neighbour (y-1, x+1). Causal under raster scan. BORDER at edges."""
    h, w = frame.shape
    tr = np.full((h, w), BORDER, dtype=np.uint8)
    tr[1:, :-1] = frame[:-1, 1:]
    return tr


def build_spatial_counts_tr(seg: np.ndarray, peel_class: int) -> np.ndarray:
    """4-neighbour spatial counts: (top, left, tl, tr) — frame 0 model.

    Shape: (5, 5, 5, 5, 4) = 625 contexts × 4 residual symbols.
    Counts accumulate only at non-peel positions; context indices use full 5-class labels.
    """
    forward, _ = make_remap_tables(peel_class)
    counts = np.zeros((N_CLASSES,) * 4 + (N_CLASSES - 1,), dtype=np.int64)
    frame = seg[0]
    top, left, tl = compute_spatial_contexts(frame)
    tr = compute_tr(frame)
    non_peel = frame != peel_class
    target = forward[frame[non_peel]]
    np.add.at(
        counts,
        (top[non_peel].ravel(), left[non_peel].ravel(),
         tl[non_peel].ravel(), tr[non_peel].ravel(), target.ravel()),
        1,
    )
    return counts


def build_temporal_counts_tr(seg: np.ndarray, peel_class: int) -> np.ndarray:
    """5-neighbour temporal counts: (top, left, tl, tr, prev) — frames 1+ model.

    Shape: (5, 5, 5, 5, 5, 4) = 3125 contexts × 4 residual symbols.
    """
    forward, _ = make_remap_tables(peel_class)
    counts = np.zeros((N_CLASSES,) * 5 + (N_CLASSES - 1,), dtype=np.int64)
    for i in range(1, seg.shape[0]):
        frame = seg[i]
        prev = seg[i - 1]
        top, left, tl = compute_spatial_contexts(frame)
        tr = compute_tr(frame)
        non_peel = frame != peel_class
        target = forward[frame[non_peel]]
        np.add.at(
            counts,
            (
                top[non_peel].ravel(), left[non_peel].ravel(),
                tl[non_peel].ravel(), tr[non_peel].ravel(),
                prev[non_peel].ravel(), target.ravel(),
            ),
            1,
        )
    return counts


def encode_seg_split_tr(
    seg: np.ndarray,
    peel_class: int,
    spatial_freqs: np.ndarray,
    temporal_freqs: np.ndarray,
    precision: int,
) -> bytes:
    """Range-encode the 4-class residual with M4-spatial-tr + M5-temporal-tr.

    Frame 0 uses (top, left, tl, tr); frames 1+ use (top, left, tl, tr, prev).
    Peel positions are skipped (reconstructed from mask).
    """
    total = 1 << precision
    n_other = N_CLASSES - 1
    forward, _ = make_remap_tables(peel_class)

    spatial_cdf = np.zeros((N_CLASSES,) * 4 + (n_other + 1,), dtype=np.int64)
    spatial_cdf[..., 1:] = np.cumsum(spatial_freqs.astype(np.int64), axis=-1)
    temporal_cdf = np.zeros((N_CLASSES,) * 5 + (n_other + 1,), dtype=np.int64)
    temporal_cdf[..., 1:] = np.cumsum(temporal_freqs.astype(np.int64), axis=-1)

    enc = RangeEncoder()
    n_frames, h, w = seg.shape
    es = enc.encode_symbol

    def encode_run(cl: list[int], ch: list[int]) -> None:
        for i in range(len(cl)):
            es(cl[i], ch[i], total)

    # frame 0: spatial Markov-4 (+tr)
    frame = seg[0]
    top, left, tl = compute_spatial_contexts(frame)
    tr = compute_tr(frame)
    non_peel = frame != peel_class
    cdf_pair = spatial_cdf[top, left, tl, tr]  # (H, W, n_other+1)
    targets_safe = np.where(non_peel, forward[frame].astype(np.int64), 0)
    cum_low = np.take_along_axis(cdf_pair, targets_safe[..., None], axis=-1)[..., 0]
    cum_high = np.take_along_axis(cdf_pair, targets_safe[..., None] + 1, axis=-1)[..., 0]
    keep = non_peel.ravel()
    encode_run(cum_low.ravel()[keep].tolist(), cum_high.ravel()[keep].tolist())

    # frames 1+: temporal Markov-5 (+tr +prev)
    for fi in range(1, n_frames):
        frame = seg[fi]
        prev = seg[fi - 1]
        top, left, tl = compute_spatial_contexts(frame)
        tr = compute_tr(frame)
        non_peel = frame != peel_class
        cdf_pair = temporal_cdf[top, left, tl, tr, prev]  # (H, W, n_other+1)
        targets_safe = np.where(non_peel, forward[frame].astype(np.int64), 0)
        cum_low = np.take_along_axis(cdf_pair, targets_safe[..., None], axis=-1)[..., 0]
        cum_high = np.take_along_axis(cdf_pair, targets_safe[..., None] + 1, axis=-1)[..., 0]
        keep = non_peel.ravel()
        encode_run(cum_low.ravel()[keep].tolist(), cum_high.ravel()[keep].tolist())
        if fi % 50 == 0:
            print(f"  encoded frame {fi}/{n_frames}", flush=True)

    return enc.finish()


def pack_archive_blob(
    n_pairs: int,
    h: int,
    w: int,
    precision: int,
    peel_class: int,
    mask_payload: bytes,
    mask_format: int,
    spatial_freqs: np.ndarray,
    temporal_freqs: np.ndarray,
    bitstream: bytes,
) -> bytes:
    """Assemble the on-disk seg_targets.bin — same format as c2split, new CDF shapes."""
    header = struct.pack("<HHHBBB", n_pairs, h, w, precision, peel_class, mask_format)
    mask_len = struct.pack("<I", len(mask_payload))
    spatial_bytes = spatial_freqs.astype("<u2").tobytes()
    temporal_bytes = temporal_freqs.astype("<u2").tobytes()
    sizes = struct.pack("<HH", len(spatial_bytes), len(temporal_bytes))
    bslen = struct.pack("<I", len(bitstream))
    return (
        header + mask_len + mask_payload
        + sizes + spatial_bytes + temporal_bytes
        + bslen + bitstream
    )


def unpack_archive_blob(blob: bytes) -> dict:
    """Inverse of pack_archive_blob for the +tr variant."""
    pos = 0
    n_pairs, h, w, precision, peel_class, mask_format = struct.unpack_from("<HHHBBB", blob, pos)
    pos += struct.calcsize("<HHHBBB")
    (mask_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    mask_payload = blob[pos : pos + mask_len]
    pos += mask_len
    spatial_size, temporal_size = struct.unpack_from("<HH", blob, pos)
    pos += 4
    spatial_freqs = np.frombuffer(
        blob, dtype="<u2", count=spatial_size // 2, offset=pos
    ).reshape((N_CLASSES,) * 4 + (N_CLASSES - 1,))
    pos += spatial_size
    temporal_freqs = np.frombuffer(
        blob, dtype="<u2", count=temporal_size // 2, offset=pos
    ).reshape((N_CLASSES,) * 5 + (N_CLASSES - 1,))
    pos += temporal_size
    (bs_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    bitstream = blob[pos : pos + bs_len]
    return {
        "n_pairs": n_pairs, "h": h, "w": w,
        "precision": precision, "peel_class": peel_class, "mask_format": mask_format,
        "mask_payload": mask_payload,
        "spatial_freqs": spatial_freqs, "temporal_freqs": temporal_freqs,
        "bitstream": bitstream,
    }


def _decode_frame_spatial_tr(
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
    """Frame-0 M4-spatial decode. tr = frame[y-1][x+1] is causal under raster."""
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


def _decode_frame_temporal_tr(
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
    """Frame-1+ M5-temporal decode: (top, left, tl, tr, prev)."""
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


def decode_seg_split_tr(blob: bytes) -> np.ndarray:
    """Decode the +tr c2split format to a (n_pairs, H, W) uint8 array."""
    fields = unpack_archive_blob(blob)
    n_pairs = fields["n_pairs"]
    h = fields["h"]
    w = fields["w"]
    precision = fields["precision"]
    peel_class = fields["peel_class"]
    spatial_freqs = fields["spatial_freqs"]
    temporal_freqs = fields["temporal_freqs"]
    bitstream = fields["bitstream"]

    mask = decode_mask_payload(fields["mask_payload"], fields["mask_format"], n_pairs, h, w)
    _, inverse = make_remap_tables(peel_class)
    inv_remap_py = inverse.tolist()

    total = 1 << precision
    n_other = N_CLASSES - 1
    spatial_cdf = np.zeros((N_CLASSES,) * 4 + (n_other + 1,), dtype=np.int64)
    spatial_cdf[..., 1:] = np.cumsum(spatial_freqs.astype(np.int64), axis=-1)
    temporal_cdf = np.zeros((N_CLASSES,) * 5 + (n_other + 1,), dtype=np.int64)
    temporal_cdf[..., 1:] = np.cumsum(temporal_freqs.astype(np.int64), axis=-1)
    spatial_py = spatial_cdf.tolist()
    temporal_py = temporal_cdf.tolist()

    dec = RangeDecoder(bitstream)
    out = np.zeros((n_pairs, h, w), dtype=np.uint8)

    _decode_frame_spatial_tr(
        dec, out[0], mask[0], spatial_py, total, h, w, peel_class, inv_remap_py
    )
    for fi in range(1, n_pairs):
        _decode_frame_temporal_tr(
            dec, out[fi], out[fi - 1], mask[fi],
            temporal_py, total, h, w, peel_class, inv_remap_py,
        )
    return out


def build_blob_from_seg(seg: np.ndarray, peel_class: int, precision: int = PRECISION) -> bytes:
    """End-to-end: counts → freqs → mask + bitstream → packed blob (M4-tr + M5-tr)."""
    print("  building M4-spatial counts (+tr), frame 0...", flush=True)
    spatial_counts = build_spatial_counts_tr(seg, peel_class)
    print("  building M5-temporal counts (+tr +prev), frames 1+...", flush=True)
    temporal_counts = build_temporal_counts_tr(seg, peel_class)

    print("  quantizing freqs...", flush=True)
    spatial_freqs = quantize_freqs(spatial_counts, precision)
    temporal_freqs = quantize_freqs(temporal_counts, precision)

    print(f"  encoding peel-class binary mask (peel=c{peel_class})...", flush=True)
    is_peel = (seg == peel_class).astype(np.uint8).reshape(-1)
    mask_payload, mask_format = encode_mask_best(is_peel)
    mask_format_names = {
        MASK_FORMAT_BZ2_RAW: "bz2-raw",
        MASK_FORMAT_BZ2_PACKBITS: "bz2-packbits",
        MASK_FORMAT_LZMA_RAW: "lzma-raw",
        MASK_FORMAT_LZMA_PACKBITS: "lzma-packbits",
    }
    print(f"    chose {mask_format_names[mask_format]}: {len(mask_payload):,} B")

    print("  encoding 4-class residual via pure-python range coder (+tr)...", flush=True)
    t0 = time.time()
    bitstream = encode_seg_split_tr(seg, peel_class, spatial_freqs, temporal_freqs, precision)
    elapsed = time.time() - t0
    n_non_peel = int((seg != peel_class).sum())
    print(
        f"    encoded {n_non_peel} non-peel symbols in {elapsed:.1f}s "
        f"({n_non_peel / max(elapsed, 1e-6):.0f} sym/s, {len(bitstream):,} B bitstream)"
    )

    blob = pack_archive_blob(
        seg.shape[0], seg.shape[1], seg.shape[2], precision,
        peel_class, mask_payload, mask_format,
        spatial_freqs, temporal_freqs, bitstream,
    )
    return blob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peel", type=int, default=4, choices=list(range(N_CLASSES)),
                        help="peel class (default 4 to match shipped archive)")
    parser.add_argument("--src", type=str, default="submissions/margin_only/seg_targets.bin",
                        help="lzma+rle source seg_targets.bin")
    parser.add_argument("--out", type=str, default=None,
                        help="output path for the new seg_targets.bin")
    parser.add_argument("--roundtrip", action="store_true",
                        help="decode + assert byte-identical reconstruction")
    parser.add_argument("--limit", type=int, default=None,
                        help="encode only first N pairs (smoke)")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    seg_path = repo / args.src
    out_path = Path(args.out) if args.out else repo / "experiments/scoping_20260417/seg_targets_tr.bin"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading seg from {seg_path.relative_to(repo)}")
    seg = load_seg_targets_lzma(seg_path)
    if args.limit is not None:
        seg = seg[: args.limit]
    print(f"  shape: {seg.shape}  peel_class: c{args.peel}")
    print()

    blob = build_blob_from_seg(seg, args.peel)
    out_path.write_bytes(blob)

    print()
    try:
        rel = out_path.relative_to(repo)
    except ValueError:
        rel = out_path
    print(f"wrote {rel}")
    print(f"  total bytes:    {len(blob):,}")

    # compare against current shipped seg_targets.bin if peel matches
    current = repo / "submissions/qat_int8_bs32_segc4split_hc28_poslaplace_corrint8_ste_b_step81/archive/seg_targets.bin"
    if current.exists():
        current_bytes = current.stat().st_size
        delta = len(blob) - current_bytes
        print(f"  vs current shipped seg_targets.bin ({current_bytes:,} B): {delta:+,} B")

    if args.roundtrip:
        print()
        print("roundtrip: decoding blob and checking byte-identical reconstruction...")
        t0 = time.time()
        recovered = decode_seg_split_tr(blob)
        elapsed = time.time() - t0
        print(f"  decoded {recovered.shape} in {elapsed:.1f}s")
        if not np.array_equal(recovered, seg):
            n_diff = int((recovered != seg).sum())
            raise SystemExit(
                f"ROUNDTRIP FAILED: {n_diff} pixels differ between encoded and decoded seg"
            )
        print("  PASS: encoded blob decodes to byte-identical seg array")


if __name__ == "__main__":
    main()
