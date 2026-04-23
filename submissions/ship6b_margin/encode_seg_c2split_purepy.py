"""Build seg_targets.bin via single-class peel split + 4-class AC residual.

Direction 1 (c2 variant) from the parallel-research synthesis 2026-04-11.
Sister module to encode_seg_v3_purepy.py — same range coder, same Markov-3
spatial / Markov-4 temporal context structure, but the dominant peel class
(default c2 = sky/background, 49.5% of pixels) is pulled out into a
separately compressed binary mask. The arithmetic-coded residual then has
a 4-symbol alphabet over the 50.5% of pixels that aren't c2.

Crucial detail: the residual encoder uses the FULL 5-class context (top,
left, top-left, prev) — i.e. the model knows when a neighbour was the peel
class — even though it only ever EMITS one of the 4 non-peel symbols.
That makes it strictly tighter than the c2_entropy.py Strategy B estimate
which collapsed contexts to 4 classes.

On-disk format ("seg_targets.bin"):

    uint16 n_pairs, uint16 H, uint16 W
    uint8  precision         # typically 16
    uint8  peel_class        # 0..N_CLASSES-1
    uint8  mask_format       # 0=bz2-raw, 1=bz2-packbits, 2=lzma-raw, 3=lzma-packbits
    uint32 mask_payload_len
    <mask_payload bytes>
    uint16 spatial_size_bytes  # 2 * 5**3 * 4
    uint16 temporal_size_bytes # 2 * 5**4 * 4
    spatial freqs  : uint16[5,5,5,4]    P(remapped_target | top,left,tl)  frame 0
    temporal freqs : uint16[5,5,5,5,4]  P(remapped_target | top,left,tl,prev) frames 1+
    uint32 bitstream_length
    <bitstream bytes>

Each freq row sums to exactly 2**precision so the decoder infers totals
without storing them. The remap maps the 4 non-peel labels into {0,1,2,3}
in increasing-original-label order (so for peel=2, residual symbols
{0,1,2,3} mean original classes {0,1,3,4}). The context indices on the
spatial/temporal axes are still the full 5-class labels.

Usage:
    uv run python experiments/arithmetic_coder/encode_seg_c2split_purepy.py --peel 2
    uv run python experiments/arithmetic_coder/encode_seg_c2split_purepy.py --peel 2 --roundtrip
"""

from __future__ import annotations

import argparse
import bz2
import lzma
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from range_coder import RangeDecoder, RangeEncoder

N_CLASSES = 5
BORDER = 0
PRECISION = 16

MASK_FORMAT_BZ2_RAW = 0
MASK_FORMAT_BZ2_PACKBITS = 1
MASK_FORMAT_LZMA_RAW = 2
MASK_FORMAT_LZMA_PACKBITS = 3


def load_seg_targets_lzma(path: Path) -> np.ndarray:
    """Read the v2 lzma+rle seg_targets.bin into a (n_pairs, H, W) uint8 array."""

    def rle_decode(data: bytes, output_size: int) -> np.ndarray:
        result = np.empty(output_size, dtype=np.uint8)
        pos = 0
        i = 0
        while i < len(data):
            result[pos : pos + data[i]] = data[i + 1]
            pos += data[i]
            i += 2
        return result

    with open(path, "rb") as f:
        n_pairs, h, w = struct.unpack("<HHH", f.read(6))
        rle = lzma.decompress(f.read())
    flat = rle_decode(rle, n_pairs * h * w)
    return flat.reshape(n_pairs, h, w)


def make_remap_tables(peel_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Build forward / inverse maps between 5-class labels and 4-class residual indices.

    forward[c] -> 0..3 for non-peel classes, 255 sentinel for peel class
    inverse[i] -> original 5-class label for residual index i in [0,4)
    """
    forward = np.full(N_CLASSES, 255, dtype=np.uint8)
    inverse = np.zeros(N_CLASSES - 1, dtype=np.uint8)
    new_idx = 0
    for c in range(N_CLASSES):
        if c == peel_class:
            continue
        forward[c] = new_idx
        inverse[new_idx] = c
        new_idx += 1
    return forward, inverse


def compute_spatial_contexts(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top / left / top-left neighbour planes with BORDER padding at edges."""
    h, w = frame.shape
    top = np.full((h, w), BORDER, dtype=np.uint8)
    top[1:, :] = frame[:-1, :]
    left = np.full((h, w), BORDER, dtype=np.uint8)
    left[:, 1:] = frame[:, :-1]
    tl = np.full((h, w), BORDER, dtype=np.uint8)
    tl[1:, 1:] = frame[:-1, :-1]
    return top, left, tl


def build_spatial_counts_split(seg: np.ndarray, peel_class: int) -> np.ndarray:
    """3-neighbour counts over the 4-class residual, full 5-class context.

    Counts are accumulated only at non-peel positions (the encoder skips
    peel positions). Context indices are still 5-class because that's what
    the decoder reconstructs from the mask + previously decoded pixels.
    """
    forward, _ = make_remap_tables(peel_class)
    counts = np.zeros((N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES - 1), dtype=np.int64)
    for i in range(seg.shape[0]):
        frame = seg[i]
        top, left, tl = compute_spatial_contexts(frame)
        non_peel = frame != peel_class
        target = forward[frame[non_peel]]
        np.add.at(
            counts,
            (top[non_peel].ravel(), left[non_peel].ravel(), tl[non_peel].ravel(), target.ravel()),
            1,
        )
    return counts


def build_temporal_counts_split(seg: np.ndarray, peel_class: int) -> np.ndarray:
    """4-neighbour counts (top, left, tl, prev) over the 4-class residual."""
    forward, _ = make_remap_tables(peel_class)
    counts = np.zeros(
        (N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES - 1), dtype=np.int64
    )
    for i in range(1, seg.shape[0]):
        frame = seg[i]
        prev = seg[i - 1]
        top, left, tl = compute_spatial_contexts(frame)
        non_peel = frame != peel_class
        target = forward[frame[non_peel]]
        np.add.at(
            counts,
            (
                top[non_peel].ravel(),
                left[non_peel].ravel(),
                tl[non_peel].ravel(),
                prev[non_peel].ravel(),
                target.ravel(),
            ),
            1,
        )
    return counts


def quantize_freqs(counts: np.ndarray, precision: int) -> np.ndarray:
    """Convert raw counts to uint16 freqs summing to exactly 2**precision per row.

    Last axis is the symbol alphabet. Identical to encode_seg_v3_purepy.quantize_freqs
    but kept here so the c2-split encoder is fully self-contained for the
    inflate-side decoder.
    """
    denom = 1 << precision
    # Integer-math quantization: float64 pairwise-sum + probs*denom had a
    # platform-dependent hazard (numpy BLOCKSIZE differs by arch/BLAS; 1 ulp
    # drift can cross the astype(int64) boundary). Int64 floor-div is
    # bit-identical across x86_64 and arm64. Exact for the values here:
    # counts max ~10^8 symbols, denom <= 2^15, so smoothed*denom fits in int64.
    smoothed = counts.astype(np.int64) + 1
    row_sums = smoothed.sum(axis=-1, keepdims=True)
    freqs = np.maximum((smoothed * denom) // row_sums, 1)
    it = np.nditer(freqs[..., 0], flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        row = freqs[idx]
        diff = denom - int(row.sum())
        target_idx = int(np.argmax(row))
        new_val = int(row[target_idx]) + diff
        if new_val <= 0:
            nonzero = [k for k in range(row.shape[0]) if row[k] > 1]
            for k in nonzero:
                if diff >= 0:
                    row[k] += 1
                    diff -= 1
                else:
                    if row[k] > 1:
                        row[k] -= 1
                        diff += 1
                if diff == 0:
                    break
        else:
            row[target_idx] = new_val
        assert int(row.sum()) == denom, f"row {idx} sums to {int(row.sum())}"
        freqs[idx] = row
    # Belt-and-suspenders: vectorized CDF-sum invariant for the whole table.
    assert (freqs.sum(axis=-1) == denom).all(), "freqs row-sum != denom after residual pass"
    return freqs.astype(np.uint16)


def encode_mask_best(mask_uint8: np.ndarray) -> tuple[bytes, int]:
    """Try four binary-mask compressors and return the smallest payload + format tag.

    mask_uint8 must be a flat 0/1 uint8 array. Returns (payload_bytes, mask_format).
    """
    raw = mask_uint8.tobytes()
    packed = np.packbits(mask_uint8).tobytes()
    options = [
        (MASK_FORMAT_BZ2_RAW, bz2.compress(raw, compresslevel=9)),
        (MASK_FORMAT_BZ2_PACKBITS, bz2.compress(packed, compresslevel=9)),
        (MASK_FORMAT_LZMA_RAW, lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)),
        (MASK_FORMAT_LZMA_PACKBITS, lzma.compress(packed, preset=9 | lzma.PRESET_EXTREME)),
    ]
    best_format, best_payload = min(options, key=lambda kv: len(kv[1]))
    return best_payload, best_format


def decode_mask_payload(
    payload: bytes, mask_format: int, n_pairs: int, h: int, w: int
) -> np.ndarray:
    """Inverse of encode_mask_best. Returns a (n_pairs, H, W) uint8 0/1 array."""
    n_pix = n_pairs * h * w
    if mask_format == MASK_FORMAT_BZ2_RAW:
        raw = bz2.decompress(payload)
        flat = np.frombuffer(raw, dtype=np.uint8)
    elif mask_format == MASK_FORMAT_BZ2_PACKBITS:
        packed = bz2.decompress(payload)
        flat = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:n_pix]
    elif mask_format == MASK_FORMAT_LZMA_RAW:
        raw = lzma.decompress(payload)
        flat = np.frombuffer(raw, dtype=np.uint8)
    elif mask_format == MASK_FORMAT_LZMA_PACKBITS:
        packed = lzma.decompress(payload)
        flat = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:n_pix]
    else:
        raise ValueError(f"unknown mask_format: {mask_format}")
    if flat.size != n_pix:
        raise ValueError(f"mask payload decoded to {flat.size} bytes, expected {n_pix}")
    return flat.reshape(n_pairs, h, w)


def encode_seg_split(
    seg: np.ndarray,
    peel_class: int,
    spatial_freqs: np.ndarray,
    temporal_freqs: np.ndarray,
    precision: int,
) -> bytes:
    """Range-encode the 4-class residual at non-peel positions only.

    Frame 0 uses Markov-3 (top, left, top-left); frames 1+ use Markov-4
    (top, left, top-left, prev). Peel positions are skipped — the decoder
    reconstructs them from the bz2 mask.
    """
    total = 1 << precision
    n_other = N_CLASSES - 1
    forward, _ = make_remap_tables(peel_class)

    spatial_cdf = np.zeros((N_CLASSES, N_CLASSES, N_CLASSES, n_other + 1), dtype=np.int64)
    spatial_cdf[..., 1:] = np.cumsum(spatial_freqs.astype(np.int64), axis=-1)
    temporal_cdf = np.zeros(
        (N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES, n_other + 1), dtype=np.int64
    )
    temporal_cdf[..., 1:] = np.cumsum(temporal_freqs.astype(np.int64), axis=-1)

    enc = RangeEncoder()
    n_frames, h, w = seg.shape
    es = enc.encode_symbol

    def encode_run(cl: list[int], ch: list[int]) -> None:
        for i in range(len(cl)):
            es(cl[i], ch[i], total)

    # frame 0: spatial Markov-3
    frame = seg[0]
    top, left, tl = compute_spatial_contexts(frame)
    non_peel = frame != peel_class
    cdf_pair = spatial_cdf[top, left, tl]  # (H, W, n_other+1)
    # clamp peel-position targets to 0 so take_along_axis stays in-bounds; we mask
    # those samples out before they hit the encoder.
    targets_safe = np.where(non_peel, forward[frame].astype(np.int64), 0)
    cum_low = np.take_along_axis(cdf_pair, targets_safe[..., None], axis=-1)[..., 0]
    cum_high = np.take_along_axis(cdf_pair, targets_safe[..., None] + 1, axis=-1)[..., 0]
    keep = non_peel.ravel()
    encode_run(cum_low.ravel()[keep].tolist(), cum_high.ravel()[keep].tolist())

    # frames 1+: temporal Markov-4
    for fi in range(1, n_frames):
        frame = seg[fi]
        prev = seg[fi - 1]
        top, left, tl = compute_spatial_contexts(frame)
        non_peel = frame != peel_class
        cdf_pair = temporal_cdf[top, left, tl, prev]  # (H, W, n_other+1)
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
    """Assemble the on-disk seg_targets.bin for the c2-split format."""
    header = struct.pack("<HHHBBB", n_pairs, h, w, precision, peel_class, mask_format)
    mask_len = struct.pack("<I", len(mask_payload))
    spatial_bytes = spatial_freqs.astype("<u2").tobytes()
    temporal_bytes = temporal_freqs.astype("<u2").tobytes()
    sizes = struct.pack("<HH", len(spatial_bytes), len(temporal_bytes))
    bslen = struct.pack("<I", len(bitstream))
    return (
        header
        + mask_len
        + mask_payload
        + sizes
        + spatial_bytes
        + temporal_bytes
        + bslen
        + bitstream
    )


def unpack_archive_blob(blob: bytes) -> dict:
    """Inverse of pack_archive_blob. Returns a dict of decoded fields for decode_seg_split."""
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
    ).reshape(N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES - 1)
    pos += spatial_size
    temporal_freqs = np.frombuffer(
        blob, dtype="<u2", count=temporal_size // 2, offset=pos
    ).reshape(N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES - 1)
    pos += temporal_size
    (bs_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    bitstream = blob[pos : pos + bs_len]
    return {
        "n_pairs": n_pairs,
        "h": h,
        "w": w,
        "precision": precision,
        "peel_class": peel_class,
        "mask_format": mask_format,
        "mask_payload": mask_payload,
        "spatial_freqs": spatial_freqs,
        "temporal_freqs": temporal_freqs,
        "bitstream": bitstream,
    }


def _decode_frame_spatial(
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
    """Inverse of the encoder's frame-0 spatial Markov-3 path. Pure python hot loop."""
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
            cdf_row = cdf_py[top_v][left_v][tl_v]
            t = get_target(total)
            s = 0
            while cdf_row[s + 1] <= t:
                s += 1
            frame_list[y][x] = inv[s]
            advance(cdf_row[s], cdf_row[s + 1], total)
    for y in range(h):
        frame[y] = frame_list[y]


def _decode_frame_temporal(
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
    """Inverse of the encoder's frame-1+ temporal Markov-4 path."""
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
            prev_v = prev_row[x]
            cdf_row = cdf_py[top_v][left_v][tl_v][prev_v]
            t = get_target(total)
            s = 0
            while cdf_row[s + 1] <= t:
                s += 1
            frame_list[y][x] = inv[s]
            advance(cdf_row[s], cdf_row[s + 1], total)
    for y in range(h):
        frame[y] = frame_list[y]


def decode_seg_split(blob: bytes) -> np.ndarray:
    """Decode the c2-split format back to a (n_pairs, H, W) uint8 array."""
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
    spatial_cdf = np.zeros((N_CLASSES, N_CLASSES, N_CLASSES, n_other + 1), dtype=np.int64)
    spatial_cdf[..., 1:] = np.cumsum(spatial_freqs.astype(np.int64), axis=-1)
    temporal_cdf = np.zeros(
        (N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES, n_other + 1), dtype=np.int64
    )
    temporal_cdf[..., 1:] = np.cumsum(temporal_freqs.astype(np.int64), axis=-1)
    spatial_py = spatial_cdf.tolist()
    temporal_py = temporal_cdf.tolist()

    dec = RangeDecoder(bitstream)
    out = np.zeros((n_pairs, h, w), dtype=np.uint8)

    _decode_frame_spatial(
        dec, out[0], mask[0], spatial_py, total, h, w, peel_class, inv_remap_py
    )
    for fi in range(1, n_pairs):
        _decode_frame_temporal(
            dec,
            out[fi],
            out[fi - 1],
            mask[fi],
            temporal_py,
            total,
            h,
            w,
            peel_class,
            inv_remap_py,
        )
    return out


def build_blob_from_seg(seg: np.ndarray, peel_class: int, precision: int = PRECISION) -> bytes:
    """End-to-end: counts -> freqs -> mask + bitstream -> packed blob."""
    print("  building spatial counts (Markov-3, frame 0 model)...", flush=True)
    spatial_counts = build_spatial_counts_split(seg, peel_class)
    print("  building temporal counts (Markov-4, frames 1+ model)...", flush=True)
    temporal_counts = build_temporal_counts_split(seg, peel_class)

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

    print("  encoding 4-class residual via pure-python range coder...", flush=True)
    t0 = time.time()
    bitstream = encode_seg_split(seg, peel_class, spatial_freqs, temporal_freqs, precision)
    elapsed = time.time() - t0
    n_non_peel = int((seg != peel_class).sum())
    print(
        f"    encoded {n_non_peel} non-peel symbols in {elapsed:.1f}s "
        f"({n_non_peel / max(elapsed, 1e-6):.0f} sym/s)"
    )

    blob = pack_archive_blob(
        seg.shape[0],
        seg.shape[1],
        seg.shape[2],
        precision,
        peel_class,
        mask_payload,
        mask_format,
        spatial_freqs,
        temporal_freqs,
        bitstream,
    )
    return blob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--peel", type=int, default=2, choices=list(range(N_CLASSES)),
        help="class to peel into a separate binary mask (default 2 = sky/background)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output path (default: submissions/qat_int8_bs64_segc2split/archive/seg_targets.bin)",
    )
    parser.add_argument(
        "--roundtrip", action="store_true",
        help="after writing the blob, decode it and assert byte-identical reconstruction",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="encode only first N pairs (smoke test)",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    seg_path = repo / "submissions/margin_only/seg_targets.bin"
    out_path = (
        Path(args.out)
        if args.out
        else repo / "submissions/qat_int8_bs64_segc2split/archive/seg_targets.bin"
    )
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
    print("  baseline (current AC-coded seg_targets.bin): 242,387 B")
    delta = len(blob) - 242_387
    print(f"  delta vs current AC: {delta:+,} B")
    if delta < 0:
        score_delta = 25 * delta / 37_545_489
        print(f"  projected rate-only score delta: {score_delta:+.4f}")

    if args.roundtrip:
        print()
        print("roundtrip: decoding blob and checking byte-identical reconstruction...")
        t0 = time.time()
        recovered = decode_seg_split(blob)
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
