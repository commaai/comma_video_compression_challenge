"""Sparse-M11 seg_targets codec: M4 / M5 / M7 with 4-feature sparse-M11 override on frames 2+.

Extends sparse-M10 from 3 features to 4 by adding `diag_trtr` (y-2, x+2 anti-diagonal
distance-2). Context space grows from 5^10 (9.77M) to 5^11 (48.83M); fire rate drops
to 0.0005% (225 rows at thr=64) but per-row gain per fire is larger, netting -1,862 B
incremental vs M10 on ma1 (teacher-forced).

Deployed 4-tuple per step-31 sweep (`sparse_m11_sweep.py`):
  - diag_tltl : (y-2, x-2)
  - left_left : (y,   x-2)
  - top_top_top : (y-3, x)
  - diag_trtr : (y-2, x+2)    NEW relative to M10

Blob format is backwards-compatible with M10 (same fields + `n_feats` byte governs
feat_ids length). Decoder is shared with M10; the only new decoder capability is the
`FEAT_DIAG_TRTR = 4` case in `_decode_frame_m10` which was added alongside this module.

Teacher-forced numbers on ma1 600-pair (vs M10 shipped at -8,277 B):
  n_fired = 225 / 48,828,125 ctx (0.0005%)
  Δbs = -11,603 B,   table_deflate = 1,463 B,   Δnet = -10,140 B
  incremental vs M10 = -1,862 B   (fold gate was 300 B, cleared 6x)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from seg_sparse_m10_codec import (  # noqa: E402
    FEAT_NAME_TO_ID,
    PRECISION,
    build_blob_from_seg as _build_blob_from_seg_m10,
    decode_seg_split_m10 as _decode_seg_split_m10,
    load_seg_targets_lzma,
)

# Deployed 4-tuple per step-31 teacher-force sweep.
DEFAULT_FEATS = ("diag_tltl", "left_left", "top_top_top", "diag_trtr")
DEFAULT_THRESHOLD_BITS = 64.0


def build_blob_from_seg(
    seg: np.ndarray,
    peel_class: int,
    precision: int = PRECISION,
    feats: tuple[str, ...] = DEFAULT_FEATS,
    threshold_bits: float = DEFAULT_THRESHOLD_BITS,
) -> tuple[bytes, dict]:
    """End-to-end M11 encode. Shares blob layout with M10; n_feats=4 distinguishes."""
    for name in feats:
        if name not in FEAT_NAME_TO_ID:
            raise ValueError(f"unknown feat: {name}")
    return _build_blob_from_seg_m10(
        seg,
        peel_class=peel_class,
        precision=precision,
        feats=feats,
        threshold_bits=threshold_bits,
    )


def decode_seg_split_m11(path: "Path | str") -> np.ndarray:
    """Decode sparse-M11 seg_targets.bin → (n_pairs, H, W) uint8 array.

    Shares the M10 decoder; differs only in that n_feats=4 and feat_ids contain
    FEAT_DIAG_TRTR.
    """
    return _decode_seg_split_m10(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peel", type=int, default=4)
    parser.add_argument(
        "--src", type=str,
        default="submissions/qat_int8_bs32_segc4split_hc28_poslaplace_corrint8_ste_b_step81_ma1/archive/seg_targets.bin",
        help="source seg_targets.bin (any currently shipped codec)",
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--ref", type=str,
        default="submissions/qat_int8_bs32_segc4split_hc28_poslaplace_corrint8_ste_b_step81_ma1/archive/seg_targets.bin",
        help="reference (shipped) seg_targets.bin for byte-delta reporting",
    )
    parser.add_argument("--roundtrip", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--feats", type=str, default=",".join(DEFAULT_FEATS),
        help="comma-separated feature names (M11 default is 4-tuple)",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_BITS)
    args = parser.parse_args()

    src_path = (REPO / args.src).resolve()
    out_path = Path(args.out) if args.out else REPO / "experiments/seg_ctw/seg_targets_m11.bin"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading seg from {src_path.relative_to(REPO)}", flush=True)
    ma1_dir = REPO / "submissions/qat_int8_bs32_segc4split_hc28_poslaplace_corrint8_ste_b_step81_ma1"
    try:
        sys.path.insert(0, str(ma1_dir))
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
        rel = out_path.relative_to(REPO)
    except ValueError:
        rel = out_path
    print(f"\nwrote {rel}", flush=True)
    print(f"  total bytes: {len(blob):,}", flush=True)
    print(f"  n_fired: {diag['n_fired']:,}  fire_rate: {diag['fire_rate'] * 100:.4f}%", flush=True)
    print(
        f"  bitstream: {diag['bitstream_size']:,} B  "
        f"mask: {diag['mask_size']:,} B  "
        f"sparse-table: {diag['m10_compressed_size']:,} B",
        flush=True,
    )

    ref_path = (REPO / args.ref).resolve()
    if ref_path.exists():
        ref_bytes = ref_path.stat().st_size
        delta = len(blob) - ref_bytes
        print(f"  vs reference {ref_path.name} ({ref_bytes:,} B): {delta:+,} B", flush=True)

    if args.roundtrip:
        print("\nroundtrip: decoding blob...", flush=True)
        t0 = time.time()
        recovered = decode_seg_split_m11(out_path)
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
