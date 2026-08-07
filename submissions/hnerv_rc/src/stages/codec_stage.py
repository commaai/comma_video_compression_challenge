"""Codec re-encode — final archive build (no training).

Reads the latest snapshot from the previous stage's output directory, runs it
through the codec, and writes a single submission file at
`<output_dir>/0.bin`. Round-trip is verified bit-exact for the INT8 weights.

(The choice to use `0.bin` matches the file-list iteration model the challenge's
`evaluate.sh` uses: for each video named `<base>` in `public_test_video_names.txt`,
inflate.sh expects the corresponding `<base>.bin` in the data dir.)
"""
import json
from pathlib import Path

import numpy as np
import torch

from codec import build_archive, parse_archive
from data import EVAL_SIZE


def run_codec_stage(prev_stage_output_dir: Path, final_output_dir: Path,
                    video_path: Path) -> dict:
    """Read the previous stage's BEST checkpoint, re-encode via the codec, and
    write the submission archive at <final_output_dir>/0.bin.

    Note: each training stage already wrote a best_archive.bin in its output dir;
    this codec stage is functionally a verified re-emission, plus it writes the
    final-format file under the canonical submission name.
    """
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Use the BEST checkpoint (lowest in-training eval score).
    decoder_sd = torch.load(prev_stage_output_dir / "decoder_f32.pt", map_location='cpu')
    latents = torch.load(prev_stage_output_dir / "latents_f32.pt", map_location='cpu')
    n_pairs = latents.shape[0]

    archive = build_archive(
        decoder_sd, latents,
        meta_dict={"n_pairs": n_pairs, "latent_dim": 28, "base_channels": 36,
                   "eval_size": list(EVAL_SIZE)},
    )
    archive_bytes = len(archive)

    with open(final_output_dir / "0.bin", "wb") as f:
        f.write(archive)

    # Verify round-trip: re-parse and check INT8 weight equivalence.
    decoder_sd_dec, _, _ = parse_archive(archive)
    for name in decoder_sd:
        orig = decoder_sd[name].detach().cpu().float()
        ma = orig.abs().max().item()
        scale = ma / 127 if ma > 0 else 1.0
        orig_q = (orig / scale).round().clamp(-127, 127)
        dec = decoder_sd_dec[name].detach().cpu().float()
        if scale > 0:
            dec_q = (dec / scale).round().clamp(-127, 127)
            if not torch.allclose(orig_q, dec_q):
                raise RuntimeError(f"Codec round-trip FAILED for {name}")

    # Copy best_meta.json forward, augmented with the final archive size.
    meta_path = prev_stage_output_dir / "best_meta.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        meta['final_archive_bytes'] = archive_bytes
        with open(final_output_dir / "final_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    return {
        'final_archive_bytes': archive_bytes,
        'archive_path': str(final_output_dir / "0.bin"),
    }
