#!/usr/bin/env python3
"""tcm_ctxmix inflater:  python inflate.py <archive_dir> <output_dir> <video_names_file>

<archive_dir>/<member> (see tcx_format.py; the member name is the layout version) -> <output_dir>/0.raw, headerless uint8 RGB [1200, 874, 1164, 3].

  1. parse the framing, start the TCM decoder (native, built by inflate.sh; path in $TCM_BIN) on the TCM stream
     -> 600 token maps
  2. meanwhile: decode the non-token sections (tcx_format.unpack), restore #153's renderer + carrier and check their
     sha256, render the even frames (carrier + EXT1 + frame-0 selector)
  3. wait for the tokens, check their sha256, render the odd frames with #153's SemanticTokenRenderer
Any failure ends with a non-zero exit, and the TCM decoder is not left running.
Device: cuda, as in #153.  Without CUDA the script stops unless INFLATE_DEVICE=cpu is set on purpose.
"""
from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
PR141 = VENDOR / "external" / "pr141"
for path in (PR141, VENDOR, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import tcx_format  # noqa: E402
import tcx_models  # noqa: E402
import tcx_render  # noqa: E402

N_MAPS, MAP_H, MAP_W = 600, 384, 512
# sha256 of PR #153's 600 token maps: the decoded tokens must be exactly these before a frame is rendered from them.
# (TCM_EXPECT_TOKEN_SHA256 overrides the value, for experiments with other token maps.)
EXPECTED_TOKEN_SHA256 = (os.environ.get("TCM_EXPECT_TOKEN_SHA256")
                         or "d64c2a094dbdcdf6b381c94810f9d672ac33c3c960eb6ae35ba09df0af51c3a1")


def log(msg):
    print(f"[tcm_ctxmix {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_device():
    name = os.environ.get("INFLATE_DEVICE")
    if not name:
        if not torch.cuda.is_available():
            raise SystemExit("ERROR: CUDA is not available. The frames are rendered on the GPU, as in PR #153; set "
                             "INFLATE_DEVICE=cpu to render on the CPU on purpose (slow, about 9 GB of RAM, and the "
                             "frames are not bit-identical to the GPU frames)")
        name = "cuda"
    device = torch.device(name)
    if device.type == "cuda":
        # #153's numerics: decoder/numerics.py (cuDNN on, TF32 off) + runtime.hpac_inference.configure_cuda_reproducibility,
        # which #153 applied (inside its token decoder) before rendering.
        from decoder import numerics  # noqa: F401  (import applies the flags)
        from runtime.hpac_inference import configure_cuda_reproducibility
        configure_cuda_reproducibility()
    else:
        torch.set_num_threads(int(os.environ.get("INFLATE_THREADS", max(1, (os.cpu_count() or 2) - 1))))
    return device


def inflate_one(archive_dir: Path, out_raw: Path, scratch: Path):
    t0 = time.time()
    device = pick_device()                                 # stops here, before anything is started, without a device
    cannot_parse = f"ERROR: cannot parse the archive in {archive_dir} ({tcx_format.LAYOUT} layout)"
    try:
        payload = tcx_format.read_payload(archive_dir)
        tcm_stream = tcx_format.tcm_stream(payload)        # framing only, so that the token decoder can start at once
    except (ValueError, IndexError, struct.error) as error:
        raise SystemExit(f"{cannot_parse}: {error}")

    tcm_in, tcm_out = scratch / "tokens.tcm", scratch / "tokens.u8"
    tcm_in.write_bytes(tcm_stream)
    tcm_bin = os.environ.get("TCM_BIN")
    if not tcm_bin or not Path(tcm_bin).is_file():
        raise SystemExit("TCM_BIN is not set to the compiled TCM decoder (run through inflate.sh)")
    tcm_proc = subprocess.Popen([tcm_bin, "d", str(tcm_in), str(tcm_out)])
    log(f"TCM decode started ({len(tcm_stream)} B stream)")
    try:
        render(payload, tcm_proc, tcm_out, out_raw, device, t0, cannot_parse)
    finally:
        if tcm_proc.poll() is None:      # an error above: do not leave the token decoder running
            tcm_proc.kill()
            tcm_proc.wait()


def render(payload: bytes, tcm_proc: subprocess.Popen, tcm_out: Path, out_raw: Path, device, t0: float, cannot_parse: str):
    """Everything that happens while and after the token decoder runs: decode the non-token sections, render the even
    frames, wait for the tokens, render the odd frames."""
    os.chdir(VENDOR)   # a few vendored runtime modules resolve paths relative to the vendor root (as in #153)
    r = tcx_render.load_renderer_module(PR141)
    try:
        sections = tcx_format.unpack(payload)              # about a second of integer Python (tcx_ncm.py)
        semantic_blob = tcx_models.restore_semantic(sections)   # the restore steps check the sha256 of what was decoded
        carrier_blob = tcx_models.restore_carrier(sections)
        ext1 = tcx_format.decode_ext1(sections)
    except (ValueError, IndexError, struct.error) as error:
        raise SystemExit(f"{cannot_parse}: {error}")
    semantic = tcx_render.build_semantic(r, semantic_blob)
    basis, coefficients, selector_blob = tcx_render.build_carrier(r, carrier_blob)
    t_models = time.time()
    log(f"renderer + carrier restored in {t_models - t0:.1f} s; device {device}")

    out_raw.parent.mkdir(parents=True, exist_ok=True)
    if out_raw.exists():
        out_raw.unlink()
    output = np.memmap(out_raw, mode="w+", dtype=np.uint8, shape=(2 * r.N, r.CAMERA_H, r.CAMERA_W, 3))
    tcx_render.render_even(r, basis, coefficients, ext1, selector_blob, output, device)
    t_even = time.time()
    log(f"even frames rendered in {t_even - t_models:.1f} s")

    if tcm_proc.wait() != 0:
        raise SystemExit(f"TCM decoder failed with exit code {tcm_proc.returncode}")
    t_tcm = time.time()
    raw_tokens = tcm_out.read_bytes()
    if len(raw_tokens) != N_MAPS * MAP_H * MAP_W:
        raise SystemExit(f"TCM decoder wrote {len(raw_tokens)} bytes, expected {N_MAPS * MAP_H * MAP_W}")
    token_sha = hashlib.sha256(raw_tokens).hexdigest()
    if token_sha != EXPECTED_TOKEN_SHA256:
        raise SystemExit(f"ERROR: token sha256 {token_sha} != expected {EXPECTED_TOKEN_SHA256}; no odd frame is rendered")
    tokens = torch.from_numpy(np.frombuffer(raw_tokens, dtype=np.uint8).reshape(N_MAPS, MAP_H, MAP_W).copy())
    log(f"tokens ready (waited {t_tcm - t_even:.1f} s after even frames; wall since start {t_tcm - t0:.1f} s) sha256 {token_sha}")

    tcx_render.render_odd(r, semantic, tokens, output, device)
    output.flush()
    del output
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.time()
    log(f"odd frames rendered in {t_end - t_tcm:.1f} s; total {t_end - t0:.1f} s")


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: inflate.py <archive_dir> <output_dir> <video_names_file>")
    archive_dir, output_dir, names_file = (Path(a).resolve() for a in sys.argv[1:])
    output_dir.mkdir(parents=True, exist_ok=True)
    for line in names_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        base = line.rsplit(".", 1)[0]
        if base != "0":
            raise SystemExit(f"unsupported video: {line} (this archive encodes the single public video 0)")
        with tempfile.TemporaryDirectory() as scratch:
            inflate_one(archive_dir, output_dir / f"{base}.raw", Path(scratch))


if __name__ == "__main__":
    main()
