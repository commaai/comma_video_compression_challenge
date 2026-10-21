#!/usr/bin/env python3
"""Canonical decoder wrapper (HPAC rowpack + RID1/RID2 shims).

Wraps the integrated decoder (`decoder/inflate_core.py`: archive.zip -> tokens -> frames -> uint8 .raw) and registers the RP1 HPAC rowpack
shim BEFORE the decoder runs.  Plain-IHS1 archives decode exactly as before (the shim passes them through); RP1 archives
have their HPAC weight region restored to canonical IHS1 before any frozen runtime consumer sees it.  Fail closed: a malformed rowpack raises.

Entry points
    python -m canonical.inflate_canonical <archive.zip> <out.raw>                     direct
    python canonical/inflate_canonical.py --data-dir <dir> <base> <dest.raw>          harness style (dir has `archive.zip` or the unzipped member `p`)
    inflate(archive_zip, out_raw)                                                      library call
Numerics: imports decoder.numerics (official decoder numerics: cuDNN on, TF32 off) exactly like the validated path.  The native RC64 / corrector libraries are taken from
CPR1_RC64_LIBRARY / F26_CORRECTOR_NATIVE_LIBRARY if set (inflate.sh builds them from external/pr141 with -ffp-contract=off); otherwise the default library locations are used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")


def install_rp1_shim() -> None:
    """Install order (each hook is idempotent and independent): (1) RP1 HPAC rowpack hook (container.token_decode POST_PURGE_HOOKS); (2) RID1/RID2 (SM3R mode 7/8) renderer-section shim
    wrapping runtime.f26_inflate._load_renderer -- the single implementation lives in folding.renderer_shim (-> folding.rid1 / folding.rid2); then the caller
    delegates to the integrated decoder.  Archives without RID sections pass through the original renderer parser unchanged."""
    from hpac import rowpack
    rowpack.install_decoder_shim()
    from folding import renderer_shim
    renderer_shim.install()


def inflate(archive_zip: Path | str, out_raw: Path | str) -> dict:
    """RP1-aware inflate of one archive.zip to a headerless uint8 RGB .raw file ([1200, 874, 1164, 3]). Returns the decoder info dict."""
    install_rp1_shim()
    from decoder import numerics as _official_numerics  # noqa: F401  (import applies the official cuDNN/TF32 flags)
    from decoder import inflate_core
    return inflate_core.inflate(Path(archive_zip), Path(out_raw))


def _archive_from_dir(data_dir: Path, scratch: Path) -> Path:
    zipped = data_dir / "archive.zip"
    if zipped.is_file():
        return zipped
    payload = data_dir / "p"
    if not payload.is_file():
        raise FileNotFoundError(f"neither {zipped} nor {payload} exists")
    rebuilt = scratch / "archive.zip"
    with zipfile.ZipFile(rebuilt, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("p", payload.read_bytes())
    return rebuilt


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("positional", nargs="+")
    args = ap.parse_args(argv)
    if args.data_dir is not None:
        if len(args.positional) != 2:
            ap.error("with --data-dir: <base> <dest.raw>")
        base, dest = args.positional
        if base != "0":
            raise ValueError(f"unsupported public video: {base}")
        with tempfile.TemporaryDirectory() as scratch:
            info = inflate(_archive_from_dir(args.data_dir, Path(scratch)), Path(dest))
    else:
        if len(args.positional) != 2:
            ap.error("<archive.zip> <out.raw>")
        info = inflate(Path(args.positional[0]), Path(args.positional[1]))
    print(json.dumps(info, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
