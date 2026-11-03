#!/usr/bin/env python3
"""Submission inflater.

Usage:
    python inflate.py <archive_dir> <output_dir> <video_names_file>

<archive_dir> is the extracted archive (it holds the zip member `p`, or `archive.zip` itself).  Writes <output_dir>/<video_stem>.raw, a headerless uint8 RGB dump of shape
(1200, 874, 1164, 3).  Only the single public video `0` is supported.  Decoding goes through `canonical/inflate_canonical.py`, i.e. the integrated decoder (`decoder/`) with the
RP1 HPAC rowpack shim registered (plain-IHS1 archives decode unchanged). 

Needs CUDA; inflate.sh builds the native RC64 / corrector libraries from external/pr141 and exports their paths.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from canonical.inflate_canonical import _archive_from_dir, inflate  # noqa: E402


def main() -> None:
    if len(sys.argv) != 4:
        print("usage: inflate.py <archive_dir> <output_dir> <video_names_file>", file=sys.stderr)
        raise SystemExit(2)
    archive_dir, output_dir, names_file = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)
    for line in names_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        base = line.rsplit(".", 1)[0]
        if base != "0":
            raise SystemExit(f"unsupported public video: {line}")
        with tempfile.TemporaryDirectory() as scratch:
            info = inflate(_archive_from_dir(archive_dir, Path(scratch)), output_dir / f"{base}.raw")
        print({k: info[k] for k in ("token_sha", "hpac_sha") if k in info}, flush=True)


if __name__ == "__main__":
    main()
