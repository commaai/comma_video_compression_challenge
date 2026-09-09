"""Turn a training checkpoint into a submission dir the official harness accepts."""
import argparse, shutil, subprocess, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import WORK, REPO

INFLATE_SH = """#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"; OUTPUT_DIR="$2"; FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  python "$HERE/inflate.py" "$DATA_DIR/model.bin" "$OUTPUT_DIR/$BASE.raw"
done < "$FILE_LIST"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--archive", default="best_archive.bin")
    args = ap.parse_args()

    src = WORK / "runs" / args.run / args.archive
    out = REPO / "submissions" / args.name
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy(WORK / "inflate_template.py", out / "inflate.py")
    (out / "inflate.sh").write_text(INFLATE_SH)
    (out / "inflate.sh").chmod(0o755)

    zp = out / "archive.zip"
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_STORED) as z:
        z.write(src, "model.bin")

    print(f"submission at {out}  archive.zip = {zp.stat().st_size:,} bytes "
          f"(rate term {25*zp.stat().st_size/37545489:.4f})")


if __name__ == "__main__":
    main()
