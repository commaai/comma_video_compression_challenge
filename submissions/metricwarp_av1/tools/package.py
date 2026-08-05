#!/usr/bin/env python
"""Package a submission: archive.zip (stream + manifest + corrections) + inflate scripts.

Usage: python package.py --name exactgrid_av1 --stream work/enc/FINAL.ivf \
         [--corrections work/corr_final.npz] [--dec-down none|bilinear|area] [--skip-first 0|1]
"""
import argparse, json, shutil, zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CORR_SCALES = np.array([0.125, 0.125, 0.5, 1.0, 0.25, 2.0], dtype=np.float32)  # dx,dy,rot,zoom,bias,gain quant steps

def quantize_corrections(npz_path):
    d = np.load(npz_path)
    P = d['params']  # (600,6) floats
    q = np.round(P / CORR_SCALES).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, (q.astype(np.float32) * CORR_SCALES)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--stream', required=True)
    ap.add_argument('--corrections', default=None)
    ap.add_argument('--chain', default='fullres', choices=['fullres', 'grid512', 'hybrid', 'hybrid2'])
    ap.add_argument('--segfix', default=None, help='segfix npz (blob + dirtable)')
    ap.add_argument('--dec-down', default='bilinear')
    ap.add_argument('--skip-first', type=int, default=0)
    args = ap.parse_args()

    sub = ROOT / 'submissions' / args.name
    sub.mkdir(parents=True, exist_ok=True)

    stream = Path(args.stream)
    man = {
        'streams': {'0': stream.name},
        'chain': args.chain,
        'dec_down': args.dec_down,
        'skip_first': args.skip_first,
    }
    members = [(stream, stream.name)]

    if args.corrections:
        import brotli
        q, _ = quantize_corrections(args.corrections)
        raw = brotli.compress(q.tobytes(), quality=11)
        corr_name = 'corr.bin'
        (HERE / 'corr_pack.bin').write_bytes(raw)
        members.append((HERE / 'corr_pack.bin', corr_name))
        man['corrections'] = corr_name
        man['corr_scales'] = [float(x) for x in CORR_SCALES]

    if args.segfix:
        import brotli
        d = np.load(args.segfix)
        raw = brotli.compress(d['blob'].tobytes(), quality=11)
        (HERE / 'segfix_pack.bin').write_bytes(raw)
        members.append((HERE / 'segfix_pack.bin', 'segfix.bin'))
        man['segfix'] = 'segfix.bin'
        man['segfix_dirtable'] = [round(float(x), 4) for x in d['dirtable'].ravel()]
        man['segfix_amps'] = [6, 12, 18]

    manp = HERE / 'manifest.json'
    manp.write_text(json.dumps(man))
    members.append((manp, 'manifest.json'))

    zp = sub / 'archive.zip'
    with zipfile.ZipFile(zp, 'w', compression=zipfile.ZIP_STORED) as z:
        for src, arc in members:
            z.write(src, arc)
    print(f"archive.zip: {zp.stat().st_size:,} bytes")

    shutil.copy(HERE / 'inflate_template.py', sub / 'inflate.py')
    (sub / 'inflate.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"; OUTPUT_DIR="$2"; FILE_LIST="$3"
python "${HERE}/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"
''')
    print(f"submission dir ready: {sub}")

if __name__ == '__main__':
    main()
