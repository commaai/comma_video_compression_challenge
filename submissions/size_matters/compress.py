import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import packing


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--download-reference', action='store_true')
    parser.add_argument('--output', type=Path, default=HERE / 'archive.zip')
    args = parser.parse_args()
    reference = json.loads((HERE / 'provenance.json').read_text())
    if args.download_reference:
        cache = HERE / 'work'
        cache.mkdir(exist_ok=True)
        args.source = cache / 'reference.zip'
        urllib.request.urlretrieve(reference['reference_archive_url'], args.source)
    if args.source is None:
        parser.error('Use --source archive.zip or --download-reference')
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != reference['reference_archive_sha256']:
        raise ValueError('The reference archive hash does not match')
    with zipfile.ZipFile(args.source) as archive:
        data = archive.read('p')
    mini = packing.encode(data)
    if packing.component_hashes(packing.decode(mini)) != packing.component_hashes(data):
        raise ValueError('Lossless round trip failed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    member = zipfile.ZipInfo('p', date_time=(1980, 1, 1, 0, 0, 0))
    member.compress_type = zipfile.ZIP_STORED
    member.external_attr = 0o100644 << 16
    with zipfile.ZipFile(args.output, 'w') as archive:
        archive.writestr(member, mini)
    print(f'Archive: {args.output}')
    print(f'Size: {args.output.stat().st_size:,} bytes')
    print(f'Saved: {len(source) - args.output.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
