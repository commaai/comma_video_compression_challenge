import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('submission', type=Path)
    parser.add_argument('videos', type=Path)
    parser.add_argument('elapsed', type=int)
    parser.add_argument('--target', type=float, default=0.125)
    args = parser.parse_args()
    report = (args.submission / 'report.txt').read_text()

    def number(label):
        match = re.search(re.escape(label) + r'\s*([0-9.eE+-]+)', report)
        if not match:
            raise ValueError(f'Missing report field: {label}')
        return float(match.group(1))

    samples = re.search(r'results over (\d+) samples', report)
    if not samples or int(samples.group(1)) != 600:
        raise ValueError('Expected all 600 samples')
    raw_bytes = (args.submission / 'inflated/0.raw').stat().st_size
    if raw_bytes != 3_662_409_600:
        raise ValueError(f'Unexpected raw file size: {raw_bytes}')
    original_bytes = sum(p.stat().st_size for p in args.videos.rglob('*') if p.is_file())
    if original_bytes != 37_545_489:
        raise ValueError(f'Unexpected original byte count: {original_bytes}')
    archive = (args.submission / 'archive.zip').read_bytes()
    pose = number('Average PoseNet Distortion:')
    seg = number('Average SegNet Distortion:')
    score = 100 * seg + math.sqrt(10 * pose) + 25 * len(archive) / original_bytes
    result = {
        'status': 'measured_gpu',
        'samples': 600,
        'pose_distortion': pose,
        'seg_distortion': seg,
        'archive_bytes': len(archive),
        'archive_sha256': hashlib.sha256(archive).hexdigest(),
        'original_bytes': original_bytes,
        'raw_bytes': raw_bytes,
        'score': score,
        'evaluation_seconds': args.elapsed,
        'target': args.target,
        'under_target': score < args.target,
    }
    (args.submission / 'measured_score.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['under_target']:
        raise SystemExit(f'The measured score did not beat {args.target}')


if __name__ == '__main__':
    main()
