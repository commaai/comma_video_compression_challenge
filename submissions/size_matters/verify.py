import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import packing


HERE = Path(__file__).resolve().parent


def restore_payload(mini, reference):
    data = packing.decode(mini)
    if packing.component_hashes(data) != reference['reference_components']:
        raise ValueError('Restored model or video components differ from the reference')
    return data


def restore(archive, reference):
    with zipfile.ZipFile(archive) as source:
        if source.namelist() != ['p']:
            raise ValueError('Expected a single payload named p')
        return restore_payload(source.read('p'), reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=HERE / 'archive.zip')
    parser.add_argument('--restore-dir', type=Path)
    parser.add_argument('--target', type=float, default=0.125)
    args = parser.parse_args()
    reference = json.loads((HERE / 'provenance.json').read_text())
    data = restore(args.archive, reference)
    hashes = json.loads((HERE / 'vendor_hashes.json').read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Decoder source changed: {name}')
    size = args.archive.stat().st_size
    upper = (100 * (0.00012544 + 5e-9) + math.sqrt(10 * (0.00000133 + 5e-9))
             + 25 * size / 37_545_489)
    result = {
        'status': 'lossless_verified_gpu_evaluation_pending',
        'archive_bytes': size,
        'saved_bytes': reference['reference_archive_bytes'] - size,
        'restored_payload_sha256': hashlib.sha256(data).hexdigest(),
        'reference_components_identical': True,
        'component_hashes': packing.component_hashes(data),
        'unchanged_decoder_files': len(hashes),
        'inferred_score_upper_bound': upper,
        'target': args.target,
        'inferred_under_target': upper < args.target,
        'score_basis': 'Published reference distortions; identical decoded HPAC, renderer, carrier, header, residual, tokens and warp bytes; not a new GPU measurement.',
    }
    baseline = reference.get('parity_baseline')
    if baseline:
        result['derived_score_from_previous_gpu_distortions'] = (100 * baseline['seg_distortion']
            + math.sqrt(10 * baseline['pose_distortion']) + 25 * size / baseline['original_bytes'])
        result['derived_under_target'] = result['derived_score_from_previous_gpu_distortions'] < args.target
    measured_path = HERE / 'measured_score.json'
    if measured_path.is_file():
        measured = json.loads(measured_path.read_text())
        digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
        if measured.get('archive_sha256') == digest and measured.get('samples') == 600:
            result['status'] = 'lossless_verified_with_saved_gpu_result'
            result['saved_gpu_score'] = measured['score']
            result['saved_gpu_evaluation_seconds'] = measured['evaluation_seconds']
            result['score_basis'] = 'Saved 600-sample CUDA evaluation of this exact archive; local restoration and source checks passed.'
    if args.restore_dir:
        args.restore_dir.mkdir(parents=True, exist_ok=True)
        (args.restore_dir / 'p').write_bytes(data)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
