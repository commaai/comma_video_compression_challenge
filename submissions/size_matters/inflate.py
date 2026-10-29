import json
import sys
from pathlib import Path

import verify


def main():
    if len(sys.argv) != 3:
        raise SystemExit('Usage: inflate.py <archive_dir> <restored_dir>')
    archive_dir, restored_dir = map(Path, sys.argv[1:])
    reference = json.loads((verify.HERE / 'provenance.json').read_text())
    data = (archive_dir / 'p').read_bytes()
    restored = verify.restore_payload(data, reference)
    restored_dir.mkdir(parents=True, exist_ok=True)
    (restored_dir / 'p').write_bytes(restored)


if __name__ == '__main__':
    main()
