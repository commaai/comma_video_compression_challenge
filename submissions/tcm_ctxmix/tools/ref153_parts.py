#!/usr/bin/env python3
"""Parse #153's archive with #153's own code (no rendering) and dump the decoder-side components that
build_archive.py compares with: semantic_blob, carrier_blob, EXT1 dict.

usage: python -I -B tools/ref153_parts.py <tools/hypery11_carrier_joint> <#153 archive.zip> <out_dir>
Writes <out_dir>/ref153_parts.json and <out_dir>/ref153_parts.pkl.
"""
import hashlib
import json
import os
import pickle
import sys
import tempfile
import zipfile
from pathlib import Path

sub, archive, out = (Path(x).resolve() for x in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(sub))
import verify  # noqa: E402  (#153: restores the #151 payload and checks the provenance hashes)
reference = json.loads((sub / "provenance.json").read_text())
with zipfile.ZipFile(archive) as z:
    restored = verify.restore_payload(z.read("p"), reference)
for m in ("verify", "packing"):
    sys.modules.pop(m, None)
sys.path.remove(str(sub))
vendor = sub / "vendor"
os.chdir(vendor)
sys.path.insert(0, str(vendor))
sys.path.insert(0, str(vendor / "external/pr141"))
from decoder import inflate_core, extfmt  # noqa: E402
with tempfile.TemporaryDirectory() as t:
    za = Path(t) / "archive.zip"
    with zipfile.ZipFile(za, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("p", restored)
    parts, warp, payload = inflate_core._load(za)
exts = extfmt.parse_payload(payload)
sha = lambda b: hashlib.sha256(b).hexdigest()
res = dict(restored_payload_sha256=sha(restored), restored_payload_bytes=len(restored),
           semantic_blob=dict(bytes=len(parts.semantic_blob), sha256=sha(parts.semantic_blob)),
           carrier_blob=dict(bytes=len(parts.carrier_blob), sha256=sha(parts.carrier_blob)),
           ext1=dict(fam=exts["ext1"]["fam"], bits=exts["ext1"]["bits"], step=exts["ext1"]["step"],
                     ids=sorted(exts["ext1"]["q"]), q={str(k): v for k, v in sorted(exts["ext1"]["q"].items())}))
(out / "ref153_parts.json").write_text(json.dumps(res, indent=1))
with open(out / "ref153_parts.pkl", "wb") as f:
    pickle.dump(dict(semantic_blob=parts.semantic_blob, carrier_blob=parts.carrier_blob, ext1=exts["ext1"]), f)
print(json.dumps({k: v for k, v in res.items() if k != "ext1"}, indent=1))
