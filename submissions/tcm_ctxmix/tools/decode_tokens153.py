#!/usr/bin/env python3
"""Decode the 600 token maps of PR #153's archive with #153's own decoder (HPAC model + RC64 token stream).

usage: python -I -B tools/decode_tokens153.py <tools/hypery11_carrier_joint> <#153 archive.zip> <out tokens.u8> [cpu|cuda]
Needs CPR1_RC64_LIBRARY and F26_CORRECTOR_NATIVE_LIBRARY (native libraries built from #153's
vendor/external/pr141 sources with the flags of #153's vendor/inflate.sh; compress.sh builds them).
Steps, as in #153's inflate path: verify.restore_payload (#151-layout payload, checked against provenance.json)
-> single-member zip -> canonical shims -> container.token_decode.decode_tokens.
Writes the maps as raw uint8 [600, 384, 512] and exits non-zero unless their sha256 is TOKEN_SHA256.
"""
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

TOKEN_SHA256 = "d64c2a094dbdcdf6b381c94810f9d672ac33c3c960eb6ae35ba09df0af51c3a1"

sub, archive, out = (Path(x).resolve() for x in sys.argv[1:4])
device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
for name in ("CPR1_RC64_LIBRARY", "F26_CORRECTOR_NATIVE_LIBRARY"):
    if not Path(os.environ.get(name, "")).is_file():
        raise SystemExit(f"ERROR: {name} must point to the compiled library")
vendor = sub / "vendor"

sys.path.insert(0, str(sub))
import verify  # noqa: E402  (#153)
reference = json.loads((sub / "provenance.json").read_text())
with zipfile.ZipFile(archive) as z:
    payload = verify.restore_payload(z.read("p"), reference)   # raises unless the components equal provenance.json
sys.path.remove(str(sub))
for m in ("packing", "verify", "carrier_pack", "carrier_joint", "tail_pack", "hpac_codec", "renderer_pack", "hpac_range", "hpac_layout"):
    sys.modules.pop(m, None)

sys.path.insert(0, str(vendor))
os.chdir(vendor)  # as #153's vendor/inflate.py: some runtime modules resolve paths relative to it
from canonical.inflate_canonical import install_rp1_shim  # noqa: E402
install_rp1_shim()
from decoder import numerics  # noqa: E402,F401  (same import order as #153's canonical inflate)
from container import token_decode  # noqa: E402

with tempfile.TemporaryDirectory() as t:
    restored = Path(t) / "restored153.zip"
    with zipfile.ZipFile(restored, "w", zipfile.ZIP_STORED) as z:
        z.writestr("p", payload)
    tokens, info = token_decode.decode_tokens(restored, vendor / "external/pr141", device)
raw = tokens.numpy().astype("uint8", copy=False).tobytes()
digest = hashlib.sha256(raw).hexdigest()
print(json.dumps({"device": device, "decode_seconds": round(info["decode_runtime_seconds"], 1), "shape": list(tokens.shape),
                  "token_sha256": digest}, indent=1))
if digest != TOKEN_SHA256:
    raise SystemExit(f"ERROR: decoded token maps have sha256 {digest}, expected {TOKEN_SHA256}")
out.write_bytes(raw)
