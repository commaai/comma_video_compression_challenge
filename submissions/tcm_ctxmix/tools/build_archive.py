#!/usr/bin/env python3
"""Build archive.zip = PR #153's non-token sections + a TCM token stream (container layout tcx_format.LAYOUT).

usage: python -I -B tools/build_archive.py --tcm <tokens.tcm> --pr153-archive <#153 archive.zip>
           --pr153-dir <tools/hypery11_carrier_joint> --ref-parts <ref153_parts.json> --out <archive.zip> [--tcm-bin <tcm>]

Deterministic: same inputs -> byte-identical archive.zip (fixed zip metadata, one stored member).
Taken from #153's archive: its renderer section (RDQ2), its carrier section (CGA2), the RX1M reserved flag byte, the
4-byte EXT1 grid header and the tail_pack coder descriptor; tcx_format.pack decides how they are stored.
The EXT1 ids + grid values are decoded from #153's joint tail block and re-encoded alone with the same coder.
Everything else in #153's archive is token-only and dropped; the TCM stream replaces it.
Checks (the build fails if one does not hold): #153's joint block re-encodes exactly; the EXT1 block round-trips;
the archive parses back to #153's renderer section and carrier body; restored renderer/carrier blobs and EXT1 equal
#153's own decoder output (--ref-parts, from ref153_parts.py); the decoder digests (tcx_format.EXPECTED_SHA256) equal
#153's data; optional TCM decode check (--tcm-bin) against the token sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import struct
import sys
import zipfile
from pathlib import Path

SUB = Path(__file__).resolve().parents[1]
TOKEN_SHA256 = "d64c2a094dbdcdf6b381c94810f9d672ac33c3c960eb6ae35ba09df0af51c3a1"
SMP4 = struct.Struct("<4sHHHBB")          # #153 packing.HEADER
TAIL = struct.Struct("<BH2s3s4sH")        # #153 tail_pack.HEADER


def sha(b): return hashlib.sha256(b).hexdigest()


def zip_bytes(payload: bytes, method: int, member: str) -> bytes:
    """One member (its name is the layout version tag, tcx_format.MEMBER), fixed metadata (same ZipInfo convention as
    #153 vendor/container/payload_format.write_archive)."""
    info = zipfile.ZipInfo(filename=member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = method
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(info, payload, compress_type=method, compresslevel=9 if method == zipfile.ZIP_DEFLATED else None)
    return buf.getvalue()


def split_153(p: bytes):
    magic, nh, nr, nc, table_mode, reserved = SMP4.unpack_from(p)
    if magic != b"SMP4":
        raise ValueError("not a #153 SMP4 payload")
    o = SMP4.size
    hpac = p[o:o + nh]; o += nh
    renderer = p[o:o + nr]; o += nr
    carrier = p[o:o + nc]; o += nc
    tail = p[o:]
    np2, ncj, scale, warphdr, gridhdr, descriptor = TAIL.unpack_from(tail)
    nt = len(tail) - TAIL.size - np2 - ncj
    tokens = tail[TAIL.size:TAIL.size + nt]
    prev2 = tail[TAIL.size + nt:TAIL.size + nt + np2]
    joint = tail[TAIL.size + nt + np2:]
    return dict(hpac=hpac, renderer=renderer, carrier=carrier, table_mode=table_mode, reserved=reserved,
                scale=scale, warphdr=warphdr, gridhdr=gridhdr, descriptor=descriptor, tokens=tokens, prev2=prev2, joint=joint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcm", type=Path, required=True)
    ap.add_argument("--pr153-archive", type=Path, required=True)
    ap.add_argument("--pr153-dir", type=Path, required=True)
    ap.add_argument("--ref-parts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tcm-bin", type=Path)
    a = ap.parse_args()

    sys.path.insert(0, str(SUB))                      # tcx_format (+ SUB/pr153 for hpac_range)
    sys.path[:0] = [str(SUB / "vendor/external/pr141"), str(SUB / "vendor")]   # vendored runtime
    import tcx_format as fmt
    sys.path.append(str(a.pr153_dir))                 # #153 tail_pack (decoder side of its joint block)
    import tail_pack
    import hpac_range
    assert Path(hpac_range.__file__).resolve() == (SUB / "pr153/hpac_range.py").resolve()
    assert sha((SUB / "pr153/hpac_range.py").read_bytes()) == sha((a.pr153_dir / "hpac_range.py").read_bytes())

    src_zip = a.pr153_archive.read_bytes()
    with zipfile.ZipFile(a.pr153_archive) as z:
        assert z.namelist() == ["p"]
        p153 = z.read("p")
    s = split_153(p153)

    # --- 1. EXT1 part of #153's joint block ---
    d = s["descriptor"]
    full_settings = (d & 255, (d >> 8) & 3, (d >> 10) & 7, (d >> 13) & 3)
    coeffs, warp, ids, grids = tail_pack.transform(hpac_range.Decoder(s["joint"]), full_settings,
                                                   gridcount=s["gridhdr"][3], bits=s["gridhdr"][1])
    e = hpac_range.Encoder()
    tail_pack.transform(e, full_settings, list(coeffs), list(warp), list(ids), list(grids), len(ids), s["gridhdr"][1])
    assert e.finish() == s["joint"], "#153 joint block does not re-encode exactly"
    ext1_block = fmt.encode_ext1(ids, grids, s["gridhdr"], s["descriptor"])

    # --- 2. TCM stream ---
    tcm = a.tcm.read_bytes()
    assert tcm[:1] == b"T" and int.from_bytes(tcm[2:4], "little") == 600, "not a 600-map TCM stream"

    sections = fmt.Sections(renderer=s["renderer"], carrier=s["carrier"], rx1_reserved=s["reserved"],
                            ext1_header=s["gridhdr"], ext1_descriptor=s["descriptor"], ext1_block=ext1_block, tcm=tcm)
    layout = fmt.LAYOUT
    p = fmt.pack(sections, layout)
    member = fmt.MEMBER[layout]
    stored, deflated = zip_bytes(p, zipfile.ZIP_STORED, member), zip_bytes(p, zipfile.ZIP_DEFLATED, member)
    archive = stored if len(stored) <= len(deflated) else deflated

    # --- 3. verification ---
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert z.namelist() == [member] and z.read(member) == p
    back = fmt.unpack(p, layout)
    import tcx_models
    norm = lambda x: (fmt.renderer_section_153(x), tcx_models.carrier_body(x), x.rx1_reserved, x.ext1_header,
                      fmt.descriptor_settings(x.ext1_descriptor), x.ext1_block, x.tcm)
    assert norm(back) == norm(sections)
    ext1 = fmt.decode_ext1(back)
    assert sorted(ext1["q"]) == list(ids) and [v for k in sorted(ext1["q"]) for v in ext1["q"][k]] == list(grids)
    checks = {"pr153_joint_block_reencodes_exactly": True, "ext1_block_roundtrip": True, "archive_parses_back": True}
    # the restored renderer, carrier and EXT1 must equal what #153's own decoder gets from #153's archive
    ref = json.loads(a.ref_parts.read_text())
    ref_pkl = pickle.loads(a.ref_parts.with_suffix(".pkl").read_bytes())
    sem, car = tcx_models.restore_semantic(back), tcx_models.restore_carrier(back)
    from folding import rid9
    ref_blob = ref_pkl["semantic_blob"]
    if back.renderer_gaps is None:
        checks["semantic_blob_equals_pr153"] = sha(sem) == ref["semantic_blob"]["sha256"]
    else:       # the decoder builds the mode-8 body directly; compare with #153's own RID9 -> mode-8 parse
        checks["semantic_blob_equals_pr153"] = sha(ref_blob) == ref["semantic_blob"]["sha256"] \
            and sem == rid9.decode_to_blob8(ref_blob)
    # the sha256 digests that the decoder checks (tcx_format.EXPECTED_SHA256) must be those of #153's own data
    import carrier_joint
    digests_153 = {"renderer body": sha(rid9.decode_to_blob8(ref_blob)), "carrier body": sha(carrier_joint.decode(s["carrier"])),
                   "EXT1 block": sha(bytes(s["gridhdr"]) + ext1_block)}
    checks["decoder_digests_equal_pr153"] = sha(ref_blob) == ref["semantic_blob"]["sha256"] and fmt.EXPECTED_SHA256 == digests_153
    checks["carrier_blob_equals_pr153"] = sha(car) == ref["carrier_blob"]["sha256"]
    checks["ext1_equals_pr153"] = (ext1["fam"], ext1["bits"], ext1["step"]) == (ref["ext1"]["fam"], ref["ext1"]["bits"], ref["ext1"]["step"]) \
        and {str(k): v for k, v in ext1["q"].items()} == ref["ext1"]["q"]
    if a.tcm_bin:
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "x.tcm").write_bytes(back.tcm)
            subprocess.run([str(a.tcm_bin), "d", str(Path(t) / "x.tcm"), str(Path(t) / "x.u8")], check=True)
            checks["tcm_decodes_to_token_sha256"] = sha((Path(t) / "x.u8").read_bytes()) == TOKEN_SHA256
    assert all(checks.values()), checks

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(archive)
    summary = dict(archive_bytes=len(archive), archive_sha256=sha(archive), layout=layout, member=member,
                   zip_method="stored" if archive is stored else "deflate", payload_bytes=len(p), tcm_bytes=len(tcm),
                   tcm_sha256=sha(tcm), ext1_block_bytes=len(ext1_block), pr153_archive_bytes=len(src_zip),
                   pr153_archive_sha256=sha(src_zip), checks=checks)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
