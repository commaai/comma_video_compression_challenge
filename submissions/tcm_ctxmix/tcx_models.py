"""Restore the renderer and carrier blobs that #153's decoder consumes, from the tcm_ctxmix sections.

The renderer and carrier sections are #153's own (renderer_pack / carrier_joint).  After those two decoders the
bytes are #151's RX1M renderer and carrier sections; this module then performs exactly the semantic/carrier steps of
#151's `runtime.residual_archive._decode_rx1_models(..., block_models=True)` (vendor/external/pr141), calling the
same vendored helpers, and returns what `read_residual_archive()` returns as parts.semantic_blob / parts.carrier_blob.
The HPAC section of that function is not needed (no HPAC model is used any more).

Both restore functions compare what they decoded with the sha256 of PR #153's own data (tcx_format.check_component)
and raise ValueError on a mismatch, so a damaged archive cannot reach the renderer.
"""
from __future__ import annotations

from tcx_format import Sections, check_component

import carrier_joint   # pr153/ (on sys.path via tcx_format)
import renderer_pack   # pr153/

from runtime import residual_archive as ra
from runtime.carrier_repack import pack_frame0_selector_carrier
from runtime.compensation_overlay import split_selector_compensation
from runtime.dx2_cabac_coefficients import restore_carrier_body as restore_dx2_carrier_body
from runtime.entropy.coefficient_ar1_codec import decode_cap1
from runtime.frame0_selector import decode_selector
from runtime.rr5_arith_basis import restore_carrier_body as restore_rr5_carrier_body

SUPPORTED_RESERVED = ra.RR5_RESERVED_ARITH_BASIS | ra.DX2_RESERVED_CABAC_COEFFICIENTS   # 0x18


def restore_semantic(s: Sections) -> bytes:
    """-> the renderer blob for tcx_render.build_semantic.  Raises ValueError unless its SM3R mode-8 body has the
    sha256 of PR #153's (tcx_format.EXPECTED_SHA256)."""
    semantic = renderer_pack.decode(s.renderer)        # SM3R RID9 body: header + #153's original array stream
    if not semantic.startswith(b"SM3R"):
        raise ValueError("renderer section is not an SM3R tagged renderer")
    if s.renderer_gaps is not None:
        return check_component("renderer body", mode8_body(semantic, s.renderer_gaps))
    from folding.rid9 import decode_to_blob8           # layouts before n3 hand #153's RID9 blob on; check its mode-8 form
    check_component("renderer body", decode_to_blob8(semantic))
    return semantic


def mode8_body(rid9_body: bytes, gaps: bytes) -> bytes:
    """SM3R mode-8 renderer body = gap_0 + packed array_0 + gap_1 + ... + tail gap.  Same steps as #153's
    folding.rid9.decode_to_blob8, except that the gaps are already decoded (it gets them from brotli.decompress(G)).
    tcx_render hands a mode-8 body to folding.rid2.decode_state_dict, exactly as rid9.decode_state_dict does."""
    from folding.rid2 import Dec
    from folding.rid9 import _code_array, pack_codes
    from tcx_ncm import GAP_LENGTHS
    specs, offset = renderer_pack.parse_header(rid9_body)
    if len(specs) + 1 != len(GAP_LENGTHS) or len(gaps) != sum(GAP_LENGTHS):
        raise ValueError("renderer arrays do not match the gap walk")
    arrays = Dec(rid9_body[offset:])
    out, position = bytearray(), 0
    for length, (count, bits, increment) in zip(GAP_LENGTHS, specs):
        out += gaps[position:position + length]
        position += length
        out += pack_codes(_code_array(arrays, count, bits, increment, decode=True), bits)
    return bytes(out + gaps[position:])


def carrier_body(s: Sections) -> bytes:
    """#153's carrier_joint.decode output: stored re-coded by the n2+ layouts, else decoded from the CGA2 section."""
    return s.carrier_body if s.carrier_body is not None else carrier_joint.decode(s.carrier)


def restore_carrier(s: Sections) -> bytes:
    reserved = s.rx1_reserved
    if reserved & ~SUPPORTED_RESERVED:
        raise ValueError(f"unsupported RX1 reserved flags 0x{reserved:02x}")
    body = check_component("carrier body", carrier_body(s))
    if reserved & ra.RR5_RESERVED_ARITH_BASIS:
        body = restore_rr5_carrier_body(body)
    if reserved & ra.DX2_RESERVED_CABAC_COEFFICIENTS:
        body = restore_dx2_carrier_body(body)
    packed_portion = 102 + 40 + ((int.from_bytes(body[0:3], "little") + 7) // 8) \
        + ((int.from_bytes(body[3:6], "little") + 7) // 8)
    if len(body) >= packed_portion + 9:
        body = ra._restore_packed_cap1_metadata(body[:packed_portion]) + body[packed_portion:]
    elif len(body) != ra.CANONICAL_CAP1_SECTION_BYTES:
        raise ValueError("carrier representation differs")
    cap1_bytes = ra._cap1_body_bytes(body)
    if cap1_bytes >= len(body):
        raise ValueError("truncated carrier")
    cap1 = ra._restore_cap1(body[:cap1_bytes])
    selector, compensation = split_selector_compensation(ra.SPARSE_SELECTOR_PREFIX + body[cap1_bytes:])
    if compensation is not None:
        raise ValueError("compensation overlay is not supported (not used by #153)")
    decode_cap1(cap1, frames=600, dimensions=12)
    decode_selector(selector)
    return pack_frame0_selector_carrier(cap1, selector)
