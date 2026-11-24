"""Archive format of tcm_ctxmix (shared by the decoder `inflate.py` and the builder `work/sub/build_archive.py`).

archive.zip holds one stored member; its name is the layout version tag (MEMBER below).  Layout "v1" (member `p`):

    offset  size  field
    0       2     u16 renderer_len        length of the renderer section (PR #153 RDQ2 renderer_pack format)
    2       2     u16 carrier_len         length of the carrier section  (PR #153 CGA2 carrier_joint format)
    4       1     u8  rx1_reserved        PR #151 RX1M reserved flags for the carrier (0x18 = RR5 basis + DX2 coefficients)
    5       R     renderer section        odd-frame SemanticTokenRenderer weights           (verbatim from #153)
    5+R     C     carrier section         even-frame carrier basis/coefficients/selector    (verbatim from #153)
    ..      4     EXT1 header             [family tag=1 grid4][bits][step*8][count]        (verbatim from #153)
    ..      2     u16 ext1_descriptor     #153 tail_pack descriptor; only limit (bits 0-7) and gridmode (bits 13-14) are used
    ..      2     u16 ext1_len            length of the EXT1 range-coded block
    ..      E     EXT1 block              EXT1 pair ids + grid values, coded ALONE with #153's tail_pack coder (ext1_transform)
    ..      rest  TCM stream              the 600 token maps, TCM context-mixing coder (tcm/tcm.c, `tcm d`); starts with
                                          "T" + format version byte, checked by tcm.c itself

Layout "compact" (member `p`, like v1: it predates the version tags) saves 50 bytes of
framing that the decoder can restore or derive: the 'RDQ2'/'CGA2' magics, the reserved flag (0x18), the EXT1 family tag
and bit width (grid4, 4) and coder descriptor (limit 16, gridmode 0) become constants; the EXT1 block length is the
byte count its range decoder consumes; the 1,925-B SM3R renderer header is brotli-compressed (1,890 B):

    2     u16 len(renderer')          renderer' = u16 len(brotli(hdr)) | brotli(SM3R header) | #153 RDQ2 section after its header
    2     u16 len(carrier')
    R'    renderer'
    C'    #153 CGA2 carrier section without its 4-byte magic
    2     EXT1 [step*8][count]
    E     EXT1 block
    rest  TCM stream

Layout "n1" (zip member `q`) = compact with the SM3R renderer header re-coded.  The header is a 127-B array
table followed by G = brotli(2,342 "gap" bytes: fp16 scales, sub-headers, pruned-FiLM masks/codes).  n1 stores the gaps
with the integer context-mixing coder of tcx_ncm.py instead (1,798 -> 1,531 B) and rebuilds G with brotli quality 11:

    renderer' = u16 len(brotli(table)) | brotli(table) | u16 len(gaps stream) | gaps stream | #153 RDQ2 section after its header

Layout "n2" (zip member `r`) = n1 with the RR5 carrier basis re-coded.  #153's CGA2 section codes the 12x3x24x32
five-bit basis values with adaptive counts (5 B weight ranges + 17 B weights + 14 B rate descriptors + 11,292 B).  n2
keeps #153's inter-plane prediction and its weights and codes the residuals with tcx_ncm.py (11,143 B); the rate
descriptors and the prior_bits byte belonged to the old coder and are gone:

    carrier' = u16 len(basis stream) | u8 fraction | 142-B carrier prefix | basis stream | DX2 coefficients + frame-0 selector
    basis stream = 3 x i8 weight minimum | u16 weight bit widths | packed weights | tcx_ncm residual stream

The decoder rebuilds the carrier body that #153's carrier_joint.decode returns (Sections.carrier_body).

Layout "n3" (zip member `s`) = n2 without the stored RID9 array table.  The 127-B table lists, for each of the
22 arithmetic-coded renderer arrays, the gap length in front of it, its value count, its bit width and #153's fixed
RID9 increment (3).  All of that follows from the renderer format: the gap lengths are tcx_ncm.GAP_LENGTHS, the counts
are tensor sizes of the fixed architecture (RID9_COUNTS) and the bit widths are read from the decoded gaps themselves
(RID2 header fields and the sub-blob depth nibbles), so rid9_table() rebuilds it:

    renderer' = u16 len(gaps stream) | gaps stream | #153 RDQ2 section after its header

From n3 on the decoder never calls brotli: unpack returns the renderer section with an SM3R header whose G is empty
(array table only) plus the decoded gaps (Sections.renderer_gaps), and tcx_models.restore_semantic assembles #153's
mode-8 renderer body from the gaps and the decoded arrays.  Only the builder rebuilds G (renderer_section_153), to
prove that the stored bytes reproduce #153's renderer section exactly.

Layout "n4" (zip member `t`) = n3 with the 15-B RID2 header and the 17-B sub-blob header inside the gaps as
format constants (tcx_ncm.GAP_CONSTANTS): the gaps coder skips them and the decoder emits them.

Layout "n5" (zip member `u`) = n4 with the 142-B carrier prefix (bit counts, fp32 scales, packed coefficient
metadata) coded by tcx_ncm in front of the basis residuals, in the same stream:

    carrier' = u16 len(carrier stream) | u8 fraction | carrier stream | DX2 coefficients + frame-0 selector
    carrier stream = 3 x i8 weight minimum | u16 weight bit widths | packed weights | tcx_ncm stream (prefix, residuals)

Layout "n6" (current; zip member `v`) = n5 with the 600 x 12 carrier coefficient symbols (the DX2 CABAC payload,
5,312 B, stored verbatim until now) re-coded at the end of the same tcx_ncm carrier stream.  The decoder re-encodes
the symbols with the DX2 rider's own cabac_encode, which gives the identical DX2 payload.  (The DX2 rider,
vendor/external/pr141/runtime/dx2_cabac_coefficients.py, first appears in PR #140 and is vendored unchanged through
#141, #150, #151, #152 and #153.)

    carrier' = u16 len(carrier stream) | u8 fraction | carrier stream | frame-0 selector
    carrier stream = weight minimum, widths, packed weights | tcx_ncm stream (prefix, basis residuals, coefficients)

The zip member name is the layout version tag (v1/compact: `p`, n1: `q`, n2: `r`, n3: `s`, n4: `t`, n5: `u`,
n6: `v`): a decoder only opens its own member, so an archive of another layout version is rejected ("member ... is
missing") instead of being misread.  inflate.py reads LAYOUT (n6); the earlier layouts are kept for reference and
for the tests, which check that each rejects the others' payloads.

Integrity: the re-coded streams carry no redundancy of their own, so after decoding, the three non-token components
are compared with the sha256 digests of PR #153's own data (EXPECTED_SHA256, check_component): the renderer body, the
carrier body and the EXT1 block.  A mismatch raises ValueError.  This costs no archive bytes; #153 checks its restored
payload against recorded digests in the same way (its verify.py / provenance.json).

Everything token-only in #153 (HPAC weights, RC64 stream, prev2 record, residual table, warp ids, their headers)
is gone: the TCM stream decodes the same 600 x 384 x 512 token maps losslessly.
"""
from __future__ import annotations

import hashlib
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE / "pr153") not in sys.path:
    sys.path.insert(0, str(HERE / "pr153"))

from hpac_range import Decoder, Encoder, Model, code_int  # noqa: E402  (#153's generic binary range coder)
import tcx_ncm  # noqa: E402  (integer context-mixing coder for the renderer gaps and the carrier stream)

LAYOUT = "n6"            # layout read by inflate.py (current); the earlier layouts are kept for reference
MEMBER = {"v1": "p", "compact": "p", "n1": "q", "n2": "r", "n3": "s", "n4": "t", "n5": "u", "n6": "v"}

# sha256 of the decoded non-token components of PR #153's archive (layout independent).  The builder refuses to build
# unless they equal what #153's own decoder produces (work/sub/build_archive.py).
EXPECTED_SHA256 = {
    # SM3R mode-8 renderer body = #153's folding.rid9.decode_to_blob8 of its renderer blob (29,738 B)
    "renderer body": "9e5491a0b89949255e6ef05d1b8c4672aee258c17c0b1258c4e868b45aa82772",
    # carrier body = #153's carrier_joint.decode of its CGA2 section (17,509 B)
    "carrier body": "d02ed53fc974ab867b98cd87dc2cff144ec393bae1c32cabe871ebb99fe64f9d",
    # 4-byte EXT1 header (tag, bits, step*8, count) + the EXT1 block re-encoded alone (134 B)
    "EXT1 block": "b283359fd398ce1230f02c7922eb0058bd5b763bfd7439912d9120d72f442d80",
}


def check_component(name: str, data: bytes) -> bytes:
    """Raise ValueError unless `data` is PR #153's component `name`; returns data."""
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256[name]:
        raise ValueError(f"decoded {name} is not PR #153's (sha256 {digest}, expected {EXPECTED_SHA256[name]}): "
                         f"the archive is damaged or was not built from #153's data")
    return data


HEAD = struct.Struct("<HHB")
EXT1_TAIL = struct.Struct("<4sHH")
GRID4_TAG = 1
GRID4_VALUES = 16


@dataclass
class Sections:
    renderer: bytes
    carrier: bytes | None   # #153 CGA2 section; None when the layout stores the decoded body instead (carrier_body)
    rx1_reserved: int
    ext1_header: bytes      # 4 bytes: tag, bits, step*8, count
    ext1_descriptor: int
    ext1_block: bytes
    tcm: bytes
    carrier_body: bytes | None = None   # what #153's carrier_joint.decode(carrier) returns (layouts n2+)
    renderer_gaps: bytes | None = None  # layouts n3+: the SM3R header of `renderer` has an empty G; the gaps are here


def pack(s: Sections, layout: str = LAYOUT) -> bytes:
    """Sections with #153's renderer (RDQ2) and carrier (CGA2) sections -> payload of the given layout."""
    if layout in N3_FAMILY:
        return pack_n3(s, layout)
    return {"v1": pack_v1, "compact": pack_compact, "n1": pack_n1, "n2": pack_n2}[layout](s)


def unpack(p: bytes, layout: str = LAYOUT) -> Sections:
    """Payload -> Sections.  Any inconsistency raises ValueError (inflate.py then exits non-zero)."""
    if layout in N3_FAMILY:
        return unpack_n3(p, layout)
    return {"v1": unpack_v1, "compact": unpack_compact, "n1": unpack_n1, "n2": unpack_n2}[layout](p)


def tcm_stream(p: bytes, layout: str = LAYOUT) -> bytes:
    """Only the TCM token stream of the payload.  Parses the framing alone (section lengths and the EXT1 block, a few
    milliseconds), so inflate.py can start the token decoder before it decodes the non-token sections with unpack()."""
    if layout == "v1":
        return unpack_v1(p).tcm
    return _unpack_frame(p, layout)[4]


def read_payload(archive_dir, layout: str = LAYOUT) -> bytes:
    """The payload of the unzipped archive.  Each layout version has its own member name, so an archive built for
    another layout version is rejected here (ValueError) instead of being parsed with the wrong rules."""
    path = Path(archive_dir) / MEMBER[layout]
    if not path.is_file():
        found = sorted(f.name for f in Path(archive_dir).iterdir()) if Path(archive_dir).is_dir() else []
        raise ValueError(f"archive member '{MEMBER[layout]}' of layout '{layout}' is missing (found: {found}); "
                         f"this archive was built for a different tcm_ctxmix layout version")
    return path.read_bytes()


def pack_v1(s: Sections) -> bytes:
    if len(s.ext1_header) != 4 or max(len(s.renderer), len(s.carrier), len(s.ext1_block)) > 0xFFFF:
        raise ValueError("section does not fit the container")
    return (HEAD.pack(len(s.renderer), len(s.carrier), s.rx1_reserved) + s.renderer + s.carrier
            + EXT1_TAIL.pack(s.ext1_header, s.ext1_descriptor, len(s.ext1_block)) + s.ext1_block + s.tcm)


def unpack_v1(p: bytes) -> Sections:
    nr, nc, reserved = HEAD.unpack_from(p)
    o = HEAD.size
    renderer = p[o:o + nr]; o += nr
    carrier = p[o:o + nc]; o += nc
    ext1_header, descriptor, ne = EXT1_TAIL.unpack_from(p, o); o += EXT1_TAIL.size
    block = p[o:o + ne]; o += ne
    tcm = p[o:]
    if len(renderer) != nr or len(carrier) != nc or len(block) != ne or len(tcm) < 45 or tcm[:1] != b"T":
        raise ValueError("truncated or malformed tcm_ctxmix payload")
    return Sections(renderer, carrier, reserved, ext1_header, descriptor, block, tcm)


# ---- compact layout -------------------------------------------------------------------------------------------
RDQ2, CGA2 = b"RDQ2", b"CGA2"
COMPACT_RESERVED = 0x18                  # #151 RX1M reserved flags: RR5 arithmetic basis + DX2 CABAC coefficients
COMPACT_DESCRIPTOR = 16                  # #153 tail_pack descriptor (16, 0, 0, 0): limit 16, gridmode 0
COMPACT_EXT1 = (GRID4_TAG, 4)            # EXT1 family grid4, 4-bit grid values


def pack_compact(s: Sections) -> bytes:
    import brotli
    nh = _check_compact_constants(s)
    hdr = brotli.compress(s.renderer[6:6 + nh], quality=11, lgwin=10)
    return _pack_frame(struct.pack("<H", len(hdr)) + hdr + s.renderer[6 + nh:], s.carrier[4:], s)


def _check_compact_constants(s: Sections) -> int:
    """The constants the compact-family layouts imply; -> length of the SM3R renderer header."""
    if s.rx1_reserved != COMPACT_RESERVED or descriptor_settings(s.ext1_descriptor) != descriptor_settings(COMPACT_DESCRIPTOR) \
            or tuple(s.ext1_header[:2]) != COMPACT_EXT1 or s.renderer[:4] != RDQ2 or s.carrier[:4] != CGA2:
        raise ValueError("sections do not match the compact layout constants")
    return struct.unpack_from("<H", s.renderer, 4)[0]


def _pack_frame(renderer: bytes, carrier: bytes, s: Sections) -> bytes:
    """Framing shared by the compact-family layouts: u16 len(renderer') | u16 len(carrier') | both | EXT1 | TCM."""
    if max(len(renderer), len(carrier)) > 0xFFFF:
        raise ValueError("section does not fit the container")
    return (struct.pack("<HH", len(renderer), len(carrier)) + renderer + carrier
            + s.ext1_header[2:4] + s.ext1_block + s.tcm)


def _bad(layout, msg):
    raise ValueError(f"malformed tcm_ctxmix payload ({layout} layout): {msg}")


def _unpack_frame(p: bytes, layout: str):
    """Inverse of _pack_frame -> (renderer', carrier', ext1_header, ext1_block, tcm); raises ValueError when inconsistent."""
    if len(p) < 8:
        _bad(layout, "truncated header")
    nr, nc = struct.unpack_from("<HH", p)
    o = 4
    if o + nr + nc + 2 > len(p) or nr < 2 or nc < 4 + 142:
        _bad(layout, f"section lengths {nr}/{nc} do not fit the {len(p)}-byte payload")
    renderer = p[o:o + nr]; o += nr
    carrier = p[o:o + nc]; o += nc
    ext1_header = bytes(COMPACT_EXT1) + p[o:o + 2]; o += 2
    if not 1 <= ext1_header[3] <= 600 or ext1_header[2] == 0:
        _bad(layout, "invalid EXT1 step/count")
    d = Decoder(p[o:])
    ids, _ = ext1_transform(d, COMPACT_DESCRIPTOR & 255, 0, count=ext1_header[3], bits=ext1_header[1])
    if d.i > len(p) - o or ids != sorted(set(ids)) or not 0 <= ids[0] <= ids[-1] < 600:
        _bad(layout, "EXT1 block is inconsistent")
    block = p[o:o + d.i]; o += d.i      # hpac_range: the bytes its decoder consumes == the encoded length
    tcm = p[o:]
    if len(tcm) < 45 or tcm[:1] != b"T":
        _bad(layout, "TCM stream is missing or does not start with 'T'")
    return renderer, carrier, ext1_header, block, tcm


def unpack_compact(p: bytes) -> Sections:
    """Parse the compact layout, restoring the constant bytes.  Any inconsistency raises ValueError, which makes
    inflate.py exit non-zero with that message (the restored blobs are further checked by their own decoders)."""
    import brotli
    renderer, carrier, ext1_header, block, tcm = _unpack_frame(p, "compact")
    nb = struct.unpack_from("<H", renderer)[0]
    if 2 + nb >= len(renderer):
        _bad("compact", "renderer header length exceeds the renderer section")
    try:
        hdr = brotli.decompress(renderer[2:2 + nb])
    except brotli.error as error:
        _bad("compact", f"renderer header does not decompress ({error})")
    if not hdr.startswith(b"SM3R") or len(hdr) > 0xFFFF:
        _bad("compact", "renderer header is not an SM3R header")
    renderer = RDQ2 + struct.pack("<H", len(hdr)) + hdr + renderer[2 + nb:]
    carrier = _restore_cga2(carrier, "compact")
    return Sections(renderer, carrier, COMPACT_RESERVED, ext1_header, COMPACT_DESCRIPTOR, block, tcm)


def _restore_cga2(carrier: bytes, layout: str) -> bytes:
    carrier = CGA2 + carrier
    if 8 + 142 + struct.unpack_from("<H", carrier, 4)[0] > len(carrier):
        _bad(layout, "carrier size field exceeds the carrier section")
    return carrier


# ---- n1 layout: compact + context-mixed renderer gaps -----------------------------------------------------------
RID9_TABLE_BYTES = 127                   # b"SM3R" 1 9 | n=22 | 22 x (gap, count, bits, inc) | tail gap | u32 len(G)


def split_sm3r_header(hdr: bytes):
    """#153's SM3R RID9 renderer header -> (table, gaps): table = the bytes before G (ending with u32 len(G)),
    gaps = brotli.decompress(G).  Checks that the table's gap lengths are those of tcx_ncm.GAP_WALK."""
    import brotli
    from renderer_pack import read_number
    if hdr[:6] != b"SM3R\x01\x09" or hdr[6] != len(tcx_ncm.GAP_LENGTHS) - 1:
        raise ValueError("not the expected SM3R RID9 header")
    o, lengths = 7, []
    for _ in range(hdr[6]):
        gap, o = read_number(hdr, o)
        _, o = read_number(hdr, o)
        o += 2
        lengths.append(gap)
    tail, o = read_number(hdr, o)
    size = struct.unpack_from("<I", hdr, o)[0]; o += 4
    if lengths + [tail] != tcx_ncm.GAP_LENGTHS or o + size != len(hdr):
        raise ValueError("SM3R RID9 table does not match the renderer gap walk")
    return hdr[:o], brotli.decompress(hdr[o:])


def join_sm3r_header(table: bytes, gaps: bytes) -> bytes:
    """Inverse of split_sm3r_header.  #153 built G with brotli quality 11; brotli 1.2.0 (pinned in uv.lock) reproduces
    its bytes exactly.  Any other valid brotli stream of the same gaps decodes to the same renderer, so the length
    field is written from the stream actually produced."""
    import brotli
    g = brotli.compress(gaps, quality=11)
    return table[:-4] + struct.pack("<I", len(g)) + g


def _pack_renderer_n1(s: Sections, nh: int) -> bytes:
    import brotli
    table, gaps = split_sm3r_header(s.renderer[6:6 + nh])
    if join_sm3r_header(table, gaps) != s.renderer[6:6 + nh]:
        raise ValueError("this brotli build does not reproduce #153's SM3R header bytes")
    table_z = brotli.compress(table, quality=11, lgwin=10)
    stream = tcx_ncm.encode_gaps(gaps)
    if tcx_ncm.decode_gaps(stream) != gaps:
        raise ValueError("renderer gaps do not round trip")
    return struct.pack("<H", len(table_z)) + table_z + struct.pack("<H", len(stream)) + stream + s.renderer[6 + nh:]


def _unpack_renderer_n1(renderer: bytes, layout: str) -> bytes:
    import brotli
    nt = struct.unpack_from("<H", renderer)[0]
    if 2 + nt + 2 > len(renderer):
        _bad(layout, "renderer table length exceeds the renderer section")
    try:
        table = brotli.decompress(renderer[2:2 + nt])
    except brotli.error as error:
        _bad(layout, f"renderer table does not decompress ({error})")
    o = 2 + nt
    ng = struct.unpack_from("<H", renderer, o)[0]; o += 2
    if o + ng >= len(renderer):
        _bad(layout, "renderer gaps stream length exceeds the renderer section")
    try:
        gaps = tcx_ncm.decode_gaps(renderer[o:o + ng])
        hdr = join_sm3r_header(table, gaps)
        if len(table) != RID9_TABLE_BYTES or split_sm3r_header(hdr) != (hdr[:RID9_TABLE_BYTES], gaps):
            raise ValueError("renderer table is not the expected 127-byte RID9 table")
    except (ValueError, IndexError, struct.error) as error:
        _bad(layout, f"renderer header: {error}")
    return RDQ2 + struct.pack("<H", len(hdr)) + hdr + renderer[o + ng:]


def pack_n1(s: Sections) -> bytes:
    nh = _check_compact_constants(s)
    return _pack_frame(_pack_renderer_n1(s, nh), s.carrier[4:], s)


def unpack_n1(p: bytes) -> Sections:
    """Parse the n1 layout.  Any inconsistency raises ValueError (inflate.py then exits non-zero)."""
    renderer, carrier, ext1_header, block, tcm = _unpack_frame(p, "n1")
    renderer = _unpack_renderer_n1(renderer, "n1")
    return Sections(renderer, _restore_cga2(carrier, "n1"), COMPACT_RESERVED, ext1_header, COMPACT_DESCRIPTOR, block, tcm)


# ---- n2 layout: n1 + context-mixed RR5 carrier basis -----------------------------------------------------------
CARRIER_PREFIX = 142                     # #153 carrier_pack.PREFIX: bit counts, fp32 scales, packed CAP1 metadata
BASIS_BITS = 5


def _basis_weights(data: bytes, o: int):
    """#153 carrier_joint weight block at data[o:]: 3 x i8 minimum, u16 of three 4-bit widths, packed weights
    -> (36 weights, offset behind the block)."""
    import carrier_joint
    *minimum, mask = struct.unpack_from("<3bH", data, o)
    widths = [(mask >> (4 * c)) & 15 for c in range(3)]
    if mask >> 12 or max(widths) > 8:
        raise ValueError("invalid basis weight ranges")
    nw = (tcx_ncm.ATOMS * sum(widths) + 7) // 8
    if o + 5 + nw > len(data):
        raise ValueError("truncated basis weights")
    packed = carrier_joint.unpack_fields(data[o + 5:o + 5 + nw], widths * tcx_ncm.ATOMS)
    return [n + minimum[i % 3] for i, n in enumerate(packed)], o + 5 + nw


def _pack_carrier_n2(cga2: bytes) -> bytes:
    import carrier_joint
    import carrier_pack
    magic, size, fraction, prior_bits = carrier_joint.HEADER.unpack_from(cga2)
    start = carrier_joint.HEADER.size + CARRIER_PREFIX
    body = carrier_joint.decode(cga2)
    atoms, basis_size = carrier_pack.read_original(body, tcx_ncm.ATOMS, tcx_ncm.PLANES, tcx_ncm.ATOM_H, tcx_ncm.ATOM_W, BASIS_BITS)
    weights, end = _basis_weights(cga2, start)
    stream = cga2[start:end] + tcx_ncm.encode_basis(atoms, weights, fraction)
    carrier = struct.pack("<HB", len(stream), fraction) + cga2[8:start] + stream + cga2[start + size:]
    if _unpack_carrier_n2(carrier) != body or body[CARRIER_PREFIX + basis_size:] != cga2[start + size:]:
        raise ValueError("carrier basis does not round trip")
    return carrier


def _unpack_carrier_n2(carrier: bytes) -> bytes:
    """-> the carrier body, byte-identical to #153's carrier_joint.decode of its CGA2 section."""
    import carrier_pack
    size, fraction = struct.unpack_from("<HB", carrier)
    start = 3 + CARRIER_PREFIX
    if not 1 <= fraction <= 7 or size < 6 or start + size >= len(carrier):
        raise ValueError("invalid carrier header")
    prefix = carrier[3:start]
    weights, o = _basis_weights(carrier[:start + size], start)
    atoms = tcx_ncm.decode_basis(carrier[o:start + size], weights, fraction)
    basis = carrier_pack.write_original(atoms, BASIS_BITS)
    if len(basis) != (int.from_bytes(prefix[:3], "little") + 7) // 8:
        raise ValueError("restored carrier basis length mismatch")
    return prefix + basis + carrier[start + size:]


def pack_n2(s: Sections) -> bytes:
    nh = _check_compact_constants(s)
    return _pack_frame(_pack_renderer_n1(s, nh), _pack_carrier_n2(s.carrier), s)


def unpack_n2(p: bytes) -> Sections:
    """Parse the n2 layout.  Any inconsistency raises ValueError (inflate.py then exits non-zero)."""
    renderer, carrier, ext1_header, block, tcm = _unpack_frame(p, "n2")
    renderer = _unpack_renderer_n1(renderer, "n2")
    try:
        body = _unpack_carrier_n2(carrier)
    except (ValueError, IndexError, struct.error) as error:
        _bad("n2", f"carrier: {error}")
    return Sections(renderer, None, COMPACT_RESERVED, ext1_header, COMPACT_DESCRIPTOR, block, tcm, carrier_body=body)


# ---- n3 layout and its successors n4, n5, n6 (N3_FAMILY below) -------------------------------------------------
# n3: n2 with the RID9 array table derived from the format
# Value counts of the 22 arithmetic-coded arrays of #153's renderer body, in body order (fixed architecture: width 96,
# 5 token classes, 4 coordinate channels, 600 frames x 8 embedding dims, 3x3 depthwise and 1x1 pointwise convolutions):
RID9_COUNTS = ([96 * 5, 96 * 4]                                   # T codes, Wc codes
               + [96, 96, 192, 96, 192, 96, 192, 96, 192]         # the nine quantised 1-D vectors
               + [600 * 8]                                        # frame_embed
               + [96 * 9, 96 * 96, 192 * 8]                       # block 0: dw, pw, film
               + [96 * 9, 96 * 96] * 3                            # blocks 1-3: dw, pw (their FiLM is row-pruned, in the gaps)
               + [3 * 96 * 9])                                    # head
RID9_INCREMENT = 3                                                # #153's RID9 count increment, the same for every array
# Position of each sub-blob array (frame_embed .. head) in the sub-blob's list of depth nibbles; the list also holds
# the three pruned FiLM tensors (positions 6, 9, 12), which are not RID9 arrays.
SUB_BLOB_DEPTH_INDEX = [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13]


def rid9_array_bits(gaps: bytes):
    """Bit width of each RID9 array, read from the header fields inside the gaps (vendor/folding/rid2.py)."""
    sub = sum(tcx_ncm.GAP_LENGTHS[:11])                 # the sub-blob header opens gap 11
    if gaps[:6] != b"SM3R\x01\x08" or gaps[sub:sub + 6] != b"SM3R\x01\x06":
        raise ValueError("renderer gaps do not start with the RID2 / sub-blob headers")
    t_bits, wc_bits, v_bits = gaps[6], gaps[7], gaps[8]
    nibbles = [(gaps[sub + 10 + i // 2] >> (4 * (i % 2))) & 15 for i in range(14)]
    return [t_bits, wc_bits] + [v_bits] * 9 + [nibbles[i] for i in SUB_BLOB_DEPTH_INDEX]


def _varint(n: int) -> bytes:
    out = bytearray()
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def rid9_table(gaps: bytes) -> bytes:
    """The RID9 array table of #153's SM3R header (vendor/folding/rid9.py), ending with a zero u32 for len(G)
    that join_sm3r_header fills in."""
    bits = rid9_array_bits(gaps)
    if len(bits) != len(RID9_COUNTS) or not all(1 <= b <= 15 for b in bits):
        raise ValueError("invalid renderer array bit widths")
    table = bytearray(b"SM3R\x01\x09" + bytes([len(RID9_COUNTS)]))
    for gap, count, width in zip(tcx_ncm.GAP_LENGTHS, RID9_COUNTS, bits):
        table += _varint(gap) + _varint(count) + bytes([width, RID9_INCREMENT])
    return bytes(table + _varint(tcx_ncm.GAP_LENGTHS[-1]) + bytes(4))


def _pack_renderer_n3(s: Sections, nh: int, constant_headers: bool = False) -> bytes:
    table, gaps = split_sm3r_header(s.renderer[6:6 + nh])
    if join_sm3r_header(rid9_table(gaps), gaps) != s.renderer[6:6 + nh]:
        raise ValueError("the derived RID9 table or this brotli build does not reproduce #153's SM3R header bytes")
    stream = tcx_ncm.encode_gaps(gaps, constant_headers)
    if tcx_ncm.decode_gaps(stream, constant_headers) != gaps:
        raise ValueError("renderer gaps do not round trip")
    return struct.pack("<H", len(stream)) + stream + s.renderer[6 + nh:]


def _unpack_renderer_n3(renderer: bytes, layout: str, constant_headers: bool = False):
    """-> (RDQ2 renderer section whose SM3R header holds the array table and an empty G, gaps).  No brotli here."""
    ng = struct.unpack_from("<H", renderer)[0]
    if 2 + ng >= len(renderer):
        _bad(layout, "renderer gaps stream length exceeds the renderer section")
    try:
        gaps = tcx_ncm.decode_gaps(renderer[2:2 + ng], constant_headers)
        table = rid9_table(gaps)
    except (ValueError, IndexError, struct.error) as error:
        _bad(layout, f"renderer header: {error}")
    return RDQ2 + struct.pack("<H", len(table)) + table + renderer[2 + ng:], gaps


def renderer_section_153(s: Sections) -> bytes:
    """Builder/test side: #153's exact RDQ2 renderer section (G rebuilt with brotli; decoding never needs it)."""
    if s.renderer_gaps is None:
        return s.renderer
    nt = struct.unpack_from("<H", s.renderer, 4)[0]
    hdr = join_sm3r_header(s.renderer[6:6 + nt], s.renderer_gaps)
    return RDQ2 + struct.pack("<H", len(hdr)) + hdr + s.renderer[6 + nt:]


# ---- n5: carrier prefix inside the tcx_ncm carrier stream ------------------------------------------------------
def _pack_carrier_n5(cga2: bytes) -> bytes:
    import carrier_joint
    import carrier_pack
    _, size, fraction, _ = carrier_joint.HEADER.unpack_from(cga2)
    start = carrier_joint.HEADER.size + CARRIER_PREFIX
    body = carrier_joint.decode(cga2)
    atoms, basis_size = carrier_pack.read_original(body, tcx_ncm.ATOMS, tcx_ncm.PLANES, tcx_ncm.ATOM_H, tcx_ncm.ATOM_W, BASIS_BITS)
    weights, end = _basis_weights(cga2, start)
    coder = tcx_ncm.CarrierCoder()
    coder.prefix(body[:CARRIER_PREFIX])
    coder.basis(weights, fraction, atoms)
    stream = cga2[start:end] + coder.finish()
    carrier = struct.pack("<HB", len(stream), fraction) + stream + cga2[start + size:]
    if _unpack_carrier_n5(carrier) != body or body[CARRIER_PREFIX + basis_size:] != cga2[start + size:]:
        raise ValueError("carrier does not round trip")
    return carrier


def _unpack_carrier_n5(carrier: bytes) -> bytes:
    """-> the carrier body, byte-identical to #153's carrier_joint.decode of its CGA2 section."""
    import carrier_pack
    size, fraction = struct.unpack_from("<HB", carrier)
    if not 1 <= fraction <= 7 or size < 6 or 3 + size >= len(carrier):
        raise ValueError("invalid carrier header")
    weights, o = _basis_weights(carrier[:3 + size], 3)
    coder = tcx_ncm.CarrierCoder(carrier[o:3 + size])
    prefix = coder.prefix()
    atoms = coder.basis(weights, fraction)
    coder.check_end()
    basis = carrier_pack.write_original(atoms, BASIS_BITS)
    if len(basis) != (int.from_bytes(prefix[:3], "little") + 7) // 8:
        raise ValueError("restored carrier basis length mismatch")
    return prefix + basis + carrier[3 + size:]


# ---- n6: carrier coefficient symbols inside the tcx_ncm carrier stream -----------------------------------------
def _byte_count(bit_count_field: bytes) -> int:
    """The carrier prefix stores its two stream sizes as 24-bit little-endian bit counts."""
    return (int.from_bytes(bit_count_field, "little") + 7) // 8


def _coefficient_ks(prefix: bytes):
    """The 12 Rice parameters of the coefficient symbols, from the packed metadata at the end of the carrier prefix."""
    from runtime.dx2_cabac_coefficients import packed_ks
    return [int(k) for k in packed_ks(bytes(prefix[CARRIER_PREFIX - 40:CARRIER_PREFIX]))]


def _pack_carrier_n6(cga2: bytes) -> bytes:
    import numpy as np
    import carrier_joint
    import carrier_pack
    from runtime import dx2_cabac_coefficients as dx2
    _, size, fraction, _ = carrier_joint.HEADER.unpack_from(cga2)
    start = carrier_joint.HEADER.size + CARRIER_PREFIX
    body = carrier_joint.decode(cga2)
    prefix = body[:CARRIER_PREFIX]
    atoms, basis_size = carrier_pack.read_original(body, tcx_ncm.ATOMS, tcx_ncm.PLANES, tcx_ncm.ATOM_H, tcx_ncm.ATOM_W, BASIS_BITS)
    o = CARRIER_PREFIX + basis_size
    payload, selector = body[o:o + _byte_count(prefix[3:6])], body[o + _byte_count(prefix[3:6]):]
    ks = _coefficient_ks(prefix)
    symbols = dx2.cabac_decode(payload, np.asarray(ks, np.int64))
    if dx2.cabac_encode(symbols, np.asarray(ks, np.int64)) != payload or payload + selector != cga2[start + size:]:
        raise ValueError("DX2 coefficient payload is not canonical")
    weights, end = _basis_weights(cga2, start)
    coder = tcx_ncm.CarrierCoder()
    coder.prefix(prefix)
    coder.basis(weights, fraction, atoms)
    coder.coefficients(ks, symbols.tolist())
    stream = cga2[start:end] + coder.finish()
    carrier = struct.pack("<HB", len(stream), fraction) + stream + selector
    if _unpack_carrier_n6(carrier) != body:
        raise ValueError("carrier does not round trip")
    return carrier


def _unpack_carrier_n6(carrier: bytes) -> bytes:
    """-> the carrier body, byte-identical to #153's carrier_joint.decode of its CGA2 section."""
    import numpy as np
    import carrier_pack
    from runtime import dx2_cabac_coefficients as dx2
    size, fraction = struct.unpack_from("<HB", carrier)
    if not 1 <= fraction <= 7 or size < 6 or 3 + size >= len(carrier):
        raise ValueError("invalid carrier header")
    weights, o = _basis_weights(carrier[:3 + size], 3)
    coder = tcx_ncm.CarrierCoder(carrier[o:3 + size])
    prefix = coder.prefix()
    atoms = coder.basis(weights, fraction)
    ks = _coefficient_ks(prefix)
    symbols = coder.coefficients(ks)
    coder.check_end()
    basis = carrier_pack.write_original(atoms, BASIS_BITS)
    coefficients = dx2.cabac_encode(np.asarray(symbols, np.int64), np.asarray(ks, np.int64))     # the DX2 payload
    if len(basis) != _byte_count(prefix[:3]) or len(coefficients) != _byte_count(prefix[3:6]):
        raise ValueError("restored carrier stream lengths differ from the prefix bit counts")
    return prefix + basis + coefficients + carrier[3 + size:]


# layout -> (gap headers are format constants (n4+), carrier packer, carrier unpacker)
N3_FAMILY = {"n3": (False, _pack_carrier_n2, _unpack_carrier_n2), "n4": (True, _pack_carrier_n2, _unpack_carrier_n2),
             "n5": (True, _pack_carrier_n5, _unpack_carrier_n5), "n6": (True, _pack_carrier_n6, _unpack_carrier_n6)}


def pack_n3(s: Sections, layout: str) -> bytes:
    nh = _check_compact_constants(s)
    constant_headers, pack_carrier, _ = N3_FAMILY[layout]
    return _pack_frame(_pack_renderer_n3(s, nh, constant_headers), pack_carrier(s.carrier), s)


def unpack_n3(p: bytes, layout: str) -> Sections:
    """Parse a layout of the n3 family.  Any inconsistency raises ValueError (inflate.py then exits non-zero)."""
    constant_headers, _, unpack_carrier = N3_FAMILY[layout]
    renderer, carrier, ext1_header, block, tcm = _unpack_frame(p, layout)
    renderer, gaps = _unpack_renderer_n3(renderer, layout, constant_headers)
    try:
        body = unpack_carrier(carrier)
    except (ValueError, IndexError, struct.error) as error:
        _bad(layout, f"carrier: {error}")
    return Sections(renderer, None, COMPACT_RESERVED, ext1_header, COMPACT_DESCRIPTOR, block, tcm,
                    carrier_body=body, renderer_gaps=gaps)


def ext1_transform(c, limit, gridmode, ids=None, grids=None, count=0, bits=4):
    """EXT1 part of #153 tail_pack.transform (pair ids, then 16 grid values per pair), same model and contexts.

    #153 coded these after its residual-table and warp-id families in ONE adaptive coder; the context families are
    disjoint, so coding the id/grid families alone gives exactly the same per-symbol probabilities.
    Works with an Encoder (values are coded) or a Decoder (values are returned)."""
    model = Model(limit)
    enc = isinstance(c, Encoder)
    if not enc:
        ids = [0] * count
        grids = [0] * (GRID4_VALUES * count)
    previous = 0
    for i in range(count):
        value = code_int(c, model, ("id",), ids[i] - previous if enc else 0)
        ids[i] = previous + value
        previous = ids[i]
    for i in range(len(grids)):
        local = i % 16
        if gridmode == 0:
            predictor = 0
        elif gridmode == 1:
            predictor = grids[i - 1] if local else 0
        elif gridmode == 2:
            predictor = grids[i - 4] if local >= 4 else 0
        else:
            left = grids[i - 1] if local % 4 else 0
            up = grids[i - 4] if local >= 4 else 0
            predictor = (left + up) // 2
        grids[i] = predictor + code_int(c, model, ("g",), grids[i] - predictor if enc else 0,
                                        bits + 1 if gridmode else bits)
    return ids, grids


def descriptor_settings(descriptor: int):
    """-> (limit, gridmode) from the #153 tail_pack descriptor."""
    return descriptor & 255, (descriptor >> 13) & 3


def encode_ext1(ids, grids, ext1_header: bytes, descriptor: int) -> bytes:
    limit, gridmode = descriptor_settings(descriptor)
    e = Encoder()
    ext1_transform(e, limit, gridmode, list(ids), list(grids), len(ids), ext1_header[1])
    return e.finish()


def decode_ext1(s: Sections) -> dict:
    """-> the EXT1 dict that #153's decoder.extfmt.parse_payload returns: dict(fam, bits, step, q={pair: [16 ints]})."""
    tag, bits, step8, count = s.ext1_header
    if tag != GRID4_TAG:
        raise ValueError("only the grid4 EXT1 family is supported")
    check_component("EXT1 block", bytes(s.ext1_header) + s.ext1_block)
    limit, gridmode = descriptor_settings(s.ext1_descriptor)
    ids, grids = ext1_transform(Decoder(s.ext1_block), limit, gridmode, count=count, bits=bits)
    return dict(fam="grid4", bits=bits, step=step8 / 8.0,
                q={p: grids[GRID4_VALUES * k:GRID4_VALUES * (k + 1)] for k, p in enumerate(ids)})
