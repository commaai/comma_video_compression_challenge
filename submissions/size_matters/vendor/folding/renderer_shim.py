"""RID1/RID2-aware decoder shim (RP1 shim + wraps runtime.f26_inflate._load_renderer so unpack_variant_semantic_or_none understands SM3R mode 7 (RID1) and mode 8 (RID2); fail closed)."""
import sys
sys.path.insert(0, '.')


def install():
    from hpac import rowpack
    rowpack.install_decoder_shim()
    from decoder import inflate_core  # noqa: F401
    import runtime.f26_inflate as F
    if getattr(F._load_renderer, "_rid2", False): return
    orig = F._load_renderer

    def patched(renderer_dir):
        m = orig(renderer_dir)
        if not getattr(m, "_rid2_patched", False):
            inner = m.unpack_variant_semantic_or_none

            def unpack(blob, template):
                if bytes(blob[:4]) == b"SM3R" and len(blob) > 5 and blob[5] == 9:
                    from folding.rid9 import decode_state_dict
                    return decode_state_dict(blob, template)
                if bytes(blob[:4]) == b"SM3R" and len(blob) > 5 and blob[5] == 8:
                    from folding.rid2 import decode_state_dict
                    return decode_state_dict(blob, template)
                if bytes(blob[:4]) == b"SM3R" and len(blob) > 5 and blob[5] == 7:
                    from folding.rid1 import decode_state_dict
                    return decode_state_dict(blob, template)
                return inner(blob, template)
            m.unpack_variant_semantic_or_none = unpack; m._rid2_patched = True
        return m
    patched._rid2 = True; F._load_renderer = patched
