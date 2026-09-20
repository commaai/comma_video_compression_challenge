"""Container checks and exact inherited-model parity; no GPU or video rendering."""
from __future__ import annotations

import io
import lzma
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from compress import FILTERS, canonical_zip, repack, single_payload, split_payload
from inflate import verify_input
from runtime.residual_archive import read_residual_archive
from runtime import rowpack


class ContainerGuardTests(unittest.TestCase):
    def parse_payload(self, payload):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "archive.zip"
            path.write_bytes(canonical_zip(payload))
            return read_residual_archive(path)

    def test_raw_lzma_expansion_is_bounded(self):
        oversized = b"F24S" + bytes(1 << 20)
        payload = lzma.compress(oversized, format=lzma.FORMAT_RAW, filters=FILTERS) + b"suffix"
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.parse_payload(payload)

    def test_zip_member_expansion_is_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "archive.zip"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("p", bytes((1 << 20) + 1))
            with self.assertRaisesRegex(ValueError, "size limit"):
                read_residual_archive(path)

    def test_extra_members_are_rejected(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("p", b"payload")
            archive.writestr("unaccounted-model", b"extra")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            single_payload(data.getvalue())

    def test_missing_suffix_is_rejected(self):
        payload = lzma.compress(b"F24S", format=lzma.FORMAT_RAW, filters=FILTERS)
        with self.assertRaisesRegex(ValueError, "missing"):
            split_payload(payload)


class ReferenceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        default = ROOT.parent / "references/pr135/archive.zip"
        cls.reference = Path(os.environ.get("HPAC_REFERENCE_ZIP", default))
        if not cls.reference.is_file():
            raise unittest.SkipTest("set HPAC_REFERENCE_ZIP to the pinned upstream ZIP for parity tests")
        cls.original = cls.reference.read_bytes()
        cls.repacked, cls.report = repack(cls.original)

    def test_repacking_is_deterministic_and_smaller(self):
        repeated, repeated_report = repack(self.original)
        self.assertEqual(self.repacked, repeated)
        self.assertEqual(self.report, repeated_report)
        self.assertEqual(len(self.repacked), 185_767)
        self.assertEqual(self.report["saved_bytes"], 957)

    def test_pinned_reference_rejects_changed_bytes(self):
        changed = bytearray(self.original)
        changed[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "hash-pinned"):
            repack(bytes(changed))

    def test_decoded_assets_equal_upstream_before_any_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "archive.zip"
            candidate.write_bytes(self.repacked)
            original_parts = read_residual_archive(self.reference)
            new_parts = read_residual_archive(candidate)
        for name in ("semantic_blob", "carrier_blob", "hpac_blob", "token_stream", "residual_payload", "schema", "token_codec"):
            self.assertEqual(getattr(original_parts, name), getattr(new_parts, name), name)
        self.assertEqual(original_parts.table.scale, new_parts.table.scale)
        self.assertTrue((original_parts.table.codes == new_parts.table.codes).all())
        self.assertTrue((original_parts.table.values == new_parts.table.values).all())

    def test_extracted_payload_must_equal_counted_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            candidate = directory / "archive.zip"
            candidate.write_bytes(self.repacked)
            payload_path = directory / "p"
            payload_path.write_bytes(single_payload(self.repacked))
            verify_input(directory, candidate)
            payload_path.write_bytes(payload_path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "accounted"):
                verify_input(directory, candidate)

    def test_corrupted_row_stream_is_rejected_by_production_parser(self):
        models, suffix = split_payload(single_payload(self.repacked))
        self.assertTrue(models.startswith(rowpack.MAGIC))
        changed = bytearray(models)
        changed[-1] ^= 1
        payload = lzma.compress(bytes(changed), format=lzma.FORMAT_RAW, filters=FILTERS) + suffix
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "archive.zip"
            candidate.write_bytes(canonical_zip(payload))
            with self.assertRaises(ValueError):
                read_residual_archive(candidate)


if __name__ == "__main__":
    unittest.main()
