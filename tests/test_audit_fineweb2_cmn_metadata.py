import unittest

from scripts import audit_fineweb2_cmn_metadata as audit


class FinewebMetadataTests(unittest.TestCase):
    def test_shard_metadata_is_bound_to_revision(self):
        self.assertEqual(len(audit.REVISION), 40)
        self.assertIn("fineweb-2", audit.BASE_API)
        self.assertTrue(audit.BASE_FILE.endswith("/"))


if __name__ == "__main__":
    unittest.main()
