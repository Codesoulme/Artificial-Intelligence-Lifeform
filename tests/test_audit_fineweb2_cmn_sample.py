import unittest

from scripts import audit_fineweb2_cmn_sample as audit


class FinewebSampleTests(unittest.TestCase):
    def test_source_binding_and_range_reader(self):
        self.assertEqual(len(audit.REVISION), 40)
        self.assertTrue(audit.PATH.endswith(".parquet"))
        self.assertEqual(audit.SIZE, 4_841_624_662)


if __name__ == "__main__": unittest.main()
