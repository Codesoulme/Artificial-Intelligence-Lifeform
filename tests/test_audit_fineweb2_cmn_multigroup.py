import unittest
from scripts import audit_fineweb2_cmn_multigroup as audit


class FinewebMultiTests(unittest.TestCase):
    def test_fixed_groups_and_source(self):
        self.assertEqual(audit.ROW_GROUPS, (0, 400, 800, 1200, 1600))
        self.assertEqual(len(audit.REVISION), 40)


if __name__ == "__main__": unittest.main()
