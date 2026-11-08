import unittest
from scripts import audit_candidate_metadata as audit


class CandidateMetadataTests(unittest.TestCase):
    def test_candidates_are_fixed(self):
        self.assertEqual(audit.DATASETS["cci"], "BAAI/CCI3-HQ")
        self.assertEqual(audit.DATASETS["skypile"], "Skywork/SkyPile-150B")


if __name__ == "__main__": unittest.main()
