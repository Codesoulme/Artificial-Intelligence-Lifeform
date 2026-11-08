import unittest
from scripts.audit_clean_dialogue_pretrain_candidate import PII, MANIFEST_SHA

class CleanDialogueAuditTests(unittest.TestCase):
    def test_manifest_binding_and_pii_pattern(self):
        self.assertEqual(len(MANIFEST_SHA),64)
        self.assertIsNotNone(PII.search('13800138000'))
if __name__=='__main__':unittest.main()
