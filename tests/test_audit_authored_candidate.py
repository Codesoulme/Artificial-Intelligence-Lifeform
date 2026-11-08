import unittest
from scripts.audit_authored_candidate import ROOT, TOK
class AuthoredAuditTests(unittest.TestCase):
 def test_tokenizer_exists(self):self.assertTrue(TOK.exists())
if __name__=="__main__":unittest.main()
