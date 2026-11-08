import unittest
from scripts.build_authored_chinese_candidate import NS

class AuthoredCandidateTests(unittest.TestCase):
 def test_word_namespace_bound(self):self.assertIn("w",NS)
if __name__=="__main__":unittest.main()
