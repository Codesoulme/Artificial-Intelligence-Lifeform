import unittest
from scripts.build_fineweb2_domain_clean_candidate import INPUT_SHA

class CandidateTests(unittest.TestCase):
 def test_input_binding(self): self.assertEqual(len(INPUT_SHA),64)

if __name__=="__main__":unittest.main()
