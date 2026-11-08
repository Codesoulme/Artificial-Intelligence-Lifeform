import unittest
from scripts.sample_fineweb2_candidate import INPUT_SHA
class CandidateSampleTests(unittest.TestCase):
 def test_bound_input(self):self.assertEqual(len(INPUT_SHA),64)
if __name__=="__main__":unittest.main()
