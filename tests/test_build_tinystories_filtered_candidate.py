import unittest
from scripts.build_tinystories_filtered_candidate import QUEUE_SHA,SOURCE_SHA
class TinyCandidateTests(unittest.TestCase):
 def test_bindings(self):self.assertEqual(len(QUEUE_SHA),64);self.assertEqual(len(SOURCE_SHA),64)
if __name__=='__main__':unittest.main()
