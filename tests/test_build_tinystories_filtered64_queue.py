import unittest
from scripts.build_tinystories_filtered64_queue import EXPECTED,REV
class TinyQueueTests(unittest.TestCase):
 def test_source_binding(self):
  self.assertEqual(len(EXPECTED),64);self.assertEqual(len(REV),40)
if __name__=='__main__':unittest.main()
