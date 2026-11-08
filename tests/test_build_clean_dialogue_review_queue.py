import unittest
from scripts.build_clean_dialogue_review_queue import flags,INPUT_SHA
class ReviewQueueTests(unittest.TestCase):
 def test_flags_and_binding(self):
  self.assertEqual(len(INPUT_SHA),64)
  self.assertIn('pii_pattern',flags('甲'*120+' 13800138000'))
if __name__=='__main__':unittest.main()
