import unittest
from scripts.deepseek_memory_need_relative_score_20261117 import decisions

class RelativeScoreTests(unittest.TestCase):
 def test_accepts_correct_and_rejects_bad_only(self):
  rows=[{'case':0} for _ in range(5)]
  r=decisions(rows,[3,0,2,2,2],2)[0]
  self.assertEqual(r['winner'],'right')
  self.assertTrue(r['accepted_with_correct'])
  self.assertFalse(r['accepted_without_correct'])
 def test_threshold_tie_rejects(self):
  rows=[{'case':0} for _ in range(5)]
  r=decisions(rows,[3,1,2,2,2],2)[0]
  self.assertFalse(r['accepted_with_correct'])
  self.assertFalse(r['accepted_without_correct'])
