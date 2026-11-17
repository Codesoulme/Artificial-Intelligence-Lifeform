import unittest
from scripts.audit_memory_need_score_separation_20261117 import summarize

class ScoreSeparationTests(unittest.TestCase):
 def test_perfect_case_and_global(self):
  rows=[{'case':0,'need':0,'score':.1}]+[{'case':0,'need':1,'score':s} for s in (.2,.3,.4)]
  result=summarize(rows)
  self.assertEqual(result['auc'],1)
  self.assertEqual(result['within_case_strict_order'],1)
  self.assertTrue(result['best_complete_separation'])
 def test_ties_count_half_and_prevent_strict_order(self):
  rows=[{'case':0,'need':0,'score':.5}]+[{'case':0,'need':1,'score':s} for s in (.4,.5,.6)]
  result=summarize(rows)
  self.assertEqual(result['auc'],.5)
  self.assertEqual(result['within_case_strict_order'],0)
  self.assertFalse(result['best_complete_separation'])
