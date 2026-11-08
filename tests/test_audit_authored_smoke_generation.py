import unittest
from scripts.audit_authored_smoke_generation import RUNS
class GenerationAuditTests(unittest.TestCase):
 def test_runs_exist(self):
  self.assertTrue(all(p.exists() for p in RUNS.values()))
if __name__=="__main__":unittest.main()
