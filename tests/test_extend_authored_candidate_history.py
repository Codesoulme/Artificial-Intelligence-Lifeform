import unittest
from scripts.extend_authored_candidate_history import NS
class HistoryTests(unittest.TestCase):
 def test_namespace(self):self.assertIn("w",NS)
if __name__=="__main__":unittest.main()
