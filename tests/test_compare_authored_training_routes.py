import unittest
from scripts.compare_authored_training_routes import RUNS
class RouteTests(unittest.TestCase):
 def test_three_routes_bound(self):self.assertEqual(set(RUNS),{"P139_pretrain","P142_full_sft","P143_mixed_5pct"})
if __name__=="__main__":unittest.main()
