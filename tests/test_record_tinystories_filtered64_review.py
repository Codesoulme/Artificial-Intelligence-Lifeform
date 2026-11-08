import unittest
from scripts.record_tinystories_filtered64_review import STATUS
class TinyReviewTests(unittest.TestCase):
 def test_has_64_distinct_reviews(self):self.assertEqual(len(STATUS),64)
if __name__=='__main__':unittest.main()
