import unittest
from scripts.prepare_authored_pretrain_adapter import split

class AdapterTests(unittest.TestCase):
 def test_split_is_stable(self):
  row={"record_id":"x","text":"x"};self.assertEqual(split(row),split(row))
if __name__=="__main__":unittest.main()
