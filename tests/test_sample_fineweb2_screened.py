import unittest
from scripts.sample_fineweb2_screened import key


class FinewebSampleTests(unittest.TestCase):
    def test_key_is_stable(self):
        row={"id":"x","url":"https://example.com/a","text":"甲"*500}
        self.assertEqual(key(row),key(row))


if __name__ == "__main__": unittest.main()
