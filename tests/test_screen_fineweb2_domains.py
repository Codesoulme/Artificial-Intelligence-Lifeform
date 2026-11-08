import unittest
from scripts.screen_fineweb2_domains import flags


class DomainTests(unittest.TestCase):
    def test_flags_suspicious_domain(self):
        self.assertIn("suspicious_keyword", flags("example-porn123.com"))

    def test_clean_domain_not_flagged(self):
        self.assertEqual(flags("example.org"), [])


if __name__ == "__main__": unittest.main()
