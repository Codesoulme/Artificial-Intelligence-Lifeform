import unittest
from scripts.deepseek_kernel_value_span_audit_20261117 import value_token_indices


class ValueSpanTests(unittest.TestCase):
    def test_exact_span_only(self):
        self.assertEqual(value_token_indices('甲喝豆浆。','豆浆',[(0,1),(1,2),(2,4),(4,5)]),[2])
        with self.assertRaises(ValueError):
            value_token_indices('甲喝豆浆。','豆浆',[(0,1),(1,3),(3,4),(4,5)])
