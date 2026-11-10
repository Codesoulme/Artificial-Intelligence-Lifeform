import unittest

from spora.temporal_routing import CURRENT, LATER, PAST, route_spec


class TemporalRoutingTests(unittest.TestCase):
    def test_each_predicted_intent_selects_its_memory_page(self):
        self.assertEqual(route_spec(CURRENT), (0, None))
        self.assertEqual(route_spec(PAST), (1, 0))
        self.assertEqual(route_spec(LATER), (1, 1))

    def test_unknown_intent_is_rejected(self):
        with self.assertRaises(ValueError):
            route_spec(99)


if __name__ == '__main__':
    unittest.main()
