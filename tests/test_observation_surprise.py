import unittest

from spora.observation_surprise import observation_surprise


class ObservationSurpriseTests(unittest.TestCase):
    def test_new_fact_is_maximally_novel(self):
        result = observation_surprise(known=False, old_value=16, new_value=3,
                                      old_quality=0, new_quality=1, old_time=-1, new_time=0)
        self.assertEqual(result['score'], 1.0)
        self.assertEqual(result['components']['new_fact'], 1.0)

    def test_correction_is_maximally_novel(self):
        result = observation_surprise(known=True, old_value=3, new_value=6,
                                      old_quality=.8, new_quality=.8, old_time=1, new_time=2)
        self.assertEqual(result['score'], 1.0)
        self.assertEqual(result['components']['value_change'], 1.0)

    def test_repeat_is_quiet(self):
        result = observation_surprise(known=True, old_value=3, new_value=3,
                                      old_quality=.8, new_quality=.8, old_time=1, new_time=1)
        self.assertEqual(result['score'], 0.0)

    def test_quality_gain_is_bounded(self):
        result = observation_surprise(known=True, old_value=3, new_value=3,
                                      old_quality=.2, new_quality=1, old_time=1, new_time=1)
        self.assertEqual(result['score'], .4)


if __name__ == '__main__':
    unittest.main()
