import unittest

from scripts.deepseek_mmc_generation_long import previous_samples


class GenerationLongTests(unittest.TestCase):
    def test_requires_same_unfinished_prior_sample(self):
        sample = {'sample_sha256': 'one', 'label': 1}
        old = {'sample_sha256': 'one', 'label': 1, 'mode': 'normal', 'ended': False, 'generated_tokens': 128}
        self.assertEqual(previous_samples({'results': [old]}, [sample])['one'], old)
        for change in ({'ended': True}, {'generated_tokens': 127}, {'label': 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                previous_samples({'results': [dict(old, **change)]}, [sample])

    def test_rejects_duplicate_previous_sample(self):
        old = {'sample_sha256': 'one', 'label': 1, 'mode': 'normal', 'ended': False, 'generated_tokens': 128}
        with self.assertRaises(ValueError):
            previous_samples({'results': [old, old]}, [{'sample_sha256': 'one', 'label': 1}])


if __name__ == '__main__':
    unittest.main()
