import unittest

from spora.training_guard import assess_guard, validate_guard


def report(nll, endings):
    return {'dialogue_nll': nll, 'samples': [
        {'prompt': str(i), 'ended_with_eos': ended} for i, ended in enumerate(endings)]}


class TrainingGuardTests(unittest.TestCase):
    def test_thresholds_and_reasons(self):
        config = {'max_dialogue_nll_increase': .15, 'max_ended_drop': 1}
        base = report(3., [True, True, True])
        self.assertFalse(assess_guard(config, base, report(3.1, [True, True, False]))['stop'])
        result = assess_guard(config, base, report(3.2, [True, False, False]))
        self.assertEqual(result['reasons'], ['dialogue_nll_increase', 'ended_count_drop'])
        self.assertIsNone(assess_guard(None, None, None))

    def test_rejects_invalid_or_changed_inputs(self):
        config = {'max_dialogue_nll_increase': .1, 'max_ended_drop': 1}
        for bad in (-1, float('nan'), True):
            with self.assertRaises(ValueError):
                validate_guard(dict(config, max_dialogue_nll_increase=bad))
        changed = report(3., [True])
        changed['samples'][0]['prompt'] = 'different'
        with self.assertRaisesRegex(ValueError, 'prompts differ'):
            assess_guard(config, report(3., [True]), changed)
        with self.assertRaisesRegex(ValueError, 'Invalid guard report'):
            assess_guard(config, report(3., [True]), report(float('nan'), [True]))


if __name__ == '__main__':
    unittest.main()
