import unittest

from spora.training_guard import (assess_guard, validate_guard, assess_multiturn_guard,
                                  validate_multiturn_guard)


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

    def test_multiturn_regression_and_mismatch(self):
        config = {'cases': 'data/dev.json', 'cases_sha256': 'a' * 64, 'budget': 96,
                  'max_ended_drop': 4, 'max_loop_increase': 4}
        baseline = {'turns': 36, 'ended': 27, 'loops': 9}
        current = {'turns': 36, 'ended': 12, 'loops': 19}
        result = assess_multiturn_guard(config, baseline, current)
        self.assertEqual(result['reasons'], ['multiturn_ended_drop', 'multiturn_loop_increase'])
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            assess_multiturn_guard(config, baseline, dict(current, turns=35))
        with self.assertRaises(ValueError):
            validate_multiturn_guard(dict(config, cases_sha256='bad'))
        self.assertIsNone(assess_multiturn_guard(None, None, None))


if __name__ == '__main__':
    unittest.main()
