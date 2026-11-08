import copy
import json
import unittest

import torch

from spora.mixed_target_budget import MixedTargetBudget, normalized_loss


def records():
    return {
        'story': [
            {'id': 's1', 'tokens': [1, 2, 3, 4, 5], 'mask': [False, True, True, True, True]},
            {'id': 's2', 'tokens': [1, 6, 7, 8], 'mask': [False, True, False, True]},
        ],
        'dialogue': [
            {'id': 'd1', 'tokens': [1, 9, 10, 11, 12], 'mask': [False, False, True, False, True]},
            {'id': 'd2', 'tokens': [1, 13, 14, 15], 'mask': [False, False, True, True]},
        ],
    }


PLAN = [{'story': 3, 'dialogue': 1}, {'story': 2, 'dialogue': 2},
        {'story': 1, 'dialogue': 1}]


def make(sources=None, plan=None, size=2):
    return MixedTargetBudget(records() if sources is None else sources,
                             PLAN if plan is None else plan, microbatch_size=size)


class MixedTargetBudgetTests(unittest.TestCase):
    def test_exact_quotas_context_and_no_lost_targets(self):
        schedule = make(size=1)
        originals = {(source, row['id']): row for source, rows in records().items() for row in rows}
        seen = {key: [] for key in originals}
        for index, quotas in enumerate(PLAN):
            batch = schedule.prepare()
            self.assertEqual(batch['update'], index)
            self.assertEqual(batch['denominator'], sum(quotas.values()))
            counts = dict.fromkeys(quotas, 0)
            for micro in batch['microbatches']:
                self.assertEqual(micro['denominator'], batch['denominator'])
                self.assertEqual(micro['target_count'], sum(sum(r['mask']) for r in micro['rows']))
                for row in micro['rows']:
                    key = row['source'], row['id']
                    original = originals[key]
                    self.assertEqual(row['tokens'], original['tokens'])
                    positions = [i for i, x in enumerate(row['mask']) if x]
                    self.assertTrue(all(original['mask'][i] for i in positions))
                    self.assertEqual(len(positions), row['target_count'])
                    counts[row['source']] += len(positions)
                    seen[key].extend(positions)
            self.assertEqual(counts, quotas)
            schedule.commit()
        for key, original in originals.items():
            self.assertEqual(seen[key], [i for i, x in enumerate(original['mask']) if x])
        with self.assertRaises(StopIteration):
            schedule.prepare()

    def test_gradient_is_global_mean_with_unequal_microbatches(self):
        torch.set_num_threads(1)
        schedule = make(size=1)
        batch = schedule.prepare()
        param = torch.tensor(0.25, requires_grad=True)
        all_values = []
        for micro in batch['microbatches']:
            values = [float(token) for row in micro['rows']
                      for token, enabled in zip(row['tokens'], row['mask']) if enabled]
            all_values.extend(values)
            loss_sum = sum((param - value) ** 2 for value in values)
            normalized_loss(loss_sum, target_count=micro['target_count'],
                            denominator=micro['denominator']).backward()
        reference = torch.tensor(0.25, requires_grad=True)
        (sum((reference - v) ** 2 for v in all_values) / len(all_values)).backward()
        self.assertEqual(len(all_values), batch['denominator'])
        self.assertTrue(torch.allclose(param.grad, reference.grad))

    def test_resume_every_update_and_pending_transaction(self):
        uninterrupted = make()
        expected_batches, states = [], [uninterrupted.state_dict()]
        for _ in PLAN:
            batch = uninterrupted.prepare()
            expected_batches.append(copy.deepcopy(batch))
            batch['microbatches'][0]['rows'][0]['tokens'][0] = 999
            self.assertNotEqual(uninterrupted.prepare(), batch)
            with self.assertRaises(ValueError):
                uninterrupted.state_dict()
            with self.assertRaises(ValueError):
                uninterrupted.load_state_dict(states[-1])
            uninterrupted.commit()
            states.append(uninterrupted.state_dict())
        for cut, state in enumerate(states):
            resumed = make()
            resumed.load_state_dict(json.loads(json.dumps(state)))
            for expected in expected_batches[cut:]:
                self.assertEqual(resumed.prepare(), expected)
                resumed.commit()
            self.assertEqual(resumed.state_dict(), states[-1])

    def test_invalid_or_tampered_inputs_do_not_restore(self):
        with self.assertRaises(ValueError):
            make(plan=[{'story': 7, 'dialogue': 0}])
        with self.assertRaises(ValueError):
            make(plan=[{'story': True, 'dialogue': 1}])
        with self.assertRaises(ValueError):
            make(size=True)
        bad_rows = records(); bad_rows['story'][0]['mask'][0] = True
        with self.assertRaises(ValueError):
            make(sources=bad_rows)
        first = make(); first.prepare(); first.commit(); state = first.state_dict()
        for field, value in [('update', True), ('fingerprint', 'bad'),
                             ('totals', {'story': 0, 'dialogue': 0}),
                             ('positions', {'story': [True, 0], 'dialogue': [0, 1]})]:
            tampered = copy.deepcopy(state); tampered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                first.load_state_dict(tampered)
            self.assertEqual(first.state_dict(), state)
        changed = records(); changed['story'][0]['tokens'][1] = 100
        with self.assertRaises(ValueError):
            make(sources=changed).load_state_dict(state)
        with self.assertRaises(ValueError):
            normalized_loss(1, target_count=4, denominator=3)
        with self.assertRaises(ValueError):
            make().commit()


if __name__ == '__main__':
    unittest.main()
