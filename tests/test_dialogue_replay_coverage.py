import copy
import json
import random
import unittest

from spora.dialogue_replay_coverage import DialogueReplayCoverage, _digest
from scripts.train_p111_chunk_coverage import record_dialogue


SOURCE = 'a' * 64
RECORDS = [
    {'source': SOURCE, 'pair_id': 'one', 'mask': [False, True, False, True], 'text': 'private dialogue'},
    {'source': SOURCE, 'pair_id': 'two', 'mask': [False, False, True, True, True]},
]


def resign(state):
    state['checksum'] = _digest({k: state[k] for k in ('version', 'fingerprint', 'records')})
    return state


class DialogueReplayCoverageTests(unittest.TestCase):
    def test_training_accounting_excludes_replaced_rows(self):
        coverage = DialogueReplayCoverage(RECORDS)
        rows = [RECORDS[0], RECORDS[1], RECORDS[0]]
        self.assertEqual(record_dialogue(coverage, SOURCE, rows, {0}), 5)
        self.assertEqual(coverage.summaries()[(SOURCE, 'one')]['presentations'], 1)
        self.assertEqual(coverage.summaries()[(SOURCE, 'two')]['target_presentations'], 3)

    def test_masks_partial_presentations_and_roundtrip(self):
        coverage = DialogueReplayCoverage(RECORDS)
        coverage.add(SOURCE.upper(), 'one')
        coverage.add(SOURCE, 'one', [3])
        coverage.add(SOURCE, 'two', [2, 4])
        state = json.loads(json.dumps(coverage.to_dict()))
        self.assertNotIn('private dialogue', json.dumps(state))
        restored = DialogueReplayCoverage.from_dict(RECORDS, state)
        self.assertEqual(restored.summaries()[(SOURCE, 'one')], {
            'presentations': 2, 'target_presentations': 3,
            'distinct_target_positions': 2, 'duplicate_target_positions': 1})
        restored.add(SOURCE, 'two')
        self.assertEqual(restored.summaries()[(SOURCE, 'two')], {
            'presentations': 2, 'target_presentations': 5,
            'distinct_target_positions': 3, 'duplicate_target_positions': 2})
        self.assertEqual(coverage.summaries()[(SOURCE, 'two')]['presentations'], 1)

    def test_random_against_reference(self):
        rng = random.Random(108)
        coverage = DialogueReplayCoverage(RECORDS)
        reference = {'one': [], 'two': []}
        for step in range(200):
            pair, valid = rng.choice([('one', [1, 3]), ('two', [2, 3, 4])])
            positions = rng.sample(valid, rng.randrange(1, len(valid) + 1))
            coverage.add(SOURCE, pair, positions)
            reference[pair].append(positions)
            if step % 17 == 0:
                coverage = DialogueReplayCoverage.from_dict(RECORDS, json.loads(json.dumps(coverage.to_dict())))
        for pair, events in reference.items():
            all_positions = [p for event in events for p in event]
            self.assertEqual(coverage.summaries()[(SOURCE, pair)], {
                'presentations': len(events), 'target_presentations': len(all_positions),
                'distinct_target_positions': len(set(all_positions)),
                'duplicate_target_positions': len(all_positions) - len(set(all_positions))})

    def test_invalid_registration_and_add_are_atomic(self):
        invalid = [[], [RECORDS[0], {**RECORDS[0], 'source': SOURCE.upper()}],
                   [{**RECORDS[0], 'mask': [True, True]}],
                   [{**RECORDS[0], 'mask': [False, 1]}],
                   [{**RECORDS[0], 'mask': [False, False]}],
                   [{**RECORDS[0], 'source': 'bad'}], [{**RECORDS[0], 'pair_id': ''}]]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                DialogueReplayCoverage(records)
        coverage = DialogueReplayCoverage(RECORDS)
        original = coverage.to_dict()
        for source, pair, positions in [(SOURCE, 'one', []), (SOURCE, 'one', [0]),
                (SOURCE, 'one', [2]), (SOURCE, 'one', [4]), (SOURCE, 'one', [True]),
                (SOURCE, 'one', [1, 1]), (SOURCE, 'missing', [1]), ('bad', 'one', [1])]:
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                coverage.add(source, pair, positions)
            self.assertEqual(coverage.to_dict(), original)

    def test_rejects_changed_and_inconsistent_snapshots(self):
        coverage = DialogueReplayCoverage(RECORDS)
        coverage.add(SOURCE, 'one')
        original = coverage.to_dict()
        changed = copy.deepcopy(original)
        changed['records'][0]['presentations'] += 1
        with self.assertRaises(ValueError):
            DialogueReplayCoverage.from_dict(RECORDS, changed)
        variations = []
        for change in ('duplicate', 'missing', 'position', 'count', 'presentations', 'extra'):
            state = copy.deepcopy(original)
            row = state['records'][0]
            if change == 'duplicate':
                state['records'].append(copy.deepcopy(row))
            elif change == 'missing':
                state['records'].pop()
            elif change == 'position':
                row['positions'][0][0] = 2
            elif change == 'count':
                row['positions'][0][1] = 2
            elif change == 'presentations':
                row['presentations'] = 3
            else:
                row['text'] = 'secret'
            variations.append(resign(state))
        for state in variations:
            with self.subTest(state=state), self.assertRaises(ValueError):
                DialogueReplayCoverage.from_dict(RECORDS, state)
        changed_records = copy.deepcopy(RECORDS)
        changed_records[0]['mask'][2] = True
        with self.assertRaises(ValueError):
            DialogueReplayCoverage.from_dict(changed_records, original)


if __name__ == '__main__':
    unittest.main()
