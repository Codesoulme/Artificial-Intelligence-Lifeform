import copy
import json
import unittest

from spora.paired_order import PairedOrder, padded


def rows():
    left, right = [], []
    for i in range(4):
        base = {'pair_id': str(i), 'group': 'g' + str(i // 2)}
        left.append(dict(base, tokens=[1, 8, i, 9, i + 10, 2], mask=[False, False, False, False, True, True]))
        right.append(dict(base, tokens=[1, 9, i + 10, 2], mask=[False, False, True, True]))
    return left, right


def make(left=None, right=None, **kwargs):
    a, b = rows()
    return PairedOrder(a if left is None else left, b if right is None else right,
                       seed=kwargs.get('seed', 19), epochs=kwargs.get('epochs', 3), length=kwargs.get('length', 5))


class PairedOrderTests(unittest.TestCase):
    def test_alignment_repetition_padding_and_order(self):
        a, b = rows()
        order = make(a, list(reversed(b)))
        history = []
        while True:
            try:
                batch = order.prepare(5)
            except StopIteration:
                break
            x, y = padded(batch['context'], 5, 0)
            u, v = padded(batch['last_user'], 5, 0)
            self.assertEqual(len(x), len(u))
            for event, left, right in zip(batch['events'], y, v):
                self.assertEqual([t for t in left if t != -100], [t for t in right if t != -100])
                self.assertEqual(event['target_count'], 2)
            history.extend(batch['events'])
            order.commit()
        self.assertEqual(len(history), 12)
        for key in ('0', '1', '2', '3'):
            self.assertEqual([e['exposure'] for e in history if e['pair_id'] == key], [1, 2, 3])
        for epoch in range(3):
            self.assertEqual({e['pair_id'] for e in history if e['epoch'] == epoch}, {'0', '1', '2', '3'})
        again = make(); again.prepare(12); again.commit()
        self.assertEqual(history, again.state_dict()['events'])

    def test_resume_at_every_position_and_pending_retry(self):
        reference = make(); reference.prepare(12); reference.commit()
        expected = reference.state_dict()
        for cut in range(13):
            first = make()
            if cut:
                batch = first.prepare(cut)
                batch['events'][0]['pair_id'] = 'tampered'
                self.assertNotEqual(first.prepare(cut)['events'][0]['pair_id'], 'tampered')
                with self.assertRaises(ValueError): first.state_dict()
                first.commit()
            state = json.loads(json.dumps(first.state_dict()))
            recovered = make(); recovered.load_state_dict(state)
            if cut < 12:
                recovered.prepare(12); recovered.commit()
            self.assertEqual(recovered.state_dict(), expected)

    def test_invalid_inputs_and_restores(self):
        a, b = rows()
        cases = []
        cases.append(([], []))
        cases.append((a + a[:1], b))
        cases.append((a, b[:-1]))
        for field, value in [('group', 'other'), ('tokens', [1, 9, 44, 2]),
                             ('mask', [True, False, True, True])]:
            altered = copy.deepcopy(b); altered[0][field] = value; cases.append((a, altered))
        for left, right in cases:
            with self.subTest(left=left, right=right), self.assertRaises(ValueError): make(left, right)
        for kw in ({'epochs': 0}, {'seed': True}, {'length': 4}):
            with self.assertRaises(ValueError): make(**kw)
        order = make(); order.prepare(3); order.commit(); state = order.state_dict()
        for key, value in [('cursor', 4), ('fingerprint', 'wrong'), ('seed', True)]:
            bad = copy.deepcopy(state); bad[key] = value
            with self.assertRaises(ValueError): make().load_state_dict(bad)
        bad = copy.deepcopy(state); bad['events'][0]['exposure'] = 42
        with self.assertRaises(ValueError): order.load_state_dict(bad)
        self.assertEqual(order.state_dict(), state)
        changed = copy.deepcopy(b); changed[0]['tokens'][1] = 7
        with self.assertRaises(ValueError): make(a, changed).load_state_dict(state)

    def test_batch_boundaries_and_transaction(self):
        order = make(epochs=1)
        for value in (0, -1, True):
            with self.assertRaises(ValueError): order.prepare(value)
        with self.assertRaises(ValueError): order.commit()
        batch = order.prepare(3)
        with self.assertRaises(ValueError): order.prepare(2)
        with self.assertRaises(ValueError): order.load_state_dict({})
        order.commit()
        self.assertEqual(len(order.prepare(20)['events']), 1)
        order.commit()
        with self.assertRaises(StopIteration): order.prepare(1)
        self.assertEqual(len(batch['events']), 3)


if __name__ == '__main__':
    unittest.main()
