"""Differential tests using generated text only; no corpus files are opened."""
import hashlib
import random
import unittest

from scripts.prepare_p28_prose import HeldIndex
from spora.fast_held_index import FastHeldIndex


def block(number):
    return f"{number:016x}"


class FastHeldIndexTests(unittest.TestCase):
    def setUp(self):
        self.slow = HeldIndex()
        self.fast = FastHeldIndex()

    def add(self, text, *extra):
        self.slow.add(text, *extra)
        self.fast.add(text, *extra)

    def check(self, text, expected=None):
        slow = self.slow.reason(text)
        self.assertEqual(slow, self.fast.reason(text))
        if expected is not None:
            self.assertEqual(slow, expected)

    def test_precedence_and_extra_keys(self):
        text = ''.join(block(i) for i in range(12))
        self.add(text)
        self.check(text, 'held_exact')
        self.check(text[:96] + block(1000), 'held_leading96')
        extra = 'another synthetic value'
        self.add('unrelated', None, hashlib.sha256(extra.replace(' ', '').encode()).hexdigest())
        self.check(extra, 'held_exact')
        self.add('third', hashlib.sha256(b'other').hexdigest())
        self.check('other', 'held_leading96')

    def test_thresholds_and_record_boundaries(self):
        target = ''.join(block(i) for i in range(10))
        self.add(target)
        # 8/10 qualifies, 7/10 does not. Leading blocks differ.
        self.check(''.join(block(i) for i in [90, 91, 2, 3, 4, 5, 6, 7, 8, 9]), 'held_sampled_lexical')
        self.check(''.join(block(i) for i in [90, 91, 92, 3, 4, 5, 6, 7, 8, 9]))
        self.assertIsNone(self.fast.reason(''.join(block(i) for i in [90, 91, 92, 3, 4, 5, 6, 7, 8, 9])))
        other = FastHeldIndex()
        other.add(''.join(block(i) for i in range(100, 110)))
        self.assertIsNone(other.reason(target))

    def test_randomized_and_repetitive_inputs(self):
        rng = random.Random(112)
        texts = ['', 'a' * 400, 'Ａ  b\n' * 55]
        texts += [''.join(block(rng.randrange(35)) for _ in range(rng.randrange(3, 35)))
                  for _ in range(250)]
        for text in texts:
            self.add(text)
        queries = texts + [''.join(block(rng.randrange(40)) for _ in range(rng.randrange(1, 40)))
                           for _ in range(500)]
        queries += [text[16:] + block(999) for text in texts if len(text) > 32]
        for text in queries:
            self.check(text)
        self.assertEqual(self.fast.sizes, self.slow.sizes)
        self.assertEqual(self.fast.index, self.slow.index)
        self.assertEqual(self.fast.exact, self.slow.exact)
        self.assertEqual(self.fast.groups, self.slow.groups)

    def test_short_threshold_and_no_cross_document_accumulation(self):
        self.add(''.join(block(i) for i in range(8)))
        self.check(''.join(block(i) for i in [90, 1, 2, 3, 4, 5, 6, 7]),
                   'held_sampled_lexical')  # 7/8, not 8/8
        self.add(''.join(block(i) for i in range(100, 110)))
        self.add(''.join(block(i) for i in range(200, 210)))
        mixed = ''.join(block(i) for i in [100, 101, 102, 103, 200, 201, 202, 203, 900, 901])
        self.check(mixed)
        self.assertIsNone(self.slow.reason(mixed))

    def test_no_cross_index_accumulation_and_tag_order(self):
        from scripts.p112_controlled_screen import screen
        q = ''.join(block(i) for i in range(10))
        for cls in (HeldIndex, FastHeldIndex):
            left, right = cls(), cls()
            left.add(''.join(block(i) for i in [0, 1, 2, 3, 40, 41, 42, 43, 44, 45]))
            right.add(''.join(block(i) for i in [4, 5, 6, 7, 50, 51, 52, 53, 54, 55]))
            self.assertIsNone(screen(q, [('held', left), ('original_train', right)]))
            left.add(q)
            right.add(q)
            self.assertEqual(screen(q, [('held', left), ('original_train', right)]), 'held_held_exact')


if __name__ == '__main__':
    unittest.main()
