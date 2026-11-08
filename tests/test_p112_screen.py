import unittest
from scripts.p112_controlled_screen import select_eligible, screen, index_digest
from scripts.prepare_p28_prose import HeldIndex
from scripts.prepare_p28_stories import identities, digest


class ControlledScreenTests(unittest.TestCase):
    def test_metadata_exclusions_are_disjoint_and_repeatable(self):
        rows = [('a','g1',0,'x',10), ('b','g1',1,'x',10),
                ('c','g2',2,'x',10), ('d','g3',3,'x',10)]
        result, counts, first = select_eligible(rows, {'a'}, {'g1'}, {'g2'})
        self.assertEqual([r[1] for r in result], ['d'])
        self.assertEqual(dict(counts), {'original_exact': 1, 'original_group_only': 1, 'other_partition_group': 1})
        self.assertEqual(first, select_eligible(rows, {'a'}, {'g1'}, {'g2'})[2])

    def test_exact_prefix_and_finite_fragment_filter_and_safe_text(self):
        base = ''.join(chr(0x4e00+i) for i in range(200))
        held = HeldIndex(); held.add(base)
        index = [('held', held)]
        self.assertEqual(screen(base, index), 'held_held_exact')
        self.assertEqual(screen(base[:96]+'结尾不同', index), 'held_held_leading96')
        self.assertEqual(screen('其他内容'*4+base[32:192], index), 'held_held_sampled_lexical')
        self.assertIsNone(screen('完全独立的合成文字。', index))
        self.assertEqual(index_digest(held), index_digest(held))
        changed = HeldIndex(); changed.add(base+'不同')
        self.assertNotEqual(index_digest(held), index_digest(changed))

    def test_unicode_normalization_and_original_train_index(self):
        idx = HeldIndex(); idx.add('ＡＢＣ １２３')
        self.assertEqual(screen('abc123', [('original_train', idx)]), 'original_train_held_exact')
        self.assertEqual(identities('ＡＢＣ １２３'), identities('abc123'))
        self.assertNotEqual(digest('ＡＢＣ １２３'), digest('abc123'))

if __name__ == '__main__':
    unittest.main()
