import unittest
from spora.research_dialogue_folds import make_folds


class DialogueFoldTests(unittest.TestCase):
    def fixture(self):
        rows, mapping = [], {}
        for c in range(60):
            for suffix in ('q', 'h'):
                mapping[f'{c}{suffix}'] = f'tree{c}'
        for c in range(60):
            for variant, history, label in [('question', [], c % 2), ('original', [f'{c}h'], 0),
                                            ('wrong', [f'{(c+1)%60}h'], c % 2)]:
                rows.append({'case': c, 'group': f'tree{c}', 'history_dependent': c % 2,
                             'current_message_id': f'{c}q', 'history_message_ids': history,
                             'variant': variant, 'need': label})
        return rows, mapping

    def test_borrowed_history_is_purged_and_test_coverage_complete(self):
        rows, mapping = self.fixture()
        folds = make_folds(rows, mapping, 3)
        self.assertEqual(sorted(i for f in folds for i in f['test']), list(range(len(rows))))
        self.assertGreater(sum(f['purged_train_rows']+f['purged_calibration_rows'] for f in folds), 0)
        for fold in folds:
            groups = [set(fold['source_trees'][k]) for k in ('train', 'calibration', 'test')]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_unknown_provenance_rejected(self):
        rows, mapping = self.fixture()
        del mapping['0h']
        with self.assertRaisesRegex(ValueError, 'Unknown provenance'):
            make_folds(rows, mapping, 3)
