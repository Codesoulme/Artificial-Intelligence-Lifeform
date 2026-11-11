import hashlib
import unittest

from scripts.deepseek_spora_cluener_dual_span_audit import select_fresh


def labeled(text, names):
    entities = []
    for name in names:
        start = text.index(name)
        entities.append({'label': 'name', 'start_offset': start,
                         'end_offset': start + len(name), 'entity': name})
    return {'text': text, 'entities': entities}


class CluenerDualSpanSelectionTests(unittest.TestCase):
    def test_selection_excludes_prior_and_deduplicates(self):
        singles = [labeled(f'这是第{i:03d}条甲记录。', ['甲']) for i in range(21)]
        multiple = [labeled(f'这是第{i:03d}条甲乙记录。', ['甲', '乙']) for i in range(10)]
        none = [labeled(f'这是第{i:03d}条普通记录。', []) for i in range(10)]
        digest = hashlib.sha256(singles[0]['text'].encode('utf-8')).hexdigest()
        prior = [{'text_sha256': digest}]
        selected = select_fresh(singles + singles[:1] + multiple + none, prior)
        self.assertEqual(sum(row['group'] == 'single' for row in selected), 20)
        self.assertEqual(sum(row['group'] == 'multiple' for row in selected), 10)
        self.assertEqual(sum(row['group'] == 'none' for row in selected), 10)
        self.assertNotIn(digest, {row['digest'] for row in selected})
        self.assertEqual(len({row['digest'] for row in selected}), 40)

    def test_missing_group_fails_instead_of_silently_shrinking(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_fresh([labeled('这是一个甲的名字记录。', ['甲'])], [])


if __name__ == '__main__':
    unittest.main()
