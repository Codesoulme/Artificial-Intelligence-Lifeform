import copy
import unittest

from scripts import prepare_p231_dialogue_candidates as p


class Encoding:
    def __init__(self, ids):
        self.ids = ids


class Tokenizer:
    def token_to_id(self, token):
        return {'<|bos|>': 1, '<|user|>': 2, '<|assistant|>': 3, '<|eos|>': 4}[token]

    def encode(self, text):
        return Encoding([5] * len(text))


def row(size=10):
    return {'group': 'original', 'sha256': 'source-hash', 'source_line': 10,
            'messages': [{'role': role, 'content': '中' * size} for role in ('user', 'assistant', 'user', 'assistant')]}


def entry(group='group', family='family', original='candidate', proposed='candidate'):
    return {'group_sha256': group, 'family_sha256': family, 'original': original, 'proposed': proposed}


class CandidateTests(unittest.TestCase):
    def test_fields_and_mask(self):
        records, counts = p.prepare_rows([(1, entry(), row())], Tokenizer())
        r = records[0]
        self.assertEqual(r['messages'], row()['messages'])
        self.assertEqual(r['origins'][0]['line'], 1)
        self.assertEqual(r['origins'][0]['source_group'], 'original')
        self.assertEqual(r['dialogue_sha256'], p.dialogue_hash(r['messages']))
        self.assertEqual(r['supervised_tokens'], 22)
        self.assertEqual(len(r['tokens']), len(r['mask']))
        self.assertEqual(counts['retained_source_rows'], 1)
        self.assertEqual(r['status'], p.STATUS)

    def test_length_is_not_truncated(self):
        records, counts = p.prepare_rows([(1, entry(), row(126)), (2, entry('other', 'other'), row(125))], Tokenizer())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['token_count'], 509)
        self.assertEqual(counts['unique_over_512_excluded'], 1)

    def test_deduplication_preserves_origins_only_within_family(self):
        records, counts = p.prepare_rows([(1, entry(), row()), (2, entry('g2'), row()), (3, entry('g3', 'different'), row())], Tokenizer())
        self.assertEqual(len(records), 2)
        self.assertEqual([o['line'] for o in records[0]['origins']], [1, 2])
        self.assertEqual(counts['same_family_duplicate_rows'], 1)
        self.assertEqual(counts['retained_source_rows'], 3)

    def test_hash_covers_roles_and_normalizes_text(self):
        a = row()['messages']
        b = copy.deepcopy(a)
        b[0]['content'] += '  '
        self.assertEqual(p.dialogue_hash(a), p.dialogue_hash(b))
        b[0]['role'] = 'assistant'
        self.assertNotEqual(p.dialogue_hash(a), p.dialogue_hash(b))

    def test_cross_partition_family_rejected(self):
        families = [entry('a'), entry('b', original='development', proposed='development')]
        for i in range(4):
            families.append(entry(str(i), str(i), proposed='quarantine'))
        summary = {'input_sha256': {'training': p.EXPECTED['source'], 'p104_index': p.EXPECTED['index']},
                   'proposed_changes': {'candidate->quarantine': 4}, 'family_count': 5}
        index = {'source_sha256': p.EXPECTED['source'], 'rows': []}
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            p.validate_index(index, summary, families)

    def test_sampling_is_deterministic_and_one_per_family(self):
        rows = [{'family_sha256': str(i), 'dialogue_sha256': str(i)} for i in range(45)]
        chosen = p.select_review(rows)
        self.assertEqual(chosen, p.select_review(list(reversed(rows))))
        self.assertEqual(len({r['family_sha256'] for r in chosen}), 40)


if __name__ == '__main__':
    unittest.main()
