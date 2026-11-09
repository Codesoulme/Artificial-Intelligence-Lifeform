import copy
import unittest

from scripts import prepare_p231_wikipedia_pilot as pilot


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return type('Encoding', (), {'ids': list(text)})()


def row(pid, start=0, size=64, body=None):
    body = body or ('x' * size)
    prefix = 'T\n\n'
    return {'status': 'pending_review_not_approved_for_training', 'page_id': pid,
            'title': 'T', 'url': 'https://example.test', 'source_text_sha256': 'source',
            'section': None, 'char_start': start, 'char_end': start + len(body),
            'paragraphs': [{'char_start': start, 'char_end': start + len(body)}],
            'text': body, 'text_sha256': pilot.digest(body.encode()), 'model_text': prefix + body,
            'body_bpe_tokens': len(body), 'extra_heading_bpe_tokens': len(prefix),
            'model_bpe_tokens': len(prefix + body), 'model_minus_body_bpe_tokens': len(prefix)}


class PilotTests(unittest.TestCase):
    def test_samples_must_be_disjoint_and_complete(self):
        samples = [[{'page_id': str(i + 40 * n)} for i in range(40)] for n in range(3)]
        self.assertEqual(len(pilot.sample_pages(samples)), 120)
        samples[2][0] = samples[0][0]
        with self.assertRaises(ValueError):
            pilot.sample_pages(samples)

    def test_audit_excludes_whole_page(self):
        audit = {'candidate_manifest_sha256': 'm', 'candidate_sha256': 'c', 'candidate_count': 2,
                 'sources': {'old': {}}, 'matched_any_source': 1,
                 'candidates': [{'candidate_id': '7:0:12', 'sources': {'old': 1}},
                                {'candidate_id': '8:0:12', 'sources': {'old': 0}}]}
        manifest = {'input_sha256': {'refined_manifest': 'm'}, 'refined_candidate_sha256': 'c'}
        excluded = pilot.audit_pages(audit, manifest)
        groups = [{'page_id': '7', 'candidates': [row('7', 30)]},
                  {'page_id': '8', 'candidates': [row('8')]}]
        kept, stats = pilot.select(groups, excluded, Tokenizer())
        self.assertEqual([r['page_id'] for r in kept], ['8'])
        self.assertEqual(stats['excluded_input_pages'], 1)

    def test_thresholds_and_determinism(self):
        groups = [{'page_id': '1', 'candidates': [row('1', 0, 63), row('1', 100, 64)]},
                  {'page_id': '2', 'candidates': [row('2', 0, 508)]},
                  {'page_id': '3', 'candidates': [row('3', 0, 507, 'z' * 507)]}]
        kept, stats = pilot.select(groups, set(), Tokenizer())
        self.assertEqual({r['page_id'] for r in kept}, {'1', '3'})
        self.assertEqual(stats['pages_without_eligible_passage'], 1)
        self.assertEqual(kept, pilot.select(list(reversed(groups)), set(), Tokenizer())[0])
        for record in kept:
            self.assertEqual(record['model_text'], record['text'])
            self.assertEqual(pilot.identities(record['text']), (record['sha256'], record['group']))

    def test_identity_dedup_and_corruption(self):
        groups = [{'page_id': str(i), 'candidates': [row(str(i))]} for i in (1, 2)]
        kept, stats = pilot.select(groups, set(), Tokenizer())
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats['identity_duplicates_removed'], 1)
        bad = copy.deepcopy(groups)
        bad[0]['candidates'][0]['model_text'] += 'tamper'
        with self.assertRaises(ValueError):
            pilot.select(bad, set(), Tokenizer())


if __name__ == '__main__':
    unittest.main()
