import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_p230_blind as prep


class BlindQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'runs').mkdir()
        self.sources = {}
        for name, (_, checkpoint, step) in prep.SOURCES.items():
            data = dict(checkpoint_sha256=checkpoint, step=step, cases_sha256=prep.CASES_SHA256,
                        checkpoint_unchanged=True, old_individual_exact=True,
                        summary={'status': 'ungraded_development_not_acceptance', 'turns': 24},
                        rows=[{'scenario': f'scenario-{s}', 'turn': t, 'prompt': f'prompt-{s}-{t}',
                               'text': f'{name}-{s}-{t}'} for s in range(8) for t in range(1, 4)])
            data.update({key: 'same' for key in prep.SHARED_FIELDS})
            self.sources[name] = data
        self.write_sources()

    def write_sources(self):
        for name, data in self.sources.items():
            (self.root / prep.SOURCES[name][0]).write_text(json.dumps(data), encoding='utf-8')

    def test_queue_preserves_each_chain_and_reproducible_mapping(self):
        for name in ('one', 'two'):
            prep.build(self.root, self.root / 'runs' / name)
        one = self.root / 'runs' / 'one'
        two = self.root / 'runs' / 'two'
        for filename in ('blind.json', 'local_mapping.json'):
            self.assertEqual((one / filename).read_bytes(), (two / filename).read_bytes())
        queue = json.loads((one / 'blind.json').read_text(encoding='utf-8'))
        manifest = json.loads((one / 'local_mapping.json').read_text(encoding='utf-8'))
        self.assertEqual(len(queue), 16)
        self.assertEqual(len(manifest['mapping']), 16)
        self.assertEqual(manifest['scope'], 'development_only_not_independent_final_test')
        for item in queue:
            self.assertEqual(set(item), {'anonymous_id', 'turns'})
            source = manifest['mapping'][item['anonymous_id']]
            original = [r for r in self.sources[source['source']]['rows'] if r['scenario'] == source['scenario']]
            self.assertEqual(item['turns'], [{'prompt': r['prompt'], 'text': r['text']} for r in original])
        with self.assertRaises(FileExistsError):
            prep.build(self.root, one)
        with self.assertRaises(ValueError):
            prep.build(self.root, self.root / 'outside')

    def test_rejects_invalid_sources_before_creating_output(self):
        mutations = [
            lambda d: d.update(checkpoint_sha256='wrong'),
            lambda d: d.update(cases_sha256='wrong'),
            lambda d: d.update(policy='different'),
            lambda d: d['rows'][0].update(prompt='different'),
            lambda d: d['rows'][1].update(turn=3),
            lambda d: d['rows'][3].update(scenario='scenario-0'),
            lambda d: d['rows'].pop(),
        ]
        original = copy.deepcopy(self.sources['p229'])
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.sources['p229'] = copy.deepcopy(original)
                mutate(self.sources['p229'])
                self.write_sources()
                output = self.root / 'runs' / 'invalid'
                with self.assertRaises(ValueError):
                    prep.build(self.root, output)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
