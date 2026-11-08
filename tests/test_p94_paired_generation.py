import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts import eval_p94_paired_generation as ev
from spora.document_stream import DocumentStream
from spora.language_training import interface_state
from spora.paired_experiment import coverage
from spora.source_review import file_sha256
from tests.test_paired_experiment import PairedExperimentTests


class FakeTokenizer:
    def decode(self, tokens, **kwargs):
        return ','.join(map(str, tokens))


class Fixed(torch.nn.Module):
    def __init__(self, token):
        super().__init__()
        self.config = type('Config', (), {'context': 16})()
        self.token = token
    def forward(self, x, state):
        logits = torch.zeros((1, x.shape[1], 8))
        logits[:, :, self.token] = 10
        return logits


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_generation_eos_budget_and_repetitions(self):
        a = ev.generate(Fixed(2), None, [1, 3], FakeTokenizer(), 1, 2, 0, 4)
        self.assertTrue(a['ended_with_eos']); self.assertEqual(a['token_ids'], [])
        b = ev.generate(Fixed(4), None, [1, 3], FakeTokenizer(), 1, 2, 0, 4)
        self.assertTrue(b['budget_exhausted']); self.assertEqual(b['token_ids'], [4]*4)
        self.assertEqual(b['repetition']['2'], {'total': 3, 'repeated': 2})
        with self.assertRaises(ValueError):
            ev.generate(Fixed(4), None, [1]*14, FakeTokenizer(), 1, 2, 0, 4)

    def test_device_selection_cpu_and_unavailable_cuda(self):
        with patch.object(ev.torch.cuda, 'is_available', side_effect=AssertionError('CPU must not probe CUDA')):
            self.assertEqual(ev.validate_device('cpu'), 'cpu')
        with patch.object(ev.torch.cuda, 'is_available', return_value=False), \
             patch.object(sys, 'argv', ['eval', '--output', 'runs/not-created.json', '--device', 'cuda']):
            with self.assertRaisesRegex(RuntimeError, 'CUDA requested but unavailable'):
                ev.main()

    def test_restored_cpu_models_generate_without_state_changes(self):
        trainer = PairedExperimentTests().make()
        trainer.step()
        payload = dict(spec=trainer.spec, frozen=trainer.before,
                       models=[interface_state(m) for m in trainer.models])
        models = ev.restore_models(trainer.organism, trainer.state, payload, 'cpu')
        before = ev.frozen_digest(trainer.organism, trainer.state)
        for model in models:
            weights = ev.tensor_hash(interface_state(model))
            result = ev.generate(model, trainer.state, [1, 3, 4], FakeTokenizer(), 1, 2, 0, 4, 'cpu')
            self.assertLessEqual(result['generated_tokens'], 4)
            self.assertEqual(weights, ev.tensor_hash(interface_state(model)))
            self.assertEqual(before, ev.frozen_digest(trainer.organism, trainer.state))
            self.assertEqual(next(model.parameters()).device.type, 'cpu')

    def test_development_prefixes_stay_inside_documents(self):
        stream = DocumentStream(np.array([1,3,4,5,2,1,6,7,8,9,2,1,10,11,12,2]), 1, 2, 0)
        self.assertEqual(ev.prompts_from_stream(stream, 2, 2), [(0, [1,3,4]), (1, [1,6,7])])
        with self.assertRaises(ValueError): ev.prompts_from_stream(stream, 2, 4)

    def test_checkpoint_summary_provenance_restore_and_corruption(self):
        trainer = PairedExperimentTests().make()
        config = asdict(trainer.models[0].config); config['memory_layers'] = list(config['memory_layers'])
        plan = dict(seeds=[7, 9], data_seeds={'7':8, '9':10}, microbatch=2, length=8,
            target_budget=12, schedule_steps=3, warmup=1, learning_rate=1e-4,
            device='cpu', model=config, parent_sha256='parent')
        source = {'synthetic': True}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            metadata = dict(format='spora-p91-run-v1', source_review=source, seed=7,
                code={}, protected={}, torch=str(torch.__version__))
            (directory/'metadata.json').write_text(json.dumps(metadata), encoding='utf8')
            trainer.provenance = dict(source=source, code={}, protected={}, parent='parent',
                metadata_sha256=file_sha256(directory/'metadata.json'))
            for _ in range(3): trainer.step()
            trainer.save(directory/'checkpoint.pt')
            summary = dict(format='spora-p91-summary-v1', status='complete',
                checkpoint_sha256=file_sha256(directory/'checkpoint.pt'), steps=trainer.steps,
                targets=trainer.targets, seed=7, coverage=coverage(trainer.spans),
                batches=trainer.trace, updates=trainer.logs, evaluations=trainer.evaluations)
            sp = directory/'summary.json'; sp.write_text(json.dumps(summary), encoding='utf8')
            payload, _, hashes = ev.read_pair(directory, plan, source)
            restored = ev.restore_models(trainer.organism, trainer.state, payload)
            for old, new in zip(trainer.models, restored):
                for name, value in interface_state(old).items():
                    self.assertTrue(torch.equal(value, interface_state(new)[name]), name)
            payload['models'][0]['embedding.weight'][0,0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'Invalid model tensor'):
                ev.restore_models(trainer.organism, trainer.state, payload)
            summary['targets'] += 1; sp.write_text(json.dumps(summary), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'content differs'):
                ev.read_pair(directory, plan, source)
            summary['targets'] -= 1; summary['status'] = 'partial'; sp.write_text(json.dumps(summary), encoding='utf8')
            with self.assertRaises(ValueError): ev.read_pair(directory, plan, source)
            summary['status'] = 'complete'; sp.write_text(json.dumps(summary), encoding='utf8')
            with (directory/'checkpoint.pt').open('ab') as f: f.write(b'broken')
            with self.assertRaisesRegex(ValueError, 'does not match'): ev.read_pair(directory, plan, source)
            self.assertNotEqual(hashes['checkpoint'], file_sha256(directory/'checkpoint.pt'))

    def test_existing_training_lock_is_read_only_and_exclusive(self):
        from scripts.run_p91_paired import directory_lock
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); lock = p/'.p91.lock'; lock.write_bytes(b'\0')
            code = ('from pathlib import Path; from scripts.eval_p94_paired_generation import read_lock; '
                    'p=Path(__import__("sys").argv[1]);\nwith read_lock(p): print("acquired")')
            with directory_lock(p):
                child = subprocess.run([sys.executable, '-B', '-c', code, tmp], capture_output=True, text=True)
                self.assertNotEqual(child.returncode, 0); self.assertIn('locked', child.stderr)
            child = subprocess.run([sys.executable, '-B', '-c', code, tmp], capture_output=True, text=True)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(lock.read_bytes(), b'\0')


if __name__ == '__main__':
    unittest.main()
