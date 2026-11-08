import contextlib
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models
from spora.conversation import ConversationBridge, ConversationConfig, restore_conversation
from spora.growing import new_state, save
from spora.language_training import interface_state
from spora.temporal import TemporalConfig, TemporalSPORA
from spora.state_codec import restore
from spora.language_training import frozen_digest
from scripts.run_p106_paired_dialogue import main, sha


class PairedDialogueTest(unittest.TestCase):
    def setup_case(self, root):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        organism = TemporalSPORA(TemporalConfig(width=16, layers=2, heads=2, memory_width=8,
                                                 expert_rank=4, memory_layers=(0, 1)))
        state = new_state(organism)
        individual = root / 'individual.pt'
        save(individual, organism, state)
        tokenizer = Tokenizer(models.WordLevel({str(i): i for i in range(1, 32)} | {'<|pad|>': 0}, unk_token='1'))
        vocab = root / 'tokenizer.json'
        tokenizer.save(str(vocab))
        cfg = ConversationConfig(vocabulary=32, width=16, depth=2, heads=2, hidden=24, context=8, memory_layers=(0, 1))
        model = ConversationBridge(organism, cfg)
        language = root / 'language.pt'
        torch.save(dict(format='spora-conversation-v1', organism_sha256=sha(individual),
                        tokenizer_sha256=sha(vocab), config=asdict(cfg), state=interface_state(model)), language)
        data = root / 'data'
        data.mkdir()
        for filename, short in [('context.jsonl', False), ('last-user.jsonl', True)]:
            with (data / filename).open('w') as stream:
                for i in range(6):
                    tokens = ([1, 2, 3, 4] if short else [1, 5, 6, 2, 3, 4])
                    row = dict(pair_id=str(i), group='large' if i < 4 else 'small',
                               tokens=tokens, mask=[False] * (len(tokens)-2) + [True, True])
                    stream.write(json.dumps(row) + '\n')
        (data / 'manifest.json').write_text(json.dumps(dict(format='matched-dialogue-v1', tokenizer_sha256=sha(vocab),
            files={name: sha(data / name) for name in ('context.jsonl', 'last-user.jsonl')})))
        return ['--individual', str(individual), '--language', str(language), '--tokenizer', str(vocab),
                '--data', str(data), '--length', '7', '--batch-size', '2', '--epochs', '2',
                '--max-per-group', '2', '--threads', '1']

    def invoke(self, args):
        with patch('sys.argv', ['run_p106_paired_dialogue.py'] + args), open_null() as sink, contextlib.redirect_stdout(sink):
            main()

    def assert_nested(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assert_nested(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested(x, y)
        else:
            self.assertEqual(a, b)

    def test_cpu_resume_and_interrupted_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self.setup_case(root)
            full, resumed = root / 'full', root / 'resumed'
            self.invoke(base + ['--output', str(full)])
            self.invoke(base + ['--output', str(resumed), '--until', '1'])
            original = torch.optim.AdamW.step
            calls = []
            def broken(opt, *args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError('simulated interrupt after first arm')
                return original(opt, *args, **kwargs)
            with patch.object(torch.optim.AdamW, 'step', broken), self.assertRaises(RuntimeError):
                self.invoke(base + ['--output', str(resumed), '--resume'])
            self.assertEqual(len(list(resumed.glob('step-*.pt'))), 1)
            self.invoke(base + ['--output', str(resumed), '--resume'])
            a = torch.load(full / 'step-00000004.pt', weights_only=True)
            b = torch.load(resumed / 'step-00000004.pt', weights_only=True)
            for key in ('sampler', 'arms', 'torch_rng', 'python_rng'):
                self.assert_nested(a[key], b[key])
            self.assert_nested(a['individual'], b['individual'])
            individual, memory = restore(a['individual']['schema'], a['individual']['tensors'], 'cpu')
            self.assertEqual(frozen_digest(individual, memory), a['contract']['frozen_digest'])
            for arm in a['arms']:
                restored = restore_conversation(individual, arm, a['contract']['inputs']['individual'],
                                                a['contract']['inputs']['tokenizer'])
                self.assert_nested(interface_state(restored), arm['state'])
            self.assertEqual(a['contract']['selected_groups'], {'large': 2, 'small': 2})
            self.assertEqual(a['logs'][-1]['group_exposures'], {'large': 4, 'small': 4})
            self.assertTrue(all(row['context_loss'] > 0 and row['last_user_loss'] > 0 for row in a['logs']))
            self.assertNotEqual(a['arms'][0]['state']['embedding.weight'].tolist(),
                                a['arms'][1]['state']['embedding.weight'].tolist())
            with self.assertRaises(FileExistsError):
                self.invoke(base + ['--output', str(full)])
            with self.assertRaises(ValueError):
                self.invoke(base + ['--output', str(resumed), '--resume', '--lr', '0.01'])


@contextlib.contextmanager
def open_null():
    import os
    with open(os.devnull, 'w') as stream:
        yield stream


if __name__ == '__main__':
    unittest.main()
