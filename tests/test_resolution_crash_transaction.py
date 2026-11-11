"""Crash-boundary check for the experimental full-checkpoint transaction.

The fact adjudication itself is stubbed: these tests exercise file commit and
publication boundaries, not provenance or neural recall quality.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts import deepseek_spora_runtime_chat as runtime
from spora.growing import new_state
from spora.integrated_checkpoint import load_integrated, save_integrated
from spora.integrated_open_base import ScaledOpenBaseMemory
from spora.temporal import TemporalConfig, TemporalSPORA
from tests.test_integrated_open_base import Base


def make_base():
    base = Base()
    base.config = SimpleNamespace(hidden_size=3, num_attention_heads=1)
    return base


def fake_record(module, model, state, key, event, *, value, valid_at, decision):
    module.records[0] = torch.tensor([key, event, valid_at, value, 0, 1])
    module.occupied[0] = True


def run_worker(root, boundary):
    torch.set_num_threads(2)
    root = Path(root)
    destination = root / 'working.pt'
    decoder, organism, state = load_integrated(destination, make_base(), root)
    chat = object.__new__(runtime.SporaChat)
    chat.decoder, chat.organism, chat.state = decoder, organism, state
    chat._working_sha256 = runtime.sha(destination)
    chat._replacements = lambda: {}
    runtime.WORKING_CHECKPOINT = destination
    runtime.MODEL = root
    marker = root / 'ready'
    original_replace = os.replace

    def controlled_replace(source, target):
        if boundary == 'after':
            original_replace(source, target)
        marker.write_text(boundary, encoding='utf-8')
        while True:
            time.sleep(0.1)

    with patch('spora.historical_resolutions.HistoricalResolutions.record', autospec=True,
               side_effect=fake_record), patch('spora.language_training.os.replace', side_effect=controlled_replace):
        chat.resolve_historical(1, 2, value=3, valid_at=1, decision='confirm')


class ResolutionCrashTransactionTests(unittest.TestCase):
    def test_kill_before_or_after_atomic_replace(self):
        for boundary in ('before', 'after'):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'model.safetensors').write_bytes(b'base weights')
                (root / 'config.json').write_text(json.dumps({'hidden_size': 3, 'num_hidden_layers': 3,
                                                             'num_attention_heads': 1, 'vocab_size': 9}), encoding='utf-8')
                for name in ('tokenizer.json', 'tokenizer_config.json'):
                    (root / name).write_text('{}', encoding='utf-8')
                organism = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
                                                        expert_rank=4, memory_layers=(0, 1),
                                                        history_writer='rls', current_address='orthogonal'))
                state = new_state(organism)
                state.cursor = 3
                destination = root / 'working.pt'
                save_integrated(destination, organism, state, ScaledOpenBaseMemory(3, 64, 0.5), root, (0, 2))
                before = runtime.sha(destination)
                creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(root), boundary],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
                try:
                    deadline = time.monotonic() + 40
                    while not (root / 'ready').exists() and child.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.1)
                    self.assertTrue((root / 'ready').exists(), 'worker did not reach requested commit boundary')
                finally:
                    if child.poll() is None:
                        child.terminate()
                    stdout, stderr = child.communicate(timeout=15)
                self.assertTrue(destination.exists())
                _, restored, restored_state = load_integrated(destination, make_base(), root)
                if boundary == 'before':
                    self.assertEqual(runtime.sha(destination), before)
                    self.assertFalse(hasattr(restored, 'historical_resolutions'))
                else:
                    self.assertNotEqual(runtime.sha(destination), before)
                    self.assertEqual(restored.historical_resolutions.lookup(1, 2)['decision'], 'confirm')
                    restored.historical_resolutions.validate(restored, restored_state)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        run_worker(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
