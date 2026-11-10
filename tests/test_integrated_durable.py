import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from spora.durable import RevisionConflict
from spora.growing import new_state
from spora.integrated_checkpoint import save_integrated
from spora.integrated_durable import create_integrated_durable, load_integrated_durable
from spora.integrated_open_base import NativeSPORAMemoryReader
from spora.internal_memory import WriteControl, attach
from spora.language_training import frozen_digest
from spora.structure_transfer import SPORALayerReplacement
from spora.temporal import TemporalConfig, TemporalSPORA
from tests.test_integrated_open_base import Base


class IntegratedDurableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'model.safetensors').write_bytes(b'base weights')
        (self.root/'config.json').write_text('{"hidden_size":3,"num_hidden_layers":3,"num_attention_heads":1}', encoding='utf-8')
        for name in ('tokenizer.json', 'tokenizer_config.json'):
            (self.root/name).write_text('{}', encoding='utf-8')
        organism = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
                                  expert_rank=4, memory_layers=(0, 1),
                                  history_writer='rls', current_address='orthogonal'))
        state = new_state(organism)
        control = WriteControl()
        with torch.no_grad():
            for param in control.parameters():
                param.zero_()
            control.network[-1].bias[0] = 20
        state = attach(organism, state, control)
        reader = NativeSPORAMemoryReader(3, 64, 0.5)
        layer = SPORALayerReplacement(3, 1, max_length=8, layer_idx=1)
        self.bootstrap = self.root/'bootstrap.pt'
        self.database = self.root/'individual.sqlite'
        save_integrated(self.bootstrap, organism, state, reader, self.root, (0, 2), replacements={1: layer})

    def tearDown(self):
        self.temp.cleanup()

    def base(self):
        model = Base()
        model.config = SimpleNamespace(hidden_size=3, num_attention_heads=1)
        return model

    def create(self):
        return create_integrated_durable(self.database, self.bootstrap, self.base(), self.root)

    def open(self):
        return load_integrated_durable(self.database, self.bootstrap, self.base(), self.root)

    def test_incremental_write_and_reopen_preserve_structure_and_state(self):
        with self.create() as runtime:
            decision, stats = runtime.observe_sourced(4, 2, 'direct', 1.0)
            self.assertTrue(decision['current_write'])
            self.assertEqual(stats['revision'], 1)
            digest = frozen_digest(runtime.organism, runtime.state)
            self.assertLess(stats['tensor_bytes_written'], self.bootstrap.stat().st_size)
        with self.open() as restored:
            self.assertEqual(restored.state.cursor, 1)
            self.assertEqual(frozen_digest(restored.organism, restored.state), digest)
            self.assertIsInstance(restored.decoder.base.model.layers[1], SPORALayerReplacement)

    def test_failed_transaction_does_not_publish_state(self):
        with self.create() as runtime:
            digest = frozen_digest(runtime.organism, runtime.state)
            def fail(stage):
                if stage == 'after_first_tensor':
                    raise RuntimeError('injected failure')
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                runtime.observe_sourced(4, 2, 'direct', 1.0, hook=fail)
            self.assertEqual(runtime.state.cursor, 0)
            self.assertEqual(frozen_digest(runtime.organism, runtime.state), digest)
        with self.open() as restored:
            self.assertEqual(restored.state.cursor, 0)
            self.assertEqual(frozen_digest(restored.organism, restored.state), digest)

    def test_failure_after_commit_publishes_committed_state(self):
        with self.create() as runtime:
            def fail(stage):
                if stage == 'after_commit':
                    raise RuntimeError('injected post-commit failure')
            with self.assertRaisesRegex(RuntimeError, 'post-commit'):
                runtime.observe_sourced(4, 2, 'direct', 1.0, hook=fail)
            self.assertEqual(runtime.state.cursor, 1)
            self.assertEqual(runtime.store.revision, 1)
        with self.open() as restored:
            self.assertEqual(restored.state.cursor, 1)

    def test_stale_writer_is_rejected_without_publishing_state(self):
        with self.create() as first, self.open() as second:
            first.observe_sourced(4, 2, 'direct', 1.0)
            with self.assertRaises(RevisionConflict):
                second.observe_sourced(5, 3, 'direct', 1.0)
            self.assertEqual(second.state.cursor, 0)

    def test_decoder_mutation_and_bootstrap_change_block_commit(self):
        with self.create() as runtime:
            layer = runtime.decoder.base.model.layers[1]
            with torch.no_grad():
                layer.position.weight[0, 0] += 1
            with self.assertRaisesRegex(ValueError, '[Dd]ecoder'):
                runtime.observe_sourced(4, 2, 'direct', 1.0)
            self.assertEqual(runtime.state.cursor, 0)
        with self.open() as runtime:
            with self.bootstrap.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'bootstrap'):
                runtime.observe_sourced(4, 2, 'direct', 1.0)
            self.assertEqual(runtime.state.cursor, 0)

    def test_changed_bootstrap_is_rejected_before_modifying_base(self):
        with self.create():
            pass
        other = self.root/'other-bootstrap.pt'
        other.write_bytes(self.bootstrap.read_bytes() + b'changed')
        base = self.base()
        previous = list(base.model.layers)
        with self.assertRaisesRegex(ValueError, 'do not match'):
            load_integrated_durable(self.database, other, base, self.root)
        self.assertTrue(all(a is b for a, b in zip(base.model.layers, previous)))

    def test_process_termination_before_and_after_commit(self):
        code = r'''
import os, sys
from pathlib import Path
from types import SimpleNamespace
from tests.test_integrated_open_base import Base
from spora.integrated_durable import load_integrated_durable
root = Path(sys.argv[1])
base = Base()
base.config = SimpleNamespace(hidden_size=3, num_attention_heads=1)
runtime = load_integrated_durable(root/'individual.sqlite', root/'bootstrap.pt', base, root)
def stop(stage):
    if stage == sys.argv[2]: os._exit(42)
runtime.observe_sourced(4, 2, 'direct', 1.0, hook=stop)
os._exit(0)
'''
        with self.create() as runtime:
            initial = frozen_digest(runtime.organism, runtime.state)
        failed = subprocess.run([sys.executable, '-c', code, str(self.root), 'after_first_tensor'],
                                capture_output=True, timeout=60)
        self.assertEqual(failed.returncode, 42, failed.stderr.decode(errors='replace'))
        with self.open() as restored:
            self.assertEqual(restored.state.cursor, 0)
            self.assertEqual(frozen_digest(restored.organism, restored.state), initial)
        committed = subprocess.run([sys.executable, '-c', code, str(self.root), 'after_commit'],
                                   capture_output=True, timeout=60)
        self.assertEqual(committed.returncode, 42, committed.stderr.decode(errors='replace'))
        with self.open() as restored:
            self.assertEqual(restored.state.cursor, 1)
            self.assertEqual(restored.store.revision, 1)


if __name__ == '__main__':
    unittest.main()
