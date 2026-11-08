import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from spora.conversation import ConversationConfig
from spora.document_stream import DocumentStream
from spora.growing import new_state
from spora.paired_experiment import PairedTrainer, coverage
from spora.temporal import TemporalConfig, TemporalSPORA


class PairedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make(self, provenance=None):
        torch.manual_seed(42)
        organism = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2,
            memory_width=64, expert_rank=4, memory_layers=(0, 1),
            history_writer='rls', current_address='orthogonal'))
        state = new_state(organism)
        config = ConversationConfig(vocabulary=64, width=32, depth=2, heads=4,
            hidden=64, context=16, memory_layers=(0, 1))
        docs = DocumentStream(np.array([1,3,4,5,2,1,6,7,8,9,2,1,10,11,2]), 1, 2, 0)
        return PairedTrainer(organism, state, config, docs, seed=7, data_seed=8,
            provenance=provenance or {'source':'synthetic'}, batch_size=2, length=8,
            budget=12, schedule_steps=3, warmup=1)

    def test_recovery_is_identical_and_batch_coverage(self):
        a = self.make(); b = self.make()
        a.step(); a.step()
        with tempfile.TemporaryDirectory() as d:
            b.step(); p = Path(d)/'pair.pt'; b.save(p)
            c = self.make(); c.restore(p); c.step()
        self.assertEqual(a.trace, c.trace)
        self.assertEqual(a.targets, coverage(a.spans)['presentations'])
        self.assertEqual(a.rng.getstate(), c.rng.getstate())
        for left, right in zip(a.models, c.models):
            for key, value in left.state_dict().items():
                self.assertTrue(torch.equal(value, right.state_dict()[key]), key)
        evaluation = c.evaluate(c.documents)
        self.assertEqual(evaluation['arms'][0]['targets'], evaluation['arms'][1]['targets'])

    def test_reject_wrong_origin_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'pair.pt'; a = self.make(); a.step(); a.save(path)
            with self.assertRaises(ValueError):
                self.make({'source':'other'}).restore(path)
            envelope = torch.load(path, weights_only=True)
            envelope['payload'] += b'broken'
            torch.save(envelope, path)
            with self.assertRaises(ValueError):
                self.make().restore(path)

    def test_protection(self):
        a = self.make()
        with torch.no_grad():
            next(a.organism.parameters()).add_(1)
        with self.assertRaises(ValueError):
            a.step()

    def test_individual_module_modes_survive_training_and_evaluation(self):
        fixture = self.make()
        modules = list(fixture.organism.modules())
        for index, module in enumerate(modules):
            module.training = bool(index % 2)
        before = [module.training for module in modules]
        trainer = PairedTrainer(fixture.organism, fixture.state, fixture.models[0].config,
            fixture.documents, seed=7, data_seed=8, provenance={'source':'synthetic'},
            batch_size=2, length=8, budget=12, schedule_steps=3, warmup=1)
        self.assertEqual(before, [module.training for module in modules])
        trainer.step()
        self.assertEqual(before, [module.training for module in modules])
        trainer.evaluate(trainer.documents)
        self.assertEqual(before, [module.training for module in modules])

    def test_failed_atomic_replacement_preserves_previous_checkpoint(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pair.pt'
            trainer = self.make(); trainer.step(); trainer.save(path)
            original = path.read_bytes()
            trainer.step()
            with patch('spora.language_training.os.replace', side_effect=OSError('interrupted replace')):
                with self.assertRaises(OSError): trainer.save(path)
            self.assertEqual(path.read_bytes(), original)
            restored = self.make(); restored.restore(path)
            self.assertEqual(restored.steps, 1)


if __name__ == '__main__':
    unittest.main()
