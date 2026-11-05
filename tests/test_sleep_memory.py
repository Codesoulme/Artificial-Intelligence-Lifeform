import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from spora.temporal import TemporalSPORA, TemporalConfig
from spora.growing import new_state, observe, grow, save, load, query, scalars
from spora.sleep_memory import SealedPage, train_page, propose, consolidate_store
from spora.durable import TensorStore
from spora.state_codec import flatten
from spora.language_bridge import LanguageBridge


class SleepMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(223)
        self.model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2,
            memory_width=16, expert_rank=4, memory_layers=(0, 1), history_writer='rls'))
        self.state = grow(self.model, new_state(self.model))
        for i in range(64):
            self.state = observe(self.model, self.state, i % 8, i % 16, int(i % 4 != 0))

    def test_learning_changes_only_independent_student(self):
        originals = {k: v.clone() for k, v in flatten(self.model, self.state)[1].items()}
        child, metrics = propose(self.model, self.state, 0, rank=4, steps=120)
        self.assertIsInstance(child.pages[0], SealedPage)
        self.assertTrue(all(b < a for a, b in zip(metrics['initial_mse'], metrics['final_mse'])))
        for k, v in flatten(self.model, self.state)[1].items():
            torch.testing.assert_close(v, originals[k], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertLess(scalars(child)['total'], scalars(self.state)['total'])
        with self.assertRaises(ValueError):
            propose(self.model, self.state, 1)

    def test_sealed_state_restarts_and_writes_only_to_active_page(self):
        child, _ = propose(self.model, self.state, 0, rank=4, steps=60)
        before = [[x.clone() for x in s] for s in child.pages[0].factors]
        continued = observe(self.model, child, 2, 9, 1)
        for a, b in zip(before, continued.pages[0].factors):
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'person.pt'; database = Path(temp)/'person.sqlite'
            save(file, self.model, continued)
            model, restored, _ = load(file)
            with TensorStore.create(database, model, restored) as store:
                other, recovered, _, _ = store.read()
                k = torch.tensor([2, 3, 2]); m = torch.tensor([0, 1, 1]); t = torch.tensor([0, 3, 64])
                with torch.no_grad():
                    a = query(self.model, continued, k, m, t)
                    torch.testing.assert_close(a, query(model, restored, k, m, t), atol=0, rtol=0)
                    torch.testing.assert_close(a, query(other, recovered, k, m, t), atol=0, rtol=0)
                    bridge = LanguageBridge(other)
                    self.assertTrue(torch.isfinite(bridge(torch.randint(0,256,(1,8)), recovered)).all())

    def test_rejected_student_never_commits(self):
        with tempfile.TemporaryDirectory() as temp:
            with TensorStore.create(Path(temp)/'person.sqlite', self.model, self.state) as store:
                with patch('spora.sleep_memory.behavior_gate', return_value={'passed': False}):
                    result = consolidate_store(store, self.model, self.state, 0, 0, rank=4, steps=20)
                self.assertFalse(result['accepted']); self.assertEqual(store.revision, 0)
                _, state, _, revision = store.read()
                self.assertIsInstance(state.pages[0], list); self.assertEqual(revision, 0)

    def test_precision_packing_preserves_novelty_geometry(self):
        page, _ = train_page(self.model, self.state.pages[0], rank=16, method='svd', steps=1)
        for site in range(2):
            matrix, precision = page.materialize(site)
            torch.testing.assert_close(precision, self.state.pages[0][site][1], atol=0, rtol=0)
            torch.testing.assert_close(matrix, self.state.pages[0][site][0], atol=2e-5, rtol=2e-5)
        with self.assertRaises(ValueError):
            observe(self.model, type(self.state)(self.state.current, [page], 0), 1, 2, 1)


if __name__ == '__main__':
    unittest.main()
