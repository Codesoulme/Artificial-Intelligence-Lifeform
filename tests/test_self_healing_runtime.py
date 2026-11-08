import tempfile
import unittest
from pathlib import Path
import torch

from spora.dna_shadow import DnaShadow, state_digest
from spora.event_ledger import EventLedger
from spora.health_gate import cases_digest, promotion_decision
from spora.awake import AwakeCore, attach_awake
from spora.awake_session import AwakeSession
from spora.temporal import TemporalSPORA, TemporalConfig
from spora.growing import new_state


class SelfHealingRuntimeTests(unittest.TestCase):
    def test_event_ledger_is_idempotent_and_hash_chained(self):
        with tempfile.TemporaryDirectory() as folder:
            with EventLedger.create(Path(folder) / 'events.sqlite') as ledger:
                first = ledger.append('observation', {'x': 1}, event_id='e1',
                                      idempotency_key='same', monotonic_ns=1, wall_ns=2)
                duplicate = ledger.append('observation', {'x': 999}, event_id='other',
                                          idempotency_key='same', monotonic_ns=3, wall_ns=4)
                self.assertFalse(first['duplicate'])
                self.assertTrue(duplicate['duplicate'])
                ledger.append('prediction', {'x': 2}, event_id='e2',
                              idempotency_key='e2', monotonic_ns=5, wall_ns=6)
                self.assertEqual(ledger.verify()['events'], 2)
                self.assertEqual([row['kind'] for row in ledger.events()], ['observation', 'prediction'])

    def test_dna_shadow_repairs_only_declared_damage_and_roundtrips(self):
        active = {'a': torch.tensor([1., 2., 3.]), 'b': torch.tensor([4.])}
        shadow = DnaShadow(active, alpha=.5)
        changed = {'a': torch.tensor([3., 2., 3.]), 'b': torch.tensor([4.])}
        shadow.update(changed)
        damaged = {'a': torch.tensor([0., 2., 0.]), 'b': torch.tensor([4.])}
        repaired, counts = shadow.repair_damaged(damaged)
        self.assertEqual(counts, {'a': 2, 'b': 0})
        torch.testing.assert_close(repaired['a'], torch.tensor([2., 2., 3.]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'dna.pt'
            shadow.save(path)
            restored = DnaShadow.load(path)
            self.assertEqual(state_digest(shadow.tensors), state_digest(restored.tensors))

    def test_health_gate_refuses_changed_holdout_and_regression(self):
        digest = cases_digest([{'prompt': 'a'}])
        self.assertFalse(promotion_decision({'loss': 1.}, {'loss': .9},
                                            expected_cases_digest=digest,
                                            candidate_cases_digest='changed',
                                            lower_is_better={'loss'})['promote'])
        result = promotion_decision({'loss': 1., 'chat': .8}, {'loss': 1.1, 'chat': .9},
                                    expected_cases_digest=digest,
                                    candidate_cases_digest=digest,
                                    lower_is_better={'loss'})
        self.assertFalse(result['promote'])
        self.assertEqual(result['regressions'], ['loss'])

    def test_awake_ticks_are_written_to_ledger(self):
        torch.manual_seed(19)
        model = TemporalSPORA(TemporalConfig(width=16, layers=2, heads=1,
            memory_width=8, expert_rank=2, memory_layers=(0, 1), history_writer='rls'))
        attach_awake(model, AwakeCore())
        state = new_state(model)
        with tempfile.TemporaryDirectory() as folder:
            with EventLedger.create(Path(folder) / 'events.sqlite') as ledger:
                session = AwakeSession(model, monotonic_ns=lambda: 100,
                                       utc_ns=lambda: 200, ledger=ledger)
                session.tick()
                self.assertEqual(ledger.verify()['events'], 1)
                self.assertEqual(ledger.events()[0]['kind'], 'awake.tick')
                self.assertEqual(state.cursor, 0)


if __name__ == '__main__':
    unittest.main()
