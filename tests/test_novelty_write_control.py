import unittest

import torch

from spora.internal_memory import NoveltyWriteControl, WriteControl, attach, observe_sourced
from spora.state_codec import flatten, restore
from spora.temporal import TemporalConfig, TemporalSPORA
from spora.growing import new_state


class NoveltyWriteControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(90118)
        self.model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
                                                  expert_rank=4, memory_layers=(0, 1),
                                                  history_writer='rls', current_address='orthogonal'))
        self.state = attach(self.model, new_state(self.model), NoveltyWriteControl())

    def test_new_controller_has_versioned_config_and_input(self):
        self.assertEqual(self.model.write_control.config_dict()['kind'], 'novelty')
        self.assertEqual(self.model.write_control(torch.zeros(2, 11)).shape, (2, 5))
        with self.assertRaises(ValueError):
            self.model.write_control(torch.zeros(2, 10))

    def test_legacy_migration_preserves_zero_novelty_output(self):
        torch.manual_seed(90119)
        legacy = WriteControl()
        candidate = NoveltyWriteControl.from_legacy(legacy)
        features = torch.randn(5, 10)
        torch.testing.assert_close(legacy(features), candidate(torch.cat([features, torch.zeros(5, 1)], dim=1)),
                                   atol=0, rtol=0)

    def test_roundtrip_restores_novelty_controller(self):
        schema, tensors = flatten(self.model, self.state)
        restored, state = restore(schema, tensors)
        self.assertIsInstance(restored.write_control, NoveltyWriteControl)
        self.assertEqual(restored.write_control.config_dict(), self.model.write_control.config_dict())
        self.assertEqual(state.cursor, self.state.cursor)

    def test_observation_supplies_novelty_feature_and_preserves_guard(self):
        with torch.no_grad():
            for parameter in self.model.write_control.parameters():
                parameter.zero_()
            self.model.write_control.network[-1].bias[0] = 20
        state, result = observe_sourced(self.model, self.state, 2, 3, 'direct', 1.0)
        self.assertTrue(result['current_write'])
        self.assertEqual(result['novelty'], 1.0)
        self.assertEqual(state.cursor, 1)


if __name__ == '__main__':
    unittest.main()
